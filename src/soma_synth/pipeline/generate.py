"""Launch a source's generator: the step ``runner.PipelineRunner`` calls for ``generate``.

The runner and the CLI both launch generators through this module, so neither imports the other
(ADR-0041). It builds the argv from the registry, inherits the environment, fans shards out and
merges them, and gives a string ``--generate-cmd`` to the shell.

What this module launches is the bundle entrypoint. The retarget-corpus stage and the post-steps a
registry v2 source declares are the runner's to orchestrate (``runner._step_generate``): they run
before and after this call, and a caller of ``run_generate`` alone gets the bundle entrypoint
alone, with the corpus it names through ``extra_args``.

**Logs.** Given a log file (``run_command``'s ``log_path``) or folder (``run_generate``'s
``log_dir``), a child's stdout and stderr are teed: copied as they come to this process's own
(the console still shows them) and appended to the log, which starts with the command line and
ends with the exit code. The runner hands its record's ``logs/`` (:data:`GENERATE_LOG`, one
``generate.shard<i>of<N>.log`` per shard). A Python child is asked for UTF-8 and unbuffered output
(``PYTHONIOENCODING``, ``PYTHONUNBUFFERED``, unless the environment sets them), so its progress
reaches the console as it is printed and the log reads the same on every PC. Without a log the
child inherits the console, as it always did.
"""

from __future__ import annotations

import codecs
import json
import os
import subprocess
import sys
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path

#: ``cli.py`` resolved this as ``Path(cli.__file__).parents[2]``; from ``pipeline/`` it is one
#: level further up. The same directory either way.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

#: What every bundle's INDEX.json must say about itself (ADR-0041): the one class the standing
#: decision covers and the one scope.
EXPECTED_ARTIFACT_CLASS = "experimental_non_candidate"
EXPECTED_DISTRIBUTION_SCOPE = "internal_only"

#: The bundle entrypoint's log in the folder ``run_generate`` is given (its merge step appends to
#: it); a sharded run writes one ``generate.shard<i>of<N>.log`` per shard beside it.
GENERATE_LOG = "generate.log"
#: How long a finished child's output is still read: a grandchild that inherited its pipes and
#: outlived it would otherwise keep the step waiting.
_DRAIN_S = 30.0


def _echo(stream, text: str) -> None:
    """Write ``text`` to a console stream, replacing what its encoding cannot show."""
    if not text or stream is None:
        return
    try:
        stream.write(text)
    except UnicodeEncodeError:
        encoding = getattr(stream, "encoding", None) or "ascii"
        stream.write(text.encode(encoding, "replace").decode(encoding, "replace"))
    try:
        stream.flush()
    except (OSError, ValueError):
        pass


class _Teed:
    """A child process whose stdout and stderr go to this process's own streams and, as they come,
    to the end of a log file (both into one file, in the order they arrive)."""

    def __init__(self, argv, *, env: Mapping[str, str] | None, log_path: Path, shell: bool = False):
        child_env = dict(os.environ if env is None else env)
        child_env.setdefault("PYTHONIOENCODING", "utf-8")
        child_env.setdefault("PYTHONUNBUFFERED", "1")
        name = child_env["PYTHONIOENCODING"].split(":", 1)[0].strip() or "utf-8"
        try:
            self._encoding = codecs.lookup(name).name
        except LookupError:
            self._encoding = "utf-8"
        self._lock = threading.Lock()
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(log_path, "ab")
        command = argv if isinstance(argv, str) else " ".join(str(part) for part in argv)
        self._write_log(f"$ {command}\n".encode("utf-8"))
        try:
            self.process = subprocess.Popen(argv, env=child_env, shell=shell,
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except BaseException:
            self._log.close()
            raise
        self._pumps = [threading.Thread(target=self._pump, args=(pipe, console), daemon=True)
                       for pipe, console in ((self.process.stdout, "stdout"),
                                             (self.process.stderr, "stderr"))]
        for pump in self._pumps:
            pump.start()

    def _write_log(self, data: bytes) -> None:
        with self._lock:
            if not self._log.closed:
                self._log.write(data)
                self._log.flush()

    def _pump(self, pipe, console: str) -> None:
        decoder = codecs.getincrementaldecoder(self._encoding)(errors="replace")
        try:
            while True:
                chunk = pipe.read1(1 << 16)
                if not chunk:
                    break
                self._write_log(chunk)
                _echo(getattr(sys, console), decoder.decode(chunk))
            _echo(getattr(sys, console), decoder.decode(b"", final=True))
        except (OSError, ValueError):
            pass
        finally:
            pipe.close()

    def poll(self):
        return self.process.poll()

    def terminate(self) -> None:
        self.process.terminate()

    def kill(self) -> None:
        self.process.kill()

    def wait(self, timeout: float | None = None) -> int:
        code = self.process.wait(timeout=timeout)
        for pump in self._pumps:
            pump.join(_DRAIN_S)
        self._write_log(f"[exit {code}]\n".encode("utf-8"))
        with self._lock:
            self._log.close()
        return code


def _stop(processes) -> None:
    """Terminate every started process that is still running and wait for each: settling a failed
    run renames directories the processes write into by path."""
    for proc in processes:
        if proc.poll() is None:
            proc.terminate()
    for proc in processes:
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def _run_teed(argv, *, env: Mapping[str, str] | None, log_path: Path, shell: bool = False) -> int:
    child = _Teed(argv, env=env, log_path=log_path, shell=shell)
    try:
        return child.wait()
    except BaseException:
        _stop([child])
        raise


def _refuse_paired_weargait(source: str) -> bool:
    if source.casefold().replace("-", "_") != "weargait_pd":
        return False
    print(
        "WearGait-PD is estimated kinematics, not a paired lineage. Its estimated pipeline stays in "
        "the parent project: use that project's CLI, 'estimated --source weargait_pd --stage "
        "preflight|verify|refresh|summarise'.",
        file=sys.stderr,
    )
    return True


def _entrypoint(source: str, registry=None):
    """The registry's entrypoint for ``source``, or None when the registry does not know it."""
    from soma_synth.pipeline import stages as pipeline_stages

    try:
        return (registry or pipeline_stages.default_registry()).for_source(source).entrypoint
    except pipeline_stages.PipelineError:
        return None


def _default_generate_cmd(source: str, dataset_dir: Path, data_root, *, extra_args=(),
                          registry=None):
    """The generator argv for a source, from the registry rather than from a branch here.

    Returns None for a source the registry does not know, because the caller's contract is "None
    means pass --generate-cmd or --skip-generate" and an unknown source is exactly that situation.
    """
    entry = _entrypoint(source, registry)
    if entry is None:
        return None
    return entry.argv(
        sys.executable, _REPOSITORY_ROOT, Path(dataset_dir), Path(data_root) if data_root else None,
        extra_args,
    )


def plan_generate_commands(source: str, dataset_dir, data_root, jobs: int = 1, *,
                           extra_args: Sequence[str] = (), registry=None):
    """What `run_generate` will spawn for a registered source: (concurrent argvs, follow-up argv).

    Pure, so a test can see the fan-out without running a generator. None when the registry does
    not know the source. ``extra_args`` follow the output argument on every command line.
    """
    entry = _entrypoint(source, registry)
    if entry is None:
        return None
    return entry.plan(
        sys.executable, _REPOSITORY_ROOT, Path(dataset_dir), Path(data_root) if data_root else None,
        jobs=jobs, extra_args=extra_args,
    )


def run_command(argv: Sequence[str], *, env: Mapping[str, str] | None = None,
                log_path: Path | str | None = None) -> int:
    """Run one command line to completion and return its exit code: the corpus entrypoint and the
    post-steps go through here, so a test can replace one seam. With ``log_path`` its output is
    teed into that file as well (the module docstring)."""
    print("   running: " + " ".join(str(c) for c in argv))
    argv = [str(c) for c in argv]
    if log_path is not None:
        return _run_teed(argv, env=env, log_path=Path(log_path))
    return subprocess.run(argv, env=env, check=False).returncode


def _run_one(argv, *, env, log_dir: Path | None, shell: bool = False) -> int:
    """One process of the bundle entrypoint to completion, teed into ``<log_dir>/generate.log``
    when a log folder is given."""
    if log_dir is not None:
        return _run_teed(argv, env=env, log_path=Path(log_dir) / GENERATE_LOG, shell=shell)
    if shell:
        return subprocess.run(argv, shell=True, env=env, check=False).returncode
    return subprocess.run(argv, env=env, check=False).returncode


def run_generate(source: str, dataset_dir, data_root, generate_cmd, jobs: int = 1, *,
                 extra_args: Sequence[str] = (), env: Mapping[str, str] | None = None,
                 registry=None, log_dir: Path | str | None = None) -> int:
    """Run the source generator and block until it finishes. Split out as a seam so the pipeline is
    testable without spawning a real generator.

    `jobs` above one reaches the generator through the registry's parallel_args: a generator with
    its own pool gets the flag, a shardable one is run as `jobs` concurrent processes over disjoint
    slices and then once more to merge. The first non-zero exit is the step's exit.

    ``extra_args`` go after the output argument (the corpus flag and a sample run's selection),
    ``env`` is the child environment (None inherits this process's), and ``registry`` replaces the
    default registry. An explicit ``generate_cmd`` runs as given and ignores ``extra_args``.
    ``log_dir`` tees every process's output into logs there (:data:`GENERATE_LOG`; a shard's into
    ``generate.shard<i>of<N>.log``).
    """
    if _refuse_paired_weargait(source):
        return 2
    if generate_cmd:
        if isinstance(generate_cmd, str):
            print(f"   running (shell): {generate_cmd}")
            return _run_one(generate_cmd, env=env, log_dir=log_dir, shell=True)
        print("   running: " + " ".join(str(c) for c in generate_cmd))
        return _run_one(generate_cmd, env=env, log_dir=log_dir)

    planned = plan_generate_commands(source, dataset_dir, data_root, jobs=jobs,
                                     extra_args=extra_args, registry=registry)
    if planned is None:
        print(f"   no default generator for source '{source}' -- pass --generate-cmd or --skip-generate")
        return 2
    concurrent, follow_up = planned
    if len(concurrent) == 1:
        print("   running: " + " ".join(str(c) for c in concurrent[0]))
        code = _run_one(concurrent[0], env=env, log_dir=log_dir)
    else:
        print(f"   running {len(concurrent)} shards: " + " ".join(str(c) for c in concurrent[0][:-2])
              + " <shard i/N> ...")
        procs: list = []
        count = len(concurrent)
        try:
            for index, argv in enumerate(concurrent, 1):
                procs.append(
                    _Teed(argv, env=env,
                          log_path=Path(log_dir) / f"generate.shard{index}of{count}.log")
                    if log_dir is not None else subprocess.Popen(argv, env=env))
            codes = [p.wait() for p in procs]
        except BaseException:
            # Settling a failed run renames directories the shards write into by path, so every
            # started shard has to be gone before this propagates.
            _stop(procs)
            raise
        code = next((c for c in codes if c != 0), 0)
        print(f"   shards exited {codes}")
    if code != 0 or follow_up is None:
        return code
    print("   running: " + " ".join(str(c) for c in follow_up))
    return _run_one(follow_up, env=env, log_dir=log_dir)


def bundle_class_problems(dataset_dir, *, artifact_class: str = EXPECTED_ARTIFACT_CLASS,
                          distribution_scope: str = EXPECTED_DISTRIBUTION_SCOPE) -> list[str]:
    """What is wrong with the class a finished bundle declares, empty when nothing is.

    ADR-0041 lets a run make ``experimental_non_candidate`` / ``internal_only`` bundles and nothing
    else. Every generator writes both fields at the top of INDEX.json; a bundle without the file,
    or one that says anything else, fails the generate step rather than going on to validation
    and registration as if it were covered by the decision.
    """
    index_path = Path(dataset_dir) / "INDEX.json"
    if not index_path.is_file():
        return [f"{index_path} does not exist, so the bundle's artifact class cannot be confirmed"]
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return [f"{index_path} does not parse: {type(error).__name__}: {error}"]
    if not isinstance(index, dict):
        return [f"{index_path} is not a JSON object"]
    problems = []
    for key, expected in (("artifact_class", artifact_class),
                          ("distribution_scope", distribution_scope)):
        if index.get(key) != expected:
            problems.append(f"INDEX.json {key} is {index.get(key)!r}, not {expected!r}")
    return problems

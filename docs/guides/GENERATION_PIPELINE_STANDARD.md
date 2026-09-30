# Generation pipeline standard — execution specification

| Item | Value |
|---|---|
| Document ID | `SOMA-GENERATION-PIPELINE-STANDARD-001` |
| Version | `1.1.1` |
| Revision history | `1.0.0` (2026-09-09) first issue. `1.1.0` (2026-09-25) applies `ADR-0041` (parent project: `docs/adr/ADR-0041-soma-synth-split-and-teammate-generation.md`): registry v2 (corpus stage, post-steps, generation policy, `selection_args`); §1.3 every generation writes into an empty folder and there is no resume (a non-empty folder needs `--replace-existing`, which moves it aside and builds in a fresh folder: on success only a production folder's old generation is kept and everything else is deleted, on failure it is restored; no `force_arg`; the declared trade-off that a dead run is rebuilt from scratch), a `_superseded` evidence requirement for replacing a production folder that holds a finished generation, the generate-stage lock (`<folder>.lock`), refusal of read-only and evidence folders, refusal of other production folders, refusal of arguments that have nowhere to go, the `.generating` marker of an unfinished generate stage, and the same refusals in `pipeline`, `validate`, `readme`, `register` and push; §2.1 installation; §3.1 `resolved_config_sha256` filled in; §4.1 the tracing record and its gaps; §6 identity; §9 generation policy (`pipeline` generate refused, the same conditions on every PC: decision of 2026-09-25); §10 two mismatch rows. `1.1.1` (2026-09-30) translated into English; §11 operator reference (exit codes, common refusals, locks, leftovers) moved here from the README |
| Status | `ACTIVE_BASELINE` |
| Baseline date | `2026-09-25` |
| Authority | The **execution specification** of §5.1, §10.1 and §10.2 of the sealed document `PIPELINE_GOVERNANCE.md` (parent project: `docs/guides/PIPELINE_GOVERNANCE.md`). It does not amend that document and releases no hold. The basis for generating is not this document but the decision record that §9 names |
| Parent documents | `PIPELINE_GOVERNANCE.md` (parent project: `docs/guides/PIPELINE_GOVERNANCE.md`), `MASTERPLAN.md` (parent project: `docs/architecture/MASTERPLAN.md`) |
| Machine-readable twin | [`configs/datasets/source_pipelines_v2.yaml`](../../configs/datasets/source_pipelines_v2.yaml) (default). v1 [`source_pipelines_v1.yaml`](../../configs/datasets/source_pipelines_v1.yaml) is a configuration that was used, so it stays as it is and can still be read |
| Companion document | `ADR-0040` (parent project: `docs/adr/ADR-0040-dataset-profiles-registry-and-envelope-conformance.md`) — the output-side specification |
| Basis | `ADR-0041` (parent project: `docs/adr/ADR-0041-soma-synth-split-and-teammate-generation.md`) |

> **This document is not a new policy.** It records **how** the requirements already in
> `PIPELINE_GOVERNANCE.md` are met. That document is a sealed file, SHA-pinned by the M0 evaluator, so it
> is not edited here. If the two disagree, follow the sealed document and fix this one.
>
> **This repository.** This document lives in the soma-synth repository. It moved here from the parent
> project together with the generation pipeline at the 2026-09-25 split. The sealed documents
> (`PIPELINE_GOVERNANCE.md`, `MASTERPLAN.md`, `LOCAL_DATA_PLANE.md`), the ADRs, the decision records in
> `research/decisions` and the retention rules stay in the parent project. This document does not link to
> them; it writes their path relative to the parent project root as text, as in "parent project: `<path>`",
> because a relative link from this document to a parent-project file does not open even when this
> repository is mounted in the parent project as `packages/soma-synth`. Links point only at files in this
> repository, and `src/soma_synth/`, `scripts/`, `configs/`, `tests/` and `packages/smpl18` are relative to
> this repository. The retention rules are carried over with the same numbers and meaning in
> [`RETENTION_RULES.md`](RETENTION_RULES.md) (if they disagree with the parent project's rules, those rules
> win). "retention rule 1.x" in this repository's messages is the rule with that number.

---

## 0. Why this document is needed

Since 2026-08-03 `PIPELINE_GOVERNANCE.md` has required a seven-stage adapter split in §5.1, the seven
run-directory artifacts in §10.1 and ten provenance items in §10.2. Measured on 2026-09-09:

| Requirement | Measured |
|---|---|
| §5.1 seven-stage split | Not implemented. Each of 11 generators has its own flow |
| §10.1 seven run-directory artifacts | **None** in all 5 bundles |
| §10.2 ten provenance items | Only **4 items** present |
| §10.1 `run_id` = hash of config, code and source | `run_id` is a take label (`gaitex-austra-gwo-unified8-marker`) |

**No bundle can be reproduced from what was recorded.** Neither the code revision nor the SMPL model
that made it is known. This document and the code that goes with it fill that gap.

### 0.1 Why this was not implemented earlier

It was not a lack of will; **there was nowhere to implement it.** The generators do not import shared
code; they load it at run time by file path (`spec_from_file_location`, 12 files). One generator is
another generator's library — `generate_amass_faithful_all.py` reads `generate_amass_faithful.py` by path
and runs it. There is no import graph to follow and no module boundary to hang stages on.

This document **does not tear that structure out now**. `amass` and `prism` are the code that made the
two bundles that currently pass validation; if their output changed in a move, reproducibility would be
lost. Instead the runner **wraps** the existing entry points and supplies what the standard requires from
the outside.

---

## 1. Canonical stages

The first seven use the names in `PIPELINE_GOVERNANCE.md` §5.1 as they are. The last eight are the
synthesis and emission this lineage adds after them.

| # | Stage | What it does | Where it is today |
|---|---|---|---|
| 1 | `register_asset` | Register the source asset, obtain its SHA-256 | `SourceIdentity.asset_sha256` |
| 2 | `inspect_native` | Check shape and counts without decoding | The loop in each generator's `main()` |
| 3 | `validate_license` | Check terms of use and gates | `configs/gates/source_gates_v1_3.yaml` (outside the generators) |
| 4 | `load_native` | Read the native format | Start of each generator's `convert()` |
| 5 | `validate_native` | Source QC | Partial, differs per source |
| 6 | `convert_intermediate` | SMPL-24 retarget | Separate corpus generation stage — `corpus` in registry v2, run by the runner (§1.3) |
| 7 | `emit_source_qc_and_provenance` | Emit source QC and provenance | `SourceIdentity` (31 fields) |
| 8 | `fit_shape` | betas, rest skeleton | `addbio_retarget.shape_fit` |
| 9 | `build_large` | 18-joint kinematics | `unified8_emit.build_bundle`; the reduction is `pipeline.reduced_model` (§2.2) |
| 10 | `build_anthro` | rest constants | `unified8_emit.build_bundle` |
| 11 | `synthesise_small` | 8-channel IMU | **The only stage that differs per source** |
| 12 | `resample_100hz` | 100 Hz grid | `unified8_emit.build_bundle` |
| 13 | `emit_take` | Five outputs + take manifest | `unified8_emit.write_bundle` |
| 14 | `emit_bundle` | INDEX, description, README | `unified8_emit.write_data_description` |
| 15 | `close_run` | Finalise the seven run-directory artifacts | **None — introduced by this standard** |

### 1.1 Only one stage differs per source

Only `synthesise_small` takes a different path depending on `small_mode`.

| `small_mode` | Source | Path |
|---|---|---|
| `measured_physical` | prism | Values measured by the worn sensors |
| `synthetic_from_smpl` | amass · hknu · addbiomechanics | SMPL forward kinematics |
| `synthetic_from_markers` | gaitex | Rigid-body fit of marker clusters (ADR-0039) |

The other fourteen stages do the same work for every source. **That is where standardisation is possible.**

### 1.2 `unified8_emit` is already the common spine

`scripts/poc/unified8_emit.py` provides `build_bundle`, `write_bundle`, `write_data_description` and
`SourceIdentity` (31 fields), and `addbiomechanics`, `gaitex` and `hknu` use it. `amass` and `prism` do not —
they came first.

That module's docstring already records why: `generate_amass_faithful.build` was not refactored because
"that function builds two catalogued L4-PASS datasets and cannot be put at risk for convenience"; instead
an **equivalence test** (`tests/poc/test_unified8_emit.py`, 6 cases) runs both paths on the same SMPL input
and compares the arrays.

**This standard adopts that judgement rather than reversing it.** The condition for unification is "prove
equivalence first", and the means for that already exist.

### 1.3 Generation is corpus → bundle → post-steps (registry v2, 2026-09-25)

`hknu`, `gaitex` and `addbiomechanics` first turn the source into an SMPL-24 retarget corpus, and the
bundle is built from that corpus. Until now the corpus was a given input. A new PC has no corpus, and an
existing corpus can be deleted. So registry v2 declares a `corpus` block for each source (entry point,
lineage, output flag, fingerprint file, the flag passed to the bundle) and the runner runs it.

| Order | What | When |
|---|---|---|
| 1 | Corpus entry point → `<data root>/runs/experimental_generation_poc_demo/<corpus lineage>` (or `--corpus`) | When that folder holds no finished corpus (fingerprint file, no `.generating`), or with `--rebuild-corpus`. A finished corpus is reused (read only) |
| 2 | Bundle entry point, given the corpus through the declared flag (`--paired`, `--retarget`, `--raw`) | Always |
| 3 | Post-steps (`post_steps`) — `prism`'s `synthesize_insole_heading.py <bundle>` | After the bundle is finished, before `KNOWN_LIMITATIONS.json` is rendered |

**Every generation writes into an empty folder. There is no resume.** The bundle and corpus entry points
always write into a folder that is absent or empty. The runner neither overwrites a folder in place nor
continues an unfinished one (see "A replacement is built in a fresh folder" below). So when a run of
several hours dies or stops, it is **rebuilt from the start.** That is the declared trade-off. Takes from
two runs (or two corpora) never mix in one bundle, and the runner does not need to know each generator's
resume predicate (§7) or the losses of resuming (`prism` records skipped takes as `"skipped"` and drops
them from the index's `ok`); in exchange, the time of an interrupted long run is lost. An operator can
still pass a generator's own `--resume` or `--force` with `--corpus-arg` or `--bundle-arg`, but an empty
folder has nothing to skip. The runner refuses the following before running.

| Refusal | Condition | What clears it |
|---|---|---|
| Non-empty bundle folder | The bundle folder exists and holds anything: a finished bundle (`INDEX.json`, no `.generating`), an unfinished generation (`.generating`), or other files. The refusal says what is there (including the run record that left the marker) | `--replace-existing` (moves the folder aside and builds in an empty one), or an absent or empty folder |
| Non-empty corpus folder | The corpus must be built (no finished corpus, or `--rebuild-corpus`) and the corpus folder holds anything: a finished corpus, an unfinished generation, or other files. A finished corpus without `--rebuild-corpus` is reused, so it is not refused | Both `--rebuild-corpus --replace-existing`, or an absent or empty folder via `--corpus` |
| Replacing a finished generation in a production folder | The folder to be replaced is a production folder (a folder directly under `runs/experimental_generation_poc_demo` in any root) and holds a finished generation (`INDEX.json` or a corpus fingerprint, no `.generating`) | In addition, one `_superseded/<name>_*` folder beside it must hold a copy with the same SHA-256 as that `INDEX.json` (or fingerprint) and a `SUPERSEDED.md` (parent project retention rule 1.2 (1)). An unfinished generation has no finished generation to compare, so this is not asked |
| One generation at a time | The folder must be moved aside but a `<folder>.replaced-*` or `<folder>.failed-*` is beside it | The refusal says how. Delete `.replaced-*` if you accept the generation now in the folder (for a production folder, with owner approval, parent project retention rule 1.5); to go back to the old generation, clear the current folder and rename `.replaced-*` back. Inspect `.failed-*`, then delete it |
| Lock | A `<folder>.lock` is beside the bundle folder, or beside the corpus folder (where the runner can write): another generate stage is building or reading that folder | Wait for that run to finish. The refusal gives the lock's owner (pid, host, run identity), its age and a command that checks whether the process is alive (`tasklist /FI "PID eq <pid>"`, `ps -p <pid>`). Delete a lock left by a dead run by hand. The runner never deletes a lock |
| Argument with nowhere to go | `--corpus-arg` for a source that has no corpus stage (`amass`, `prism`), or `--corpus-arg`/`--bundle-arg` together with an explicit generation command (`generate_cmd`). If they were dropped, the sample selection would vanish and the whole source would run | Give bundle arguments with `--bundle-arg` (selection arguments go through the production-lineage check) |
| Sample argument | The output is a production lineage folder (under this run's data root, or shaped like `runs/experimental_generation_poc_demo/<lineage>` under another root) and it received a registry `selection_args` flag (`--subjects`, `--trials`, `--only` and so on, including `--subjects=S04` and abbreviations) | Send the output to a scratch folder |
| Sample corpus → production bundle | The bundle folder is a production lineage but the corpus is not its own corpus lineage beside it | Send the bundle to a scratch folder too |
| Sample run → production corpus build | The corpus folder is a production corpus lineage and this run builds or rebuilds it (no fingerprint, or `--rebuild-corpus`), while the bundle is not the production bundle lineage beside it or selection arguments are present. A sample that only **reuses** a finished production corpus is not refused | `--corpus <scratch folder>` |
| Flags the runner sets | `--corpus-arg`/`--bundle-arg` sets the corpus output (`--out`), the bundle output (`--out`, `--out-root`), the corpus flag passed to the bundle (`--paired`, `--retarget`, `--raw`), a source location (`--source-root`, `--root`, `--hknu-root`, `--amass-root`, `--extracted`) or PRISM's `--data-root` (including `--x=v` and argparse abbreviations; an abbreviation that argparse reads as another declared flag passes) | The bundle is `soma-synth run`'s dataset_dir, the corpus is `--corpus`, the data root is `--data-root`, the sources are `SOMA_SOURCE_ROOT` |
| Synchronised folder (**not a refusal, a warning only**) | The bundle folder, the corpus folder (when this run builds or rebuilds it), the run-record folder (`--runs-root`) or the catalog (when there is a register stage) is inside a Dropbox, OneDrive, Google Drive (`My Drive`, `Shared drives` and their localised names, such as the Korean ones), Synology Drive or iCloud Drive folder, or inside a team shared drive named in `SOMA_SHARED_DRIVE_NAMES` (comma-separated folder names, no default; `paths.on_shared_drive`). Folder names match case-insensitively on every OS (including `~/Dropbox`, `~/OneDrive*`, `~/Google Drive`, `/Volumes/GoogleDrive*`). macOS `~/Library/CloudStorage/*` (every File Provider sync client), `~/Library/Mobile Documents` (iCloud) and Linux gvfs `google-drive:` mounts are checked on POSIX paths only (`paths.synced_location`). The run is not refused and continues; a `warning:` line is printed to standard error once per path (`paths.warn_if_synced`; owner decision of 2026-09-30). A local disk is recommended because a sync client can lock, delay or partially upload files being written, and a bundle there can be shared off this PC (sharing needs separate approval). `readme`, `register`, `validate`, `hash-sources --out` and `stage-sources --to` behave the same | To silence the warning, use a `SOMA_DATA_ROOT` on a local disk |
| Read-only and evidence folders | The bundle folder, or a corpus folder this run builds or rebuilds, is inside `extracted`, `raw_archives`, the body model folder (including `SOMA_BODY_MODEL_DIR`), `_superseded`, `_manifest_backfill` or `_runs` of this run's data root or of `SOMA_DATA_ROOT`, or inside a source folder of `SOMA_SOURCE_ROOT`, or is a container itself (the data root, `runs`, `runs/experimental_generation_poc_demo`, `SOMA_SOURCE_ROOT`). A bundle cannot be inside, or be a folder containing, the corpus it reads (`paths.check_output_dir(root=...)`). An evidence folder under any other root (`runs/experimental_generation_poc_demo/{_superseded,_runs,_manifest_backfill}`) is refused too (`paths.evidence_folder`). The run-record folder and the catalog may be written in `_runs` but not inside the other folders above, and not in a container itself or inside a lineage folder (`runs/experimental_generation_poc_demo/<lineage>`, which holds a bundle or corpus) (`paths.check_record_dir`) | A scratch folder or the lineage folder (run records in `<container>/_runs`, the catalog in `<data root>/experimental/catalog`) |
| Another production folder | The bundle folder, or the corpus folder this run reads, is directly under `runs/experimental_generation_poc_demo` in any root but is not this source's own bundle or corpus lineage: whether another lineage the registry declares (`prism` into `amass_faithful_full`, an `hknu` bundle into `hknu_smpl24_paired`, `gaitex_smpl24` as the `hknu` corpus) or a folder nobody declares (`_dryrun_*`, `my_experiment` and so on) (`runner.production_directory_refusal`) | The source's own lineage folder or a scratch folder |
| Replacement leftovers | The bundle or corpus folder is itself a folder moved aside or the output of a failed replacement (`*.replaced-*`, `*.failed-*`) | The original folder (the name without `.replaced-…` or `.failed-…`) |

The last four refusals happen **before** the run record is opened, because the run record would be
written into or beside that very folder (the default is `<bundle parent>/_runs`). So no record is left,
`soma-synth run` ends with exit code 3 and writes the reason to standard error. The other refusals are
left in the run record's `logs/refusal.txt`. The read-only and evidence-folder refusal exists because of
the runner's own writes: moving a folder aside, the run record, and the `VALIDATION_REPORT.json`,
`validation_ledger.json`, `README.md` and `KNOWN_LIMITATIONS.json` that validate and readme write all
happen before the generator's guard or without a generator (`--skip-generate`, `--start-at validate`), and
the generator's guard only looks at the data root the runner passed.

The old `soma-synth pipeline` command refuses the same way. When validate or readme writes into the bundle
folder it applies `paths.check_output_dir(root=<data root>)` to that folder; `--ledger` and `--report`
written outside the bundle get `paths.check_record_file` (`check_record_dir` on the folder, and it does not
overwrite the data root's `README.md` or, if present, `MASTER.md` and `state/local_archive_inventory.json`);
with register, the catalog gets `paths.check_record_dir`; readme's `--top-level-root` gets
`paths.check_readme_root` (it refuses the data root or `SOMA_SOURCE_ROOT` itself and anything inside a
read-only or evidence folder, allows the lineage container, and does not overwrite `<data root>/README.md`,
[retention rule](RETENTION_RULES.md) 1.4); and any lineage other than `--source`'s bundle is refused. A
refusal ends with exit code 3 before anything is written. `soma-synth readme --top-level-root` applies the
same `check_readme_root`.

The same holds when `soma-synth validate`, `readme` and `register` are called on their own. The commands
that write into a bundle folder — `readme`, and `validate` when `--report` or `--ledger` points inside the
bundle — apply `paths.check_bundle_dir(root=<data root>)` to that folder: everything in `check_output_dir`,
plus a refusal inside `runs/experimental_generation_poc_demo/{_superseded,_runs,_manifest_backfill}` in any
root (`paths.evidence_folder`; even when `SOMA_DATA_ROOT` does not point at that root). `_superseded/<name>/`
keeps the replaced generation's `INDEX.json`, `README.md` and validation report, so `readme` does not
overwrite that `README.md`. `register` does not write into the bundle, but so that archived evidence is not
promoted to a live asset it refuses the same evidence folders and read-only folders (`extracted`,
`raw_archives`, source and body model folders) (`check_bundle_dir(writes=False)`) and applies
`check_record_dir` to `--catalog`. `pipeline` applies the same bundle check to a register-only run
(`--start-at register`). `--report` and `--ledger` written outside the bundle get `check_record_file`, and
then the bundle is only read, so it is not refused. `validate` without `--report` or `--ledger` writes
nothing, so it does not check locations. A refusal is exit code 3, before anything is written.

**An unfinished bundle is never taken over.** `soma-synth validate`, `readme`, `register` and `pipeline`,
the runner's validate, readme and register stages, and the parent project's `scripts/push_sample_data.py`
refuse the following (exit code 3, naming the command that rebuilds it): a bundle with `.generating`
(rebuild it from the start with `soma-synth run --source <source> <bundle> --replace-existing`), and a
folder named `*.replaced-*` (an old generation moved aside) or `*.failed-*` (the output of a failed
replacement) (these are not bundles; use the original folder). `register` and `pipeline` also refuse any
other registry-declared lineage that is not `--source`'s bundle lineage.

Arguments that do not select a part, such as `--compress`, `--resume` and `--force`, are passed on to a
production lineage too. Changing a production lineage follows the order of [retention rule](RETENTION_RULES.md)
1.2 on any PC (evidence to `_superseded/` → owner-approved deletion → regeneration). Done in that order, the
folder after the deletion is absent or empty and `--replace-existing` is not needed. `--replace-existing`
does not replace that order. Given for a production folder that holds a finished generation, the runner
proceeds only when one of the `_superseded/<name>_*` folders beside it holds a copy with the same SHA-256 as
the `INDEX.json` (or corpus fingerprint) being replaced and a `SUPERSEDED.md`, that is, only when 1.2 (1)
has been done for this generation. A corpus fingerprint has no `generated_utc`, so the match is by content,
not by folder date. A scratch folder is replaced without this check. A reused corpus does not receive
`--corpus-arg`; the log and `manifest.json` `corpus.corpus_args_used: false` say so (it is not refused, so
that the same command can re-run a sample).

**A replacement is built in a fresh folder.** When a non-empty folder is replaced with `--replace-existing`
(for a corpus, together with `--rebuild-corpus`), the runner only renames the whole folder to
`<folder>.replaced-<run>` on the same volume (`os.replace`, no copy), and as soon as `os.replace` returns it
puts the moved folder on the list this run will restore. It then copies the small evidence files
(`INDEX.json`, `INDEX_shard_*.json`, `SUMMARY.json`, `_run.json`, `VALIDATION_REPORT.json`,
`validation_ledger.json`, `KNOWN_LIMITATIONS.json`, `README.md`, `DATA_DESCRIPTION_EN.md`,
`reduced_model_fit.json`, the corpus fingerprint and so on, `runner.REPLACED_EVIDENCE_FILES`) to the run
record's `replaced/<bundle|corpus>/`, writes them in `manifest.json` `replaced.<bundle|corpus>` with their
SHA-256, what the folder held (`found`) and where it was moved, and then runs the entry point on an empty
folder. `<run>` is the first 12 characters of the run identity (a run that repeats the same identity gets
`-2` and so on, as the record folder does). So the old generation's takes, shard indexes, validation
reports and corpus files cannot remain in the new generation, and the entry point never needs a flag to
rebuild skipped takes. Registry v2 has no such flag (`force_arg`) and no resume fields, and a
`--bundle-arg=--force` given by the operator is passed on like any other argument. Because it builds in an
empty folder, `amass` no longer rewrites the subject fits in `reduced_model_fit.json`. The bundle folder is
moved after the corpus stage and its class check are finished, just before the bundle entry point. If the
corpus stage fails, the bundle is left untouched.

- **On success**, after the whole generate stage (entry point, post-steps, class check,
  `KNOWN_LIMITATIONS.json`, `source_manifest.json`) is finished, the folder moved aside is deleted
  (`replaced.<what>.outcome: deleted`). A production folder's — a folder directly under
  `runs/experimental_generation_poc_demo` in any root, not only a registry-declared lineage — is not
  deleted. Deleting a production payload needs owner approval ([retention rule](RETENTION_RULES.md) 1.5), so
  it stays where it is and the generate stage's result names its path (`kept_production_directory`).
- **On failure**, nothing is deleted except the one marker this run wrote. The new folder is renamed to
  `<folder>.failed-<run>` (deleted if it holds only the marker) and the folder moved aside is renamed back
  to `<folder>`. The previous state comes back byte for byte (`outcome: restored`, `failed_output_relative`).
  A run that moved both a corpus and a bundle restores both. The same holds when a stage dies with an
  exception, including an exception while copying the evidence files right after the move. If the restore
  fails (a locked file and so on) the result names the path of the old generation and asks for a manual
  restore.
- **One generation at a time.** While `<folder>.replaced-*` or `<folder>.failed-*` is beside it, the folder
  is not replaced again (table above). A successful replacement of a production folder leaves the old
  generation beside it, so the next replacement is possible only after the owner approves and deletes it.
- **A dead run.** If the process dies before its failure handling, the new folder keeps `.generating`, and
  `<folder>.replaced-<run>` and the lock file stay beside it. No run takes them over or continues them.
  While the lock is there, runs are refused because of the lock; after the lock is deleted by hand, the
  folder is an unfinished generation and is refused without `--replace-existing`, and with it, refused
  because of the folder beside it. That refusal says how to restore (clear the current folder and rename
  `.replaced-*` back) and how to clean up.

**Lock.** Before inspecting any folder, the generate stage creates `<folder>.lock` with
`O_CREAT | O_EXCL` beside the bundle folder and, when there is a corpus stage, beside the corpus folder.
Only the first run to create it succeeds. It holds the pid, host, start time, run identity and run-record
location. It is deleted however the stage ends (success, failure, exception, refusal), but only if the file
is still exactly what this run wrote (a lock taken by another run after the operator deleted this one
belongs to that run). A corpus is locked even by a run that only reads it, so that no other run rebuilds it
while a bundle reads it; as a result two sample runs that read the same corpus do not run at the same
time. A corpus in a place no run can write (an evidence or read-only folder, `paths.check_output_dir`)
cannot be rebuilt by anyone, so it is not locked and nothing is written beside it. The runner never deletes
another run's lock. The refusal for a stale lock gives its owner and age and a command that checks whether
the process is alive.

**An unfinished generate stage.** The generator ends by writing `INDEX.json`, but the generate stage still
has post-steps (`prism`'s `synthesize_insole_heading.py`), `KNOWN_LIMITATIONS.json` and
`source_manifest.json` after that. So when the runner starts building in an empty folder (before the bundle
entry point) it writes `.generating` (which run is building it: run-record location, run identity, start
time) and deletes it only after all of these are finished. The corpus stage writes the same marker into the
corpus folder before the corpus entry point and deletes it after the fingerprint and class check. The
marker does not say what to continue; its only use is refusal. A corpus with the marker is not reused. If a
post-step fails, the bundle holds both a new `INDEX.json` and `.generating`. That bundle is not finished.
validate, readme and register (the runner's stages, `--skip-generate`, `--start-at validate`, or the
`soma-synth` commands), `pipeline` and push (the parent project's script) refuse it, naming the command that
rebuilds it (`logs/refusal.txt`, exit code 3), and the next generate stage refuses it without
`--replace-existing`. With it, the folder is moved aside and built from the start. That `INDEX.json` was
not written by a finished run, so it is not compared with `_superseded` (for a production folder, the
folder moved aside is kept and named in the result). If the generator fails before it starts and the
folder holds only the marker, the marker is deleted, and a folder this run created is deleted entirely
(the same for a corpus).

**Corpus class.** When a corpus fingerprint file records `artifact_class` and `distribution_scope`, the
same check as for bundles is applied. The `hknu` and `gaitex` fingerprints record both fields.
`addbiomechanics`'s `SUMMARY.json` does not (each artifact records them), so it is left as
`corpus.class_check.declared: false`.

---

## 2. Contracts between stages

- Each stage takes **only the previous stage's output** as input. It reads no global state.
- A stage that fails **leaves no partial output**. A take directory is either complete or absent.
- Every stage except `close_run` works per take; `close_run` works per run.
- A stage may be skipped (resume), but **the skip is recorded in `metrics.json`.**

### 2.1 Execution environment — `smpl18` must be on the path (2026-09-16)

The rules of the SMPL conversion (model file per sex, skeleton definition, forward kinematics) are owned
by a **separate repository**, which this repository mounts as a **submodule** at `packages/smpl18` (private
`JeongHyunho/smpl18`, switched on 2026-09-16). The runner and generators use it through `import smpl18`, so
the submodule must be populated and its path must be in the execution environment.

The package was renamed from `smpl24` to `smpl18` on 2026-09-16 — the corpus it produces is an 18-joint
reduced model (after going through the full 24-joint SMPL pose, four joints are fixed to per-subject
constants and the two hand joints are dropped) — and the remote repository moved to
`JeongHyunho/smpl18`. No paths or imports under the old name remain.

```bash
# Once: populate the submodule (for a fresh clone, git clone --recurse-submodules)
git submodule update --init

# Install (both editable; the baseline versions are in constraints.txt)
python -m pip install -c constraints.txt -e packages/smpl18 -e .
soma-synth run --source <name> ...

# Development run without installing: put both source trees on the path (macOS / Linux separator is ':')
PYTHONPATH="src:packages/smpl18/src" python -m soma_synth.cli run --source <name> ...
```

```powershell
# Development run without installing (Windows PowerShell, separator is ';')
$env:PYTHONPATH = "src;packages/smpl18/src"; python -m soma_synth.cli run --source <name> ...
```

The baseline environment is CPython 3.13.5, numpy 2.2.6, scipy 1.16.2 (§6). The shell's `python` may not
be the venv's interpreter, so call the venv's Python by path or activate the venv. The runner needs
`SOMA_DATA_ROOT` (it can also be given with `--data-root`). Sources are read from `SOMA_SOURCE_ROOT`
(default `<SOMA_DATA_ROOT>/extracted`) and body models from `SOMA_BODY_MODEL_DIR` (default
`<SOMA_DATA_ROOT>/body_models/smpl`).

If the submodule is empty, `import smpl18` fails — that failure must be visible to tell "not populated"
apart from "the rules have diverged". Which commit is used is set by the parent repository's gitlink, and
that value is the reproduction baseline (the same discipline as the model-set hash in §5.2). The runner
records the revision of the `smpl18` it actually imported next to the code revision: with a checkout
(`packages/smpl18/.git`), its HEAD, dirty state and patch hash; for an installed copy, the version and the
SHA-256 of each package file (§4.1).

pytest picks up the same two paths through `pythonpath` in `pyproject.toml`. If they are not on the path,
`import smpl18` fails, and it is better that this shows without a silent fallback — the model selection
rules splitting into two copies was the original problem.

To update the submodule, commit and push in that repository, then move the gitlink in the parent:

```bash
cd packages/smpl18 && git commit -am "..." && git push && cd ../..
git add packages/smpl18 && git commit -m "bump the smpl18 pin"
```

### 2.2 The 18-joint reduction — `smpl18` defines it, fitted once per subject (2026-09-16)

The 18 joints of `large_reference.npz` are SMPL-24 with `spine1`, `spine2` and both `collar` joints fixed to
per-subject constants and the two hand joints dropped. What is fixed, how the joints below them (`spine3`,
both shoulders) absorb the removed rotation, and how the four constants are fitted are all defined by
`smpl18.reduce`. Until 2026-09-16 this repository had its own copy of the same computation in
`scripts/poc/anthro_smpl.py`, and four of the five sources (hknu, addbiomechanics, gaitex, amass) fitted the
constants **per take**.

Now everything goes through `soma_synth.pipeline.reduced_model`. This module does only the two things the
package leaves to the caller.

| What | How |
|---|---|
| Which frames are pooled | Sampled evenly as if **all takes** of one subject were concatenated (a long take counts in proportion to its length). Takes on different rest skeletons are not mixed; they form separate groups |
| Where the numbers come from | The `reduce` section (`sample_frames`, `optimiser`, `max_evaluations`) of the profile named by the registry's `smpl18_profile`. The code has no defaults |

The pooling unit per source:

| source | Fit group | Frames used for the fit |
|---|---|---|
| `hknu` | subject | Every trial in the corpus, at the native sample rate, on the bone-length-corrected skeleton |
| `addbiomechanics` | subject | Every readable trial, at the native sample rate |
| `gaitex` | subject | The pair-window rows of every take — exactly the rows the bundle carries |
| `amass` | subject × sex model × MoSh betas | Every motion sequence in the group, at the native sample rate. Even when several shards run at once the fit happens once (lock file); the other shards read its result file |
| `prism` | subject | All frames of every take |

The record is kept in two places. `reduced_model_fit.json` at the bundle root holds, per group, the four
constants (`constants_wxyz`), the absorbing joints, the residual before and after the fit (RMS and maximum,
m), per-joint RMS, the number of frames used, convergence, and the settings and profile hash, and it says
which group's constants each take used. Every bundle must have this file
(`dataset_profiles_v1.yaml` `universal.bundle_files`). Each take's `manifest.json` carries the same record
as `anthro_reconstruction.reduced_model_fit`, and the take's own residual is, as before, in
`validation.reduced_model_fit_residual_m`. When the fit used a single take without a group (a generator
called on its own), the record's `scope` is `take`.

### 2.3 Preparing sources — the list and staging (`hash-sources`, `stage-sources`)

The five source folders (`amass`, `prism`, `gaitex`, `hknu_fullbody`, `addbiomechanics`) reach a PC by
copying, from a folder that holds all five (for example a read-only shared copy). Two commands do that
(`pipeline/source_files.py`).

```bash
# List: every file in the source folders, GNU coreutils format ("<sha256>  <folder>/<relative posix path>", sorted by name, one per line)
soma-synth hash-sources [--source-root <root>] [--sources hknu,prism] --out <SHA256SUMS> [--force]

# Stage: copy only the listed files that are missing. --verify-only compares without copying
soma-synth stage-sources --from <folder holding the source folders> [--to <source root>] [--sources ...] --sums <SHA256SUMS> [--verify-only]
```

- `hash-sources` only **reads** the sources. `--source-root` defaults to `SOMA_SOURCE_ROOT`, else
  `<SOMA_DATA_ROOT>/extracted`. `--sources` takes folder names (`hknu` is `hknu_fullbody`), all five by
  default. The list is never written inside the source root, inside `extracted/` or `raw_archives/` of
  `SOMA_DATA_ROOT`, inside an evidence folder (`_superseded`, `_runs`, `_manifest_backfill`, matched by name
  wherever it is), inside the lineage container `runs/experimental_generation_poc_demo` (that of
  `SOMA_DATA_ROOT`, and any path whose folder names continue that way), inside a bundle or corpus (an
  `INDEX.json` in some folder above), or over the data root's own files (`README.md` and, if present,
  `MASTER.md` and `state/local_archive_inventory.json`) (exit code 3). A synchronised folder only warns. An
  existing file is replaced by the new list only when it reads as a list this command wrote (every line is
  a `sha256sum` line naming a file in a source folder, the names are unique and sorted, and it ends with a
  newline). Any other file is overwritten only with `--force` (`--force` does not lift location refusals).
  The list can also be checked with `sha256sum -c`.
- `stage-sources` copies from `--from` only the listed files that are **missing** in `--to` (default
  `SOMA_SOURCE_ROOT`, else `<SOMA_DATA_ROOT>/extracted`). A copy is written under a temporary name in the
  destination folder (`.<name>.*.staging`) and renamed to its own name only when it matches the listed
  SHA-256 (a rename that does not overwrite). A file that exists with a matching hash is skipped. If any
  existing file has a **different** hash, all of them are reported, **nothing is copied**, and it ends with
  exit code 3. This command never deletes or overwrites anything. A file missing from `--from`, or a file in
  `--from` that differs from the list (a drive still syncing), is not written; it is reported and the exit
  code is 1. `--verify-only` reports missing and different files without copying (exit code 1 if any).
- A copy left under a temporary name by an interrupted staging (`.<name>.<random>.staging`) is not a source
  file. The enumeration for the list and for source tracing (`iter_source_files`) leaves it out, and
  `stage-sources`, including `--verify-only`, reports every one left under the destination's source folders
  (those chosen with `--sources`) as `LEFTOVER` each time. It does not delete them — the owner deletes them
  by hand. They do not affect the exit code.
- `--from` may be a synchronised folder or a shared drive (it is only read). `--to` gets the guard the source
  root gets (a synchronised folder or shared drive only warns): not inside an evidence folder (`_superseded`,
  `_runs`, `_manifest_backfill`), not the data root itself or inside its `runs`, `raw_archives` or body model
  folder, and not overlapping `--from`. Staging only **adds files that were missing** to the source folders.
  Existing sources are neither changed nor deleted, per retention rule 1.4.
- Files the operating system leaves (`.DS_Store`, `Thumbs.db`, `desktop.ini`, `._*`) are not source files and
  are not listed. The run record's source tracing (§4.1) uses the same enumeration.

---

## 3. run identity

### 3.1 Keep `run_id` and add `run_identity`

Today `run_id` is a take label and a required manifest field. Changing its meaning would change the
manifests of the existing five bundles. So **`run_id` is left untouched** and a run-level identifier is
added under a new name.

```
run_identity = sha256(
    canonical_json({
        "spec_id":            <spec_id>,
        "spec_version":       <spec_version>,
        "source_name":        <source>,
        "entrypoint":         <script path>,
        "resolved_config":    <sha256 of resolved_config.yaml>,
        "code_revision":      <git HEAD sha + dirty flag>,
        "input_corpus":       <input corpus path and the sha256 of its SUMMARY/INDEX>,
        "body_model_sha256":  <sha256 of the SMPL model files>,
    })
)[:32]
```

This meets §10.1's "hash including config/code/source identity" as written. The same inputs give the same
`run_identity`.

The `resolved_config` item (`resolved_config_sha256`) was not filled in before registry v2 (2026-09-25), so
two runs that differed only in output folder, sample selection or stages got the same identity. It is now
filled with the hash of the canonical JSON of `resolved_config.yaml`'s `config` block. The data root, and the
absolute path of a `dataset_dir` that has a path relative to the data root, are left out of the hash — the
same run on another PC gets the same identity. Repeating exactly the same run gives the same identity, so
the runner does not overwrite the first run's folder and records into `run_<identity>-2`.

> **Remaining mismatch.** The wording of §10.1 asks that `run_id` itself be that hash. For compatibility
> this specification uses a new field, so it does not match the wording exactly. The sealed document cannot
> be edited, so the difference is recorded here.

---

## 4. The run directory

`runs/<run_identity>/` holds the seven artifacts of §10.1.

| File | Content | Written |
|---|---|---|
| `resolved_config.yaml` | The full resolved configuration (defaults included) | Run start |
| `manifest.json` | Run-level manifest — `run_identity`, source, stage list, results | `close_run` |
| `environment.lock` | Python and dependency versions, platform | Run start |
| `source_assets.json` | Input asset paths and SHA-256 | Accumulated by `register_asset` |
| `metrics.json` | Take count, skipped count, elapsed time, per-stage totals | `close_run` |
| `exclusions.json` | Excluded takes and reasons | `close_run` |
| `logs/` | Run logs: `runner.log` (what the runner printed), and for each child stage its standard output and standard error — `corpus.log` (corpus entry point), `generate.log` (bundle entry point and merge; per shard `generate.shard<i>of<N>.log`), `post_step.<script>.log` (post-steps). Each starts with the command line (`$ ...`) and ends with the exit code (`[exit N]`), and also appears on the console unchanged (`pipeline.generate` tees it). `refusal.txt` for a refusal, `error.txt` for an unexpected failure | Throughout |

A take directory's `manifest.json` (contract §12.2) and the run directory's `manifest.json` (§10.1) are
**different things**. The first is the take's field contract; the second is the record of the run.

### 4.1 The tracing record — prepared for a licence withdrawal (ADR-0041, 2026-09-25)

A bundle a teammate makes lives only on that PC and is not in a central catalog. If a source's licence is
withdrawn, the run records are collected to find the affected outputs (`PIPELINE_GOVERNANCE.md` §12). So a
run that makes a bundle leaves the following in addition to the seven artifacts.

| What | Where |
|---|---|
| Source reference list — every `relative_path` (or `source_asset_id`) the take manifests record and the hash beside it (`sha256`, `source_asset_sha256`, `source_asset_sha256_take`), the take and field of each reference, the count and a combined hash | `source_manifest.json` |
| Whole-file source hashes (sources with a corpus) — for every source file this run's corpus and bundle stages could read: logical id (`extracted/<folder>/<path>`), size, SHA-256 of the **whole file**, the stages that could read it, the selection arguments of each stage, the count, total size, combined hash and time spent hashing | `source_manifest.json` `source_files`, summary in `manifest.json` `tracing.source_manifest.source_files` |
| Corpus — lineage, whether this run built it (`built`, `rebuilt`, `reused`), whether corpus arguments were used (`corpus_args_used`), the fingerprint file and its SHA-256, what the folder held when opened (`found_at_open`: `null` for nothing, a finished corpus, an unfinished generation, other files), whether it was moved aside (`moved_aside`) | `manifest.json` `corpus`, `source_assets.json` |
| `smpl18` revision — the checkout's HEAD, dirty state and patch hash, or the version and per-file hashes | `environment.lock`, `manifest.json` `provenance.smpl18_revision` |
| Basis for generating — the basis (`standing_decision`, `owner_record`), the ADR, the decision record path and its SHA-256, the class. If an `--authorized-by` record authorised the run, that record's SHA-256 (`authorization_record_sha256`) and a copy in the run record (`authorization/`) | `manifest.json` `generation_decision` |
| Intended basis — written before the decision is asked (`generation_basis_intended`; `null` without a generate stage or with `--skip-generate`). The SHA-256 of the `--authorized-by` record given (`authorized_by_sha256`) | `resolved_config.yaml` |
| A summary of the three above (source list count and combined hash, code and `smpl18` revisions, decision record path and SHA-256, the actual basis `generation_basis` and the intended basis `generation_basis_intended`). For a refused run the actual basis is `null` or the reason for refusal (`artifact_class`, `gates`) | `manifest.json` `tracing` |
| Source root — where the generators read (`SOMA_SOURCE_ROOT`, else `<data root>/extracted`) and where that came from (`source_root_from`) | `resolved_config.yaml` `source_root_relative`, `source_root` |
| Bundle folder — whether `INDEX.json` existed before generation (`index_present`), what the folder held when opened (`found_at_open`: `null` for nothing, a finished bundle, an unfinished generation and the run that left its marker, other files), whether it was moved aside (`moved_aside`), and whether the generate stage finished and removed `.generating` (`generating_marker_removed`) | `manifest.json` `bundle` |
| The catalog location, and each recorded path relative to the data root. The paths added with registry v2 (catalog, `_runs`, body model folder, per-file model overrides, corpus folder, the bundle in `source_manifest.json`) are recorded **as relative paths only** when inside the data root, and as absolute paths only when outside. Teammates' run records are collected and read, so they should not multiply that PC's absolute paths. `dataset_dir` and `data_root` keep their absolute paths as before | `resolved_config.yaml`, `source_assets.json`, `provenance.body_model` |
| What was replaced — for each bundle or corpus folder moved aside: the original location, where it was moved (`aside_relative`, `aside_name`), what it held (`found`), whether it is a production folder (`production_directory`), the location and SHA-256 of the copies of the small evidence files (`evidence`), and the outcome (`outcome`: `deleted` or `kept_production_directory` on success, `restored` and the `.failed-*` location of the failed output on failure) | `replaced/`, `manifest.json` `replaced` |

The manifests of bundles built from a corpus (`hknu`, `gaitex`, `addbiomechanics`) record **corpus files**
as their source. Which source files that corpus came from is in the corpus's own record, and the run record
also keeps the corpus fingerprint. `gaitex` also records the source marker and IMU files the small was
synthesised from (`small_synthesis.inputs`), so those are in the list too.

**Source hashes beyond the corpus are in the run record.** The corpus's own record describes its sources
only this way.

| Corpus | How it records its sources |
|---|---|
| `hknu_smpl24_paired` (`generate_hknu_faithful.py`) | `SUMMARY.json` and the corpus npz record only subject and trial names. No source file hashes |
| `gaitex_smpl24`, `addbio_smpl24_raw` | Each artifact's `adapter_hash` is the first 16 hex digits of the SHA-256 of the **first 1 MiB** of the source file. Not a whole-file hash |

This record feeds npz members, manifests and `pair_id`, so it is not changed. Instead the runner adds a
`source_files` block to the run record's `source_manifest.json` (`pipeline/source_trace.py`). The whole-file
hashes are there.

- **Which files.** Every file in that source's folder under the run's source root (`SOMA_SOURCE_ROOT`, else
  `<data root>/extracted`) that this run's stages could read. With selection arguments it is narrowed to the
  selected subjects. Files that belong to no subject (workbooks, README, `_provenance`) are always included.

  | source | Corpus stage (only when this run builds the corpus) | Bundle stage |
  |---|---|---|
  | `hknu` | `hknu_fullbody`, narrowed by `--subjects` (`Dataset_Processed/<S>`, `Dataset_Raw/C3D/<S>`) | `hknu_fullbody` (reads the workbook), narrowed by the bundle's `--subjects` |
  | `gaitex` | `gaitex`, narrowed by `--subjects` (`<subject>/…`) | `gaitex` (reads each take's marker and IMU files), narrowed by the bundle's `--subjects` |
  | `addbiomechanics` | `addbiomechanics`, narrowed by `--only <study>/<subject>`; with `--per-study N`, the first N non-empty `.b3d` per study in the generator's order and by its rule (the generator adds a file before comparing the count, so even N of 0 or less gives the first 1) | Reads no sources (reads only the corpus) |

  The sources of a reused corpus are in the record of the run that built it, and this record has that
  corpus's fingerprint. `amass` and `prism` have no corpus and their take manifests record source files with
  whole-file SHA-256, so `source_files` records `status: not_recorded` with the reason. An explicit
  generation command (`generate_cmd`) cannot know what it read, so it is also `not_recorded`.
- **What is recorded.** Per file: the logical id `extracted/<folder>/<relative path>`, size, SHA-256 (whole
  file, read 1 MiB at a time) and the stages that could read it. The block holds each stage's selection
  arguments and whether it was narrowed (`narrowed`), the source root relative to the data root
  (`source_root_relative`, or the absolute `source_root` when outside), the count, total bytes, combined hash
  (SHA-256 of the sorted, concatenated `<relative_path>\t<sha256>` lines) and the seconds spent hashing.
  Unreadable files are kept in `unreadable` with a reason, and a missing source folder gives
  `status: unavailable`. Each file is hashed once. The enumeration is the same as `hash-sources` (§2.3): the
  same files, in the same order, with the same hashes.
- **When it is hashed.** At the end of the generate stage, before `.generating` is removed. **The sources of
  the corpus stage are hashed only by the run that builds that corpus.** A run that reuses the corpus does
  not hash its sources again: the run that built it already traced them (in that run record's
  `source_files`), and this record has the reused corpus's fingerprint. For `hknu` (workbook) and `gaitex`
  (each take's marker and IMU files), whose bundle stage reads sources, a run that rebuilds only the bundle
  still hashes the sources the bundle stage could read, narrowed to the selected subjects.
- **Cost.** Hashing time is proportional to source size. Here GB is 10^9 bytes (the unit `hash-sources`
  prints). `hknu_fullbody` is 843 files, 27.88 GB, and a run that builds the whole `hknu` corpus or bundle
  reads that much more. `gaitex` is 17.7 GB. The 1,137 `.b3d` files of `addbiomechanics`, about 568 GB
  (529 GiB), are hashed **only by a run that builds the whole AddBio corpus afresh** (that run itself takes
  hours, so this is accepted). An `--only` or `--per-study` sample run hashes only the chosen `.b3d`, and an
  AddBio run that reuses the corpus and builds only the bundle hashes no sources (the AddBio bundle stage
  reads only the corpus).

The catalog defaults to `<data root>/experimental/catalog`. The old default `<bundle parent>/catalog` was
inside the lineage container, so it was not a catalog. An asset's `relative_path` is **relative to the data
root**: `run` uses the runner's data root, `pipeline` and `register` use `--data-root`, else
`SOMA_DATA_ROOT`, and stop with exit code 2 if neither is set. A bundle that is not inside the data root is
also refused with exit code 2. Relative to `<bundle parent>`, paths would come out as `gaitex_unified8/...`
and not match the entries `run` registered (`runs/experimental_generation_poc_demo/gaitex_unified8/...`), so
that is not used as the base.

The catalog files (`asset_catalog.json`, `<source>/source_manifest.json`) are written this way
(`datasets/catalog.py`).

- **Lock.** A writer holds `.catalog.lock` in the catalog folder while it reads, modifies and writes. The lock
  is an operating-system lock, **not the file's existence** (on Windows `msvcrt.locking` on the first byte,
  elsewhere `fcntl.flock`). The file is created if missing and used as it is if present (an empty lock file
  created by another writer may already be there). Either way it is **never deleted**: it may be someone
  else's lock, and deleting a lock file someone is about to take lets two writers in together. The lock is
  released when the process ends. If another writer holds it, the writer waits (default 900 seconds) and then
  stops with `CatalogLocked`. Asset hashes are computed first, outside the lock.
- **Atomic replacement.** The new content is written to a temporary file in the same folder (`.<name>.*.tmp`),
  flushed and fsynced, and swapped in with `os.replace`. Readers see the old file or the new one, never a
  half-written file. On failure the file is unchanged and the temporary file is deleted (if the process dies,
  a temporary file may remain, but the catalog file is intact).
- **Change record (journal).** Each catalog file has `_history/<stem>.journal/` beside it, and every write adds
  one small gzip JSON file `<8-digit sequence>.<UTC time>Z.json.gz` to that folder (written under a temporary
  name, fsynced, and then linked only to a name nobody has written: no overwriting). A record holds the write
  time (`utc`), the writer (`registered_by`: the tool name, or the run's `run_<identity>` when the runner
  registers), the operation (`operation`), the SHA-256 and size of the file before and after (`before`,
  `after`), and **only the entries that changed** — for each entry added, replaced or deleted: its key (JSON
  Pointer: `/assets/<index>` for assets, `/<field>` for other fields), the old entry (null when added) and the
  new entry (null when deleted). No full copy of the catalog is kept: registering a few assets gives a record
  of a few KB, even when the catalog is hundreds of MB (assets are compared by position; this tool's writers
  only append assets or modify them in place). When `run`, `register` or `pipeline` registers a bundle for the
  first time, the record follows the bundle's asset count: about 55 bytes per asset gzipped, about 20 KB for
  the 360 assets of `gaitex`, about 11 MB for the 202,300 assets of `addbiomechanics`. A
  `supersede_spec_version` of a whole version holds both the old and new entries, about twice that.
  Re-registration is as large as the assets that changed (below), and nothing changed means no record. This
  tool never modifies or deletes a record. Every earlier entry can be restored from the current file and the
  records: `catalog.catalog_as_of(path, n)` rebuilds the file as it was right after record n was applied by
  undoing the later records from the current file, and reports whether the bytes match the record's SHA-256.
- **Order and dead writes.** New content fsynced to the temporary file → record fsynced and linked →
  `os.replace`. Dying before the record changes nothing. Dying between the record and the swap, or a failed
  swap, leaves the file unchanged and a record whose `after` the file never had. That record's `before` is the
  current file's hash, and so is the next write's `before`, so `catalog.journal_story(path)` identifies it as
  a record that was not applied (a change no record explains is marked "changed outside the record"). The
  reverse order (swap first) would lose the replaced entries' old values forever if it died before they were
  recorded. This order loses nothing. For a catalog from before there were records (before this tool), the
  first record's `before` is the starting point.
- **Re-registration reconciles.** A bundle rebuilt in place keeps its asset ids and paths; only the bytes
  change. If registration only appended, each id would have two live entries and the old entry's SHA-256 would
  not match the file. Registration now reconciles with the dataset's earlier entries
  (`catalog.merge_registration`). The dataset's entries are the live (not `superseded`) entries with the same
  asset id under the same bundle folder, or with the same relative path in the same lineage (source, spec_id,
  spec_version). Looking newest first, the first entry identical to the asset being registered (the eight
  fields of `catalog.ASSET_FIELDS` are equal; `registration_evidence` and `metadata` added beside them by
  another writer are ignored) is kept and that asset is not appended again. All others are marked in place by
  the convention the existing catalog already uses: `superseded: true`,
  `supersession_reason: "registration_replaced"`, `superseded_by` the new entry's asset id,
  `superseded_by_sha256` the new entry's SHA-256. Entries are never deleted or moved. Live entries of the same
  lineage under the same bundle folder that none of the assets being registered replaces — takes no longer
  `ok` in the new INDEX, entries for take folders or role files that are gone — are marked in the same write:
  `superseded: true`, `supersession_reason: "registration_dropped"`, `superseded_by: null`. After an in-place
  replacement that file no longer exists, so leaving the entry live would point at a missing file. This rule
  is not applied when the bundle folder is the data root itself. The marked entries go into that write's
  record as `replaced` changes (the whole old entry and the new entry), the appended entries as `added`, and
  the record's `registered_by` names the run. `operation` has the counts of `added`, `unchanged`,
  `superseded` and `dropped`. A registration that would change nothing does not write (no record, and the file
  stays the same; `<source>/source_manifest.json` likewise). So a catalog registered while registration only
  appended is cleaned up in **one record** by registering the same bundle again: the old entries are marked,
  duplicate live ids disappear, every live entry's SHA-256 matches its file, and registering once more writes
  nothing. The clean-up command is either of these. The first one leaves a run record and puts that run's id
  in the record's `registered_by`.

  ```bash
  soma-synth run --source gaitex --start-at register --stop-after register
  soma-synth register "$SOMA_DATA_ROOT/runs/experimental_generation_poc_demo/gaitex_unified8" \
      --catalog "$SOMA_DATA_ROOT/experimental/catalog" --source gaitex --data-root "$SOMA_DATA_ROOT"
  ```

  Either way `SOMA_DATA_ROOT` must be the data root that holds the bundle. In the output,
  `... earlier entries superseded` is the number of entries replaced and `... dropped` the number of entries
  marked because they are no longer in the bundle. An asset id does not name its folder, so the same take in
  another folder (a scratch sample) is a different dataset: they do not supersede each other's entries.
  Entries registering the same folder under a different spec_version are not superseded either (that is
  `supersede_spec_version`'s job).

---

## 5. The ten provenance items

| # | §10.2 item | Source | Now |
|---|---|---|---|
| 1 | Code revision + dirty hash | `git rev-parse HEAD` + hash of `git status --porcelain` | **None** |
| 2 | Contract, schema and config versions | `spec_id`/`spec_version`/`canonicalization_config` | Present |
| 3 | Source asset SHA-256 and licence | `SourceIdentity.asset_sha256` + gate settings | Present |
| 4 | Spec snapshot hash | `spec_documents` | Present |
| 5 | Python and dependency versions | `importlib.metadata` | prism only |
| 6 | Model/checkpoint hash | Per-file SHA-256 of the body model **set** — §5.2 | Recorded by the runner (2026-09-16) |
| 7 | seed | — **see §5.1** | Not applicable |
| 8 | Host environment (non-sensitive) | OS, architecture, CPU count | **None** |
| 9 | Input/output artifact hashes | Accumulated by `emit_take` as it writes | **None** |
| 10 | pass/warn/reject, quarantine | `quality_gate`, `validation`, `exclusions.json` | Present |

**Credentials, tokens, cookies and personally identifying information are never recorded** (§10.2).

### 5.1 seed — record that there is nothing to fill in

The registered paired generation path has **zero** uses of randomness. The check covers
`scripts/poc/generate_*.py`, the common emitter and `src/soma_synth/`, excluding only
`src/soma_synth/experimental/`, which is separate research code. That research code's seeds are not to be
confused with seeds of paired generation. If the generation path imports that namespace, the boundary check
fails. Relative imports and aliases are checked too, and a failure to read or parse a file does not pass.
This is a static source check, not a proof about arbitrary dynamic execution; computed dynamic import targets
and the like need separate review. The ban on randomness in the actual generation code stays as it is.

So no seed is invented; its absence is stated.

```json
"seeds": {"present": false, "reason": "no stochastic step in the generation path"}
```

And **that claim is guarded by a test**. A blank cannot tell "none" from "not recorded", and nobody would
notice if a stochastic step came in later.

### 5.2 Item 6 — record the set, not one file (2026-09-16)

§10.2 item 6 asks for a "model/checkpoint hash", but the five generators choose the model matching each
**subject's** sex. Naming one file would misdescribe a bundle that mixes men and women, so the run record
holds the **whole set**: the directory, the selection rule, the SHA-256 of each file, and `set_digest`, the
hash of those hashes sorted. `set_digest` is also the run identity's `body_model_sha256`, so if any model
changes, the run identity changes.

Two things each come from exactly one place.

| What | From where |
|---|---|
| Where the set is | The `SOMA_BODY_MODEL_DIR` folder if set, else [`source_pipelines_v2.yaml`](../../configs/datasets/source_pipelines_v2.yaml) `body_model_sets.<name>.relative_path` relative to the run's data root — that is, `pipeline.paths.body_model_dir()` |
| Sex → file name | `smpl18.model.select` (`packages/smpl18`) — the generators choose with the same function |

In code, `soma_synth.pipeline.body_models` joins the two, and `addbio_retarget/shape_fit.py` and
`scripts/poc/anthro_smpl.py` do not define the file-name rule themselves but delegate to the same function.
If the set cannot be found, no hash is invented; `unavailable` and the reason are recorded (the same
discipline as §5.1).

The runner resolves the body model folder with the same function as the generators
(`paths.body_model_dir`), passes its data root to the generators as `SOMA_DATA_ROOT`, and hashes the folder
it resolved. The record holds the folder and where it came from (`directory_from`: `SOMA_BODY_MODEL_DIR` or
`data_root`). The per-file overrides `SOMA_SMPL_MODEL_MALE` and `SOMA_SMPL_MODEL_FEMALE` accepted by
`scripts/poc/anthro_smpl.py` (which amass and prism go through) bypass that folder, so when set their path
and SHA-256 are recorded in `resolved_config.yaml` `body_model_file_overrides`.

---

## 6. Determinism

- File iteration is always sorted. Nothing depends on directory enumeration order.
- Parallel execution does not change results. Takes are independent and share no mutable state.
- The order of floating-point operations does not change with the degree of parallelism.
- Process pools start their workers with `spawn` on every platform (`multiprocessing.get_context("spawn")`,
  `ProcessPoolExecutor(mp_context=...)`). Windows has only `spawn` and Linux defaults to `fork`, and a worker
  that inherits the parent's modules, `sys.path` edits and open files could behave differently from PC to PC.
  The pools in use today are the two `--jobs` of `addbiomechanics` (`generate_addbio_unified8.py`,
  `generate_addbio_faithful.py`), and `tests/pipeline/test_process_start_method.py` scans `src/` and
  `scripts/` by AST and rejects any pool made without `spawn`. The runner's `--jobs` shards start generator
  processes with `subprocess`, so they are not affected.

### 6.1 Byte identity is claimed only on the same PC in the same pinned environment (ADR-0041)

- On **the same PC in the same pinned environment** (CPython 3.13.5, numpy 2.2.6, scipy 1.16.2, the same
  `smpl18` revision) the outputs must be byte-identical. This is checked with the harness (`scripts/harness/`,
  `run_harness.py` → `hash_outputs.py` → `compare_hashes.py`). npz files are compared by per-member array hash,
  JSON after parsing with time fields masked. The baseline is a hash file made on the same PC with the code
  before the change (it is not stored in this repository; its path is given to `compare_hashes.py`).
- **Across PCs, results are judged within a tolerance.** CPU, BLAS, numpy, zlib and line endings affect the
  result. A difference of about 1 float32 ulp has been observed across machines. Teammates' bundles stay on
  their PCs (§9), so byte identity across PCs is not required.
- **A declared difference.** The meaning of `code_hash` was not changed (ADR-0041 decision 4). So when a
  generation code file changes, the `code=` token of the corpus's `content_hashes` changes, and so does the
  `pair_id` derived from it. That the `code=` and `pair_id` of `hknu`, `gaitex` and `addbiomechanics` differ
  across the soma-synth split is a declared difference, not a defect (in the split only `gaitex` and
  `addbiomechanics` actually changed; `amass`, `prism` and `hknu` were byte-identical). The harness comparison
  (`compare_hashes.py`) classifies a difference this way only when `code` is the only token that differs. A
  difference in `model`, `adapter` or `config` is a failure.
- The move to registry v2 changed no generator code. The only thing a generator reads from the registry is
  `smpl18_profile`, and its values are the same as in v1. What changed is how the runner calls them: `prism`
  gets `--out <bundle>` instead of `--data-root <root>` (sources from `SOMA_SOURCE_ROOT`), and the three
  sources with a corpus are given the corpus explicitly through the declared flag.

---

## 7. Resume and partial failure

Today the five generators use **four different resume predicates**.

| Source | Predicate | Does it notice a half-written take? |
|---|---|---|
| amass | 4 files exist + `manifest.json` parses | **Yes** |
| prism | core files exist | No |
| addbiomechanics | `manifest.json` exists | No |
| gaitex · hknu | none (always regenerated) | Not applicable |

**The standard adopts the strongest** — `bundle_opens`: skip only when every output file exists and
`manifest.json` parses. The others are recorded as they are today in `source_pipelines_v2.yaml`, to be raised
to this predicate when the runner unifies them.

The number of skipped takes is recorded in `metrics.json`. A silent skip is not a resume but an omission.

`soma-synth run` does not make any generator resume (§1.3): every bundle and corpus is built in an empty
folder, so these predicates have no takes to skip. The predicates above describe the behaviour when a
generator is run directly or an operator passes a flag like `--bundle-arg=--resume`.

Validation has one skip too. The validation ledger remembers each take that PASSed together with its content
digest, `spec_version`, generation code digest and **validator digest** (`catalog.default_validator_digest`:
the bytes of the spec and of `validation/checks.py`, `npz_io.py` and `report.py`), and does not look again
only when all four are the same (`takes_trusted`). So when the validator code changes, every ledger's trust is
released once and the next validation looks at every take again (like `--full`).

The L4 governance absolute-path check catches drive letters and POSIX absolute paths: a path of `/`, a folder
name and another `/` at the start of a value or after whitespace, a quote, `=`, `:`, a bracket or a separator
is a FAIL whatever its root (not only `/Users/`, `/home/`, `/Volumes/`, `/tmp/`, but also `/data/`, `/opt/`,
`/srv/`, `/scratch/`, `/nfs/`, `/gpfs/`, `/workspace/`). Logical ids (`extracted/…`, `runs/…/tmp/…`), units
(`m/s^2`) and URLs (`https://host/home/…`, `file:///home/…`: the character before the path is `/`) are not
caught, and a lone `/tmp` with no path below it is not caught either.

The conformance check (`checks.check_dataset_profile_conformance`) adds a WARN when a file a bundle must have
before distribution (`README.md`, `VALIDATION_REPORT.json`, `validation_ledger.json`) is missing. The runner's
validate stage runs this check before writing the report and ledger and before the readme stage, so when the
caller passes the files it will write after validation as `pending_files`, they are not reported missing. The
runner passes its own report and ledger, which it writes whatever the result, and drops the `README.md` WARN
from the report only when the plan has a readme stage **and validation passed**
(`checks.missing_distribution_file`). If validation fails, the run stops before the readme stage (fail-fast)
and the bundle has no `README.md`, so that WARN stays. With `--stop-after validate` one `README.md` WARN stays:
that run does not write it. `soma-synth validate` passes only the names of `--report` and `--ledger` written
directly into the bundle folder, and a validation that writes nothing warns about all three. A file the bundle
must produce (`KNOWN_LIMITATIONS.json` and so on) is a FAIL when missing even if it is passed.

---

## 8. Per-source differences

**This section is generated from [`configs/datasets/source_pipelines_v2.yaml`](../../configs/datasets/source_pipelines_v2.yaml).
Do not edit it by hand.**

```
soma-synth pipeline-doc --render
```

A hand-maintained table goes stale, so this section is generated.

<!-- BEGIN GENERATED: source pipelines -->

> Generated. The source is [`source_pipelines_v2.yaml`](../../configs/datasets/source_pipelines_v2.yaml);
> `soma-synth pipeline-doc --render` rewrites this block. Do not edit it by hand.

| source | `small_mode` | emitter | resume predicate | parallel | entry point | bundle lineage | corpus | smpl18 profile |
|---|---|---|---|---|---|---|---|---|
| `addbiomechanics` | `synthetic_from_smpl` | `unified8_emit` | `manifest_exists` | `process_pool` | `generate_addbio_unified8.py` | `addbio_unified8` | `addbio_smpl24_raw` | `addbiomechanics` |
| `amass` | `synthetic_from_smpl` | `inline` | `bundle_opens` | `shard` | `generate_amass_faithful_all.py` | `amass_faithful_full` | — | `amass` |
| `gaitex` | `synthetic_from_markers` | `unified8_emit` | `none` | `none` | `generate_gaitex_unified8.py` | `gaitex_unified8` | `gaitex_smpl24` | `gaitex` |
| `hknu` | `synthetic_from_smpl` | `unified8_emit` | `none` | `none` | `generate_hknu_unified8.py` | `hknu_unified8` | `hknu_smpl24_paired` | `configs/smpl18/profiles/hknu.yaml` |
| `prism` | `measured_physical` | `inline` | `core_files_exist` | `none` | `generate_prism_measured_all.py` | `prism_faithful_full` | — | `configs/smpl18/profiles/prism.yaml` |

**What the resume predicates mean**

- `bundle_opens` — skips when every deliverable exists and manifest.json parses
- `core_files_exist` — skips when every core deliverable file exists, without opening any of them
- `manifest_exists` — skips when manifest.json exists, even if the npz beside it are truncated
- `none` — no skip logic; every take is rebuilt on every run

**Shared emitter**: `addbiomechanics`, `gaitex`, `hknu` — **Own implementation**: `amass`, `prism`

**Generation stages (corpus → bundle → post-steps)**

- `addbiomechanics` — corpus `generate_addbio_smpl24.py --out <addbio_smpl24_raw>` (fingerprint `SUMMARY.json`, passed to the bundle as `--raw`) → bundle `generate_addbio_unified8.py --out <addbio_unified8>`
- `amass` — bundle `generate_amass_faithful_all.py --out-root <amass_faithful_full>`
- `gaitex` — corpus `generate_gaitex_smpl24.py --out <gaitex_smpl24>` (fingerprint `_run.json`, passed to the bundle as `--retarget`) → bundle `generate_gaitex_unified8.py --out <gaitex_unified8>`
- `hknu` — corpus `generate_hknu_faithful.py --out <hknu_smpl24_paired>` (fingerprint `SUMMARY.json`, passed to the bundle as `--paired`) → bundle `generate_hknu_unified8.py --out <hknu_unified8>`
- `prism` — bundle `generate_prism_measured_all.py --out <prism_faithful_full>` → post-step `synthesize_insole_heading.py {dataset_dir}`

**Environment variables read**

- `addbiomechanics` — `SOMA_DATA_ROOT`, `SOMA_SOURCE_ROOT`, `SOMA_BODY_MODEL_DIR`, `ADDBIO_RETARGET_CORPUS`
- `amass` — `SOMA_DATA_ROOT`, `SOMA_SOURCE_ROOT`, `SOMA_BODY_MODEL_DIR`
- `gaitex` — `SOMA_DATA_ROOT`, `SOMA_SOURCE_ROOT`, `SOMA_BODY_MODEL_DIR`
- `hknu` — `SOMA_DATA_ROOT`, `SOMA_SOURCE_ROOT`, `SOMA_BODY_MODEL_DIR`
- `prism` — `SOMA_DATA_ROOT`, `SOMA_SOURCE_ROOT`, `SOMA_BODY_MODEL_DIR`

**Body model sets**

- `smpl_clean` — `body_models/smpl` under the data root (or `SOMA_BODY_MODEL_DIR` when set; `pipeline.paths.body_model_dir()`); the file for each sex is chosen by `smpl18.model.select`. Used by: `addbiomechanics`, `amass`, `gaitex`, `hknu`, `prism`

**Generation policy (`generation_policy`)**

- Basis: `standing_decision` — ADR-0041 (`docs/adr/ADR-0041-soma-synth-split-and-teammate-generation.md`), decision record `research/decisions/2026-09-25_source_distribution_and_pipeline_split_decision.md` §4
- Class `experimental_non_candidate`, distribution scope `internal_only`, `releases_holds: false`

<!-- END GENERATED: source pipelines -->

---

## 9. What this pipeline does not do

- **It does not push.** As `_next_step_guide` in `src/soma_synth/cli.py` states, soma-synth uploads bundles
  nowhere. A bundle stays on the PC that made it. This is by design, not unfinished work.
- **It makes nothing that is not `experimental_non_candidate`.** The generation policy is §9.1 below.
  Canonical and internal-candidate generation stay under hold (the parent project's hold rules), and this
  pipeline does not open them.
- **It renders `KNOWN_LIMITATIONS.json` when generation ends.** The validator requires this file of every
  bundle, so the runner, not the generator, writes it from `configs/datasets/known_limitations_v1.yaml`. If
  there is no block for the bundle's name, the file is not written, `limitations_block_missing` is recorded in
  `metrics.json`, and the following validate reports the absence as a FAIL.
- **It does not edit sealed documents.**
- **It does not keep thresholds as code constants.** Thresholds live in versioned configuration files.

### 9.1 Generation policy — standing permission by one decision record (ADR-0041, 2026-09-25)

**Standing permission.** An internal user making `experimental_non_candidate` · `internal_only` bundles with
`soma-synth run` on their own PC is permitted by one decision (`2026-09-25_source_distribution_and_pipeline_split_decision.md`
(parent project: `research/decisions/2026-09-25_source_distribution_and_pipeline_split_decision.md`) §4),
recorded by `ADR-0041` (parent project: `docs/adr/ADR-0041-soma-synth-split-and-teammate-generation.md`). No
owner record is required per run. Registry v2's `generation_policy` names this decision, and a
`soma-synth run` without `--authorized-by` records it in the run `manifest.json` `generation_decision`: the
basis (`standing_decision`), the ADR, the decision record path and its SHA-256, the class, the distribution
scope and `releases_holds: false`. On load, the registry is refused unless the policy's class is
`experimental_non_candidate`, its scope `internal_only` and `releases_holds` `false`.

**Class enforcement.** The runner checks twice.

- If the registry declares for a source an `artifact_class` other than `experimental_non_candidate`, the run
  is refused before anything runs (even with an owner record).
- If the finished bundle's `INDEX.json` is not `artifact_class: experimental_non_candidate`,
  `distribution_scope: internal_only`, or there is no `INDEX.json`, the generate stage ends as a failure
  (`logs/artifact_class.txt`). If a corpus fingerprint file records the two fields, they must have the same
  values (§1.3).

**Generation outside `soma-synth run` is not covered by the standing permission.** The condition of the
standing permission is the run record (decision, source hashes, code revision), and only `soma-synth run`
leaves one. The generate stage of `soma-synth pipeline` is refused (exit code 3) and points to
`soma-synth run`. `pipeline` is for validate → readme → register of an existing bundle (`--skip-generate` or
`--start-at validate`). Running a generator script directly cannot be stopped by the runner, so it is
recorded here as out of scope.

**The same on anyone's PC.** Decision record §4 covers internal users generating **on their own PC**
(ADR-0041 `1.2.0`). The runner applies the same conditions (class enforcement, run record) whoever's PC it is.
On every PC, [retention rules](RETENTION_RULES.md) §1 stay as they are. Changing a production lineage follows
the order of 1.2, and the runner refuses `--replace-existing` on a production lineage when that evidence is not
in `_superseded` (§1.3). `_superseded`, `_runs`, `_manifest_backfill` and the sources are never deleted (1.4),
and the runner does not write inside the sources, the body models or the evidence folders (§1.3).

**A refusal is not a success.** A refused run leaves a run record (`logs/refusal.txt`) and `soma-synth run`
ends with exit code 3 (1 for a failed stage, 2 when paths or the registry cannot be resolved). A read-only or
evidence-folder refusal happens before the run record is opened, so it ends with exit code 3 and no record
(§1.3).

**Per-source owner records still work.** When the owner gives an approval record in `research/decisions/`
that states the date, the source, the `experimental_non_candidate` scope and `releases_holds: false`
(the 2026-09-14 method) with `--authorized-by <record>`, the runner, as before, reads it with `gates.py`,
checks the source and class, and records the gate's reasons for refusal together with the record in
`generation_decision` (basis `owner_record`) and `logs/authorization.txt`. The record file's SHA-256 and a copy
(`authorization/<record>`) are kept in the run record too. A per-source record also permits only
`experimental_non_candidate` (`gates.py` refuses a record that names another class). Canonical and candidate
generation are under the hold and its approved transitions (the parent project's hold rules), and neither the
standing permission nor a per-source record opens them. `gates.py`'s refusal is the parent project's hold check.

**The governance root exists only when mounted in the parent project.** The M0 evaluator
(`configs/evaluators`), the gate settings and the owner records (`research/decisions`) that `gates.py` reads
belong to the parent project. `gates.GOVERNANCE_ROOT` is the parent project root only when this repository is
mounted in it as `packages/soma-synth` and that root has both `configs/evaluators` and `research/decisions`; in
any other arrangement (a standalone checkout, a parent project missing either folder) there is none. It cannot
be changed by environment variable or flag (fail-closed). The evaluator, the gate settings it points to, the
owner records and the decision record's SHA-256 are all read from this root, and an `--authorized-by` given as
a relative path is resolved against this root, not the working folder. Without a governance root, a run with
`--authorized-by` and a run without a record on a registry with no policy (v1) are refused before the decision,
and the reason says there is no governance root (`logs/refusal.txt`, exit code 3). A run without
`--authorized-by` on registry v2, that is, the standing permission, is not affected. The decision record file
cannot be read then, so `generation_decision.decision_record_sha256` is
`{"status": "unavailable", "reason": "... is not in this checkout"}` instead of a hash.

**The tracing record.** Every run record (`_runs/run_<identity>/`) keeps the tracing record of §4.1: the source
hash list (`source_manifest.json`), the code revision and `smpl18` revision, and the path of the decision record
that was the basis for generating. If a licence is withdrawn, these are used to find the affected outputs. For
bundles built from a corpus (`hknu`, `gaitex`, `addbiomechanics`), the corpus's own record names its sources
only by subject and trial names or partial hashes, but the `source_files` block of the run record's
`source_manifest.json` records every source file that run's corpus and bundle stages could read, with its
whole-file SHA-256 (§4.1). For a run that reused a corpus, the corpus sources are in the record of the run that
built it, and this record has that corpus's fingerprint.

**Bundles stay on that PC.** Registration goes only into that PC's catalog (`<data root>/experimental/catalog`).
Sharing or publishing needs separate approval.

---

## 10. Known mismatches

Where the sealed documents and reality disagree is recorded here. **It is reported, not fixed** (the parent
project's rule for sealed documents).

| Location | What the sealed document says | What is actually the case |
|---|---|---|
| `PIPELINE_GOVERNANCE.md` §0 | Three sources, `AMASS`, `PRISM`, `AddBiomechanics` | **5** (+`gaitex`, `hknu`) |
| `PIPELINE_GOVERNANCE.md` §0 | `6 IMU Small` | **8-channel** |
| `PIPELINE_GOVERNANCE.md` §0 | `15-joint Large` | **18-joint** |
| `PIPELINE_GOVERNANCE.md` §10.1 | `run_id` is the identity hash | `run_id` is a take label; §3.1 works around it with `run_identity` |
| `PIPELINE_GOVERNANCE.md` §2 | Pipeline location `src/soma_synthetic_imu/` | The generation pipeline moved to the separate repository `soma-synth` (package `soma_synth`, the parent project's submodule `packages/soma-synth`) (ADR-0041 decision 1) |
| `LOCAL_DATA_PLANE.md` §1, §5, §8 | Assumes one fixed data area | On each PC `SOMA_DATA_ROOT` is the output area and the sources are `SOMA_SOURCE_ROOT` (ADR-0041 decision 3). Distributing originals on the shared drive only after §8 is updated |

Resolving these is a review-set matter that includes promoting the M0 evaluator. The last two rows (§2,
`LOCAL_DATA_PLANE`) are fixed in the next M0 promotion review set (ADR-0041).

---

## 11. Operator reference

What an operator meets most often, in one place. The rules behind each item are in the sections cited.

**Exit codes**

| Code | `run` | `stage-sources` |
|---|---|---|
| 0 | every stage succeeded | every listed file is present and matches |
| 1 | a stage failed (including a validation FAIL) | a file could not be staged (missing from `--from`, copy hash mismatch), or with `--verify-only` a file is missing or differs |
| 2 | locations or the registry cannot be resolved (`SOMA_DATA_ROOT` unset, relative or missing; unknown source) | the list, a folder or an argument cannot be used |
| 3 | **refused** — the reason is on standard error | refused — a destination file differs from the list (nothing copied), or the destination location |

**Common refusals (exit code 3)** (§1.3)

| Gist of the message | What clears it |
|---|---|
| `inside ... which is read-only` (`extracted`, `raw_archives`, a source folder, the body models, `_superseded`, `_manifest_backfill`) | send the output to a lineage folder or a scratch folder |
| `it already holds ...` (the bundle or corpus folder is not empty) | `--replace-existing` (for a corpus also `--rebuild-corpus`), or an empty folder |
| `selects part of the source ... production lineage directory` | send a sample to a scratch folder |
| `sets --out, which the runner controls` | the output is `dataset_dir`, the corpus `--corpus`, the sources `SOMA_SOURCE_ROOT` |
| `no _superseded/<name>_* folder ... holds a copy` | put the evidence in `_superseded/` before replacing a production folder ([retention rule](RETENTION_RULES.md) 1.2) |
| `its generation did not finish -- .generating is there` | that bundle is unfinished; rebuild it with `--replace-existing` |

`warning: ... cloud-synchronised folder` and `team shared drive` are warnings, not refusals; the run continues.
To silence them, move the data root and outputs to a local disk.

**First-sample selection values** (case-sensitive). PRISM's `--only` takes the **subject id** `prism_subj001`,
not the source folder name (`subj001` is refused with "no PRISM take of that subject"). HKNU's and GAITEX's
`--subjects` take the source folder names as they are (`S01`, `austra`). Pass them in the `=` form
(`--bundle-arg=--only --bundle-arg=prism_subj001`); for a source with a corpus, send the corpus to the scratch
folder too (`--corpus <scratch>/hknu_smpl24_paired` with the same `--corpus-arg` selection).

**Locks** (§1.3, §4.1). The generate stage creates `<folder>.lock` beside the bundle (and corpus) folder and
deletes it when it ends. The refusal names the lock's pid, host and start time. If that process no longer
exists (`tasklist /FI "PID eq <pid>"` or `ps -p <pid>`), a dead run left it: delete it by hand. The runner
never deletes someone else's lock. The catalog holds an operating-system lock on `.catalog.lock`; if another
registration holds it, the writer waits up to 900 seconds and stops with `CatalogLocked`. Never delete
`.catalog.lock` (its presence is normal).

**`<folder>.replaced-<run>` / `<folder>.failed-<run>`** (§1.3). The previous generation moved aside by
`--replace-existing`, and the output of a failed replacement. On success the previous generation in a scratch
folder is deleted, and that of a production folder stays until the owner approves its deletion. On failure the
new output becomes `.failed-<run>` and the previous generation is renamed back. While either is beside it, the
folder is not replaced again: to accept the new generation delete `.replaced-*` (a production folder's, after
approval); to go back, clear the current folder and rename `.replaced-*` back. Inspect `.failed-*`, then
delete it.

**Staging leftovers** (§2.3). An interrupted `stage-sources` can leave unverified copies named
`.<name>.<random>.staging`. They are not source files, are reported as `LEFTOVER`, and are never deleted by the
command: delete them by hand.

**Catalog history size** (§4.1). Registering a bundle for the first time writes one gzip record of about 55
bytes per asset: `gaitex` (360 assets) about 20 KB, `prism` (900) about 50 KB, `hknu` (1,400) about 80 KB,
`amass` (41,305) about 2.4 MB, `addbiomechanics` (202,300) about 11 MB. Superseding a whole version is about
twice that; re-registering a rebuilt bundle is about three times the changed assets' share, and nothing
changed means no record.

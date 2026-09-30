"""Render the standard's per-source table from the registry, so the two cannot drift.

A hand-written table of how the datasets differ goes stale: it is right when written and
becomes wrong the day a dataset is added, because the same facts are written down twice.

So §8 of GENERATION_PIPELINE_STANDARD.md is generated between markers rather than maintained.
A v2 registry adds each source's bundle lineage, its corpus stage and post-steps, and the standing
generation policy; a v1 registry renders as it always did.
"""

from __future__ import annotations

from pathlib import Path

from soma_synth.pipeline import stages as stages_mod

BEGIN = "<!-- BEGIN GENERATED: source pipelines -->"
END = "<!-- END GENERATED: source pipelines -->"


def _code(values) -> str:
    return ", ".join(f"`{value}`" for value in values) or "—"


def render_table(registry: stages_mod.PipelineRegistry | None = None) -> str:
    reg = registry or stages_mod.default_registry()
    v2 = reg.schema == stages_mod.SCHEMA_V2

    lines = [
        BEGIN,
        "",
        f"> Generated. The source is [`{reg.path.name}`](../../configs/datasets/{reg.path.name});",
        "> `soma-synth pipeline-doc --render` rewrites this block. Do not edit it by hand.",
        "",
    ]
    if v2:
        lines += [
            "| source | `small_mode` | emitter | resume predicate | parallel | entry point | bundle lineage | corpus | smpl18 profile |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
    else:
        lines += [
            "| source | `small_mode` | emitter | resume predicate | parallel | entry point | smpl18 profile |",
            "|---|---|---|---|---|---|---|",
        ]
    for name in sorted(reg.sources):
        source = reg.for_source(name)
        profile = f"`{source.smpl18_profile}`" if source.smpl18_profile else "—"
        row = (f"| `{name}` | `{source.small_mode}` | `{source.emitter}` | "
               f"`{source.resume}` | `{source.parallel}` | `{Path(source.entrypoint.script).name}` | ")
        if v2:
            corpus = f"`{source.corpus.lineage}`" if source.corpus else "—"
            row += f"`{source.bundle_lineage}` | {corpus} | "
        lines.append(row + f"{profile} |")

    lines += ["", "**What the resume predicates mean**", ""]
    for key in sorted(reg.resume_predicates):
        lines.append(f"- `{key}` — {reg.resume_predicates[key]['description']}")

    inline = sorted(n for n in reg.sources if not reg.for_source(n).uses_shared_emitter())
    shared = sorted(n for n in reg.sources if reg.for_source(n).uses_shared_emitter())
    lines += [
        "",
        (f"**Shared emitter**: {', '.join(f'`{n}`' for n in shared)} — "
         f"**Own implementation**: {', '.join(f'`{n}`' for n in inline)}"),
    ]

    if v2:
        lines += ["", "**Generation stages (corpus → bundle → post-steps)**", ""]
        for name in sorted(reg.sources):
            source = reg.for_source(name)
            entry = source.entrypoint
            parts = []
            if source.corpus:
                corpus = source.corpus
                parts.append(
                    f"corpus `{Path(corpus.script).name} {corpus.output_arg} <{corpus.lineage}>` "
                    f"(fingerprint `{corpus.fingerprint}`, passed to the bundle as `{corpus.bundle_arg}`)"
                )
            parts.append(
                f"bundle `{Path(entry.script).name} {entry.output_arg} <{source.bundle_lineage}>`")
            for step in source.post_steps:
                parts.append(f"post-step `{Path(step.script).name} {' '.join(step.args)}`")
            lines.append(f"- `{name}` — " + " → ".join(parts))
        lines += ["", "**Environment variables read**", ""]
        for name in sorted(reg.sources):
            lines.append(f"- `{name}` — {_code(reg.for_source(name).reads_env)}")

    lines += ["", "**Body model sets**", ""]
    for name in sorted(reg.body_model_sets):
        declared = reg.body_model_sets[name]
        users = sorted(n for n in reg.sources if reg.for_source(n).body_model_set == name)
        location = f"`{declared.relative_path}` under the data root"
        if v2:
            location += " (or `SOMA_BODY_MODEL_DIR` when set; `pipeline.paths.body_model_dir()`)"
        lines.append(
            f"- `{name}` — {location}; the file for each sex is chosen by "
            f"`{declared.resolver}`. Used by: {', '.join(f'`{n}`' for n in users)}"
        )

    policy = reg.generation_policy
    if policy is not None:
        lines += [
            "",
            "**Generation policy (`generation_policy`)**",
            "",
            (f"- Basis: `{policy.basis}` — {policy.adr} (`{policy.adr_path}`), "
             f"decision record `{policy.decision_record}` {policy.decision_section}"),
            (f"- Class `{policy.artifact_class}`, distribution scope `{policy.distribution_scope}`, "
             f"`releases_holds: {str(policy.releases_holds).lower()}`"),
        ]
    lines += ["", END]
    return "\n".join(lines)


def write_into(document: Path | str, registry: stages_mod.PipelineRegistry | None = None) -> Path:
    """Replace the generated block in ``document``, or append it if the markers are absent."""
    path = Path(document)
    text = path.read_text(encoding="utf-8")
    block = render_table(registry)

    if BEGIN in text and END in text:
        head, _, rest = text.partition(BEGIN)
        _, _, tail = rest.partition(END)
        text = head + block + tail
    else:
        text = text.rstrip("\n") + "\n\n" + block + "\n"

    path.write_text(text, encoding="utf-8", newline="\n")
    return path

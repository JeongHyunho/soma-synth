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
        f"> 생성물이다. 원본은 [`{reg.path.name}`](../../configs/datasets/{reg.path.name}) 이고,",
        "> `soma-synth pipeline-doc --render` 가 이 블록을 다시 쓴다. 손으로 고치지 말 것.",
        "",
    ]
    if v2:
        lines += [
            "| source | `small_mode` | emitter | 재개 판정 | 병렬 | 진입점 | 번들 lineage | 코퍼스 | smpl18 프로파일 |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
    else:
        lines += [
            "| source | `small_mode` | emitter | 재개 판정 | 병렬 | 진입점 | smpl18 프로파일 |",
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

    lines += ["", "**재개 판정의 뜻**", ""]
    for key in sorted(reg.resume_predicates):
        lines.append(f"- `{key}` — {reg.resume_predicates[key]['description']}")

    inline = sorted(n for n in reg.sources if not reg.for_source(n).uses_shared_emitter())
    shared = sorted(n for n in reg.sources if reg.for_source(n).uses_shared_emitter())
    lines += [
        "",
        (f"**공유 emitter 사용**: {', '.join(f'`{n}`' for n in shared)} — "
         f"**미사용(자체 구현)**: {', '.join(f'`{n}`' for n in inline)}"),
    ]

    if v2:
        lines += ["", "**생성 단계 (코퍼스 → 번들 → 후처리)**", ""]
        for name in sorted(reg.sources):
            source = reg.for_source(name)
            entry = source.entrypoint
            parts = []
            if source.corpus:
                corpus = source.corpus
                parts.append(
                    f"코퍼스 `{Path(corpus.script).name} {corpus.output_arg} <{corpus.lineage}>` "
                    f"(지문 `{corpus.fingerprint}`, 번들에 `{corpus.bundle_arg}` 로 전달)"
                )
            parts.append(
                f"번들 `{Path(entry.script).name} {entry.output_arg} <{source.bundle_lineage}>`")
            for step in source.post_steps:
                parts.append(f"후처리 `{Path(step.script).name} {' '.join(step.args)}`")
            lines.append(f"- `{name}` — " + " → ".join(parts))
        lines += ["", "**읽는 환경 변수**", ""]
        for name in sorted(reg.sources):
            lines.append(f"- `{name}` — {_code(reg.for_source(name).reads_env)}")

    lines += ["", "**바디 모델 세트**", ""]
    for name in sorted(reg.body_model_sets):
        declared = reg.body_model_sets[name]
        users = sorted(n for n in reg.sources if reg.for_source(n).body_model_set == name)
        location = f"data root 기준 `{declared.relative_path}`"
        if v2:
            location += " (`SOMA_BODY_MODEL_DIR` 가 있으면 그 폴더, `pipeline.paths.body_model_dir()`)"
        lines.append(
            f"- `{name}` — {location}, 성별에 따른 파일 선택은 "
            f"`{declared.resolver}` 가 정한다. 사용: {', '.join(f'`{n}`' for n in users)}"
        )

    policy = reg.generation_policy
    if policy is not None:
        lines += [
            "",
            "**생성 정책 (`generation_policy`)**",
            "",
            (f"- 근거: `{policy.basis}` — {policy.adr} (`{policy.adr_path}`), "
             f"결정 기록 `{policy.decision_record}` {policy.decision_section}"),
            (f"- 등급 `{policy.artifact_class}`, 배포 범위 `{policy.distribution_scope}`, "
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

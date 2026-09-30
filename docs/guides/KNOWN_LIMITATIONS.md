# KNOWN_LIMITATIONS.json — a common format for what a bundle cannot promise

> Written 2026-09-14 · schema `dataset_limitations_v1` · code `src/soma_synth/contracts/dataset_limitations.py`
> Source of the values `configs/datasets/known_limitations_v1.yaml` · renderer `scripts/write_known_limitations.py`

## 1. Why it exists

Every bundle has limitations — a channel substituted by a model because there is no real sensor, an insole
whose heading drifts because it has no magnetometer, a source that gives only pose parameters, a retarget
that had to repair wrapped coordinates. Until now each bundle described them in prose, in a different
section of its `DATA_DESCRIPTION_EN.md` and in a different voice. Comparing two bundles meant reading two
essays, and no program could read either.

This format **fixes one shape**. The values live per bundle in one configuration file in the repository and
are rendered to `KNOWN_LIMITATIONS.json` at the bundle root. The validator **requires** that file (registry
`universal.bundle_files`), checks it against the schema and checks that it is **the same** as the
repository's configuration — editing only the bundle's copy by hand is a FAIL.

## 2. The shape of one entry

```yaml
- id: prism.insole_heading_unreferenced        # <source>.<snake_case_topic>, stable
  kind: hardware_limit                          # table below
  severity: caution                             # info | caution | do_not_use
  scope: {level: site, sites: [foot_l, foot_r]} # bundle | site | array | take | subject
  statement: >-                                 # what is limited, one or two sentences
    The insoles are six-axis; their heading is the gyro integral and drifts.
  consequence: >-                               # what a consumer must therefore not do
    Do not use the measured insole heading as an absolute reference.
  evidence: small_reference.npz imu_orientation_absolute_heading = [T,T,T,T,T,T,F,F]
  status: mitigated                             # open | mitigated | fixed
  since: "2026-09-09"
  mitigation: heading_synth_reference.npz carries a heading transferred from the shank.
  related: [prism.foot_gyro_axis_relabel]       # optional
```

| `kind` | Meaning — **who can remove it** |
|---|---|
| `hardware_limit` | The sensor cannot measure it (six axes have no heading reference) |
| `source_defect` | A defect in the source data; we did not make it |
| `source_scope` | The source never recorded it (no GRF, no measured IMU) |
| `pipeline_defect` | Our processing made it — we can fix it on our side |
| `proxy` | Another channel stands in for a missing sensor |
| `approximation` | A value derived under a stated simplifying assumption |
| `coverage` | Some takes, sites or subjects are missing or excluded |
| `rights` | What may and may not be done with this data |

| `severity` | What the consumer does |
|---|---|
| `info` | Good to know; no change in use |
| `caution` | Usable, but keep `consequence` in mind |
| `do_not_use` | **Must not be used** for the stated purpose |

`status: mitigated` requires `mitigation`, and `status: fixed` requires `fixed_in`. Fields not in the schema
are **refused** — so that what happened to `source_attribution`, which grew a different shape in each
bundle, does not happen again.

This schema only checks presence and enumerations. It holds no thresholds or tolerances
(`DETERMINISTIC_EXECUTION_CONFIG_HOLD`). A limitation is written as a **statement, scope and evidence**,
not as a score.

## 3. How to change it

1. Edit the bundle's block in `configs/datasets/known_limitations_v1.yaml`. For a new bundle, add a block —
   it needs at least one `rights` entry.
2. Render: `python scripts/write_known_limitations.py <bundle_dir> [...]`.
   When the content is the same, `generated_utc` is preserved, so a re-render without changes is
   byte-identical.
3. Validate: `python scripts/write_known_limitations.py <bundle_dir> --check`, or validate the bundle
   (`soma-synth run ... --level L4`) — a missing file is an L0 FAIL, a schema violation or a configuration
   mismatch an L4 FAIL.
4. If there is a shared copy, the parent project's `push_sample_data.py` moves only the changed files
   (soma-synth does not upload bundles).

## 4. What is written and what is not

**Written**: what the bundle itself documents (description, manifest, npz provenance arrays) and what the
repository records (registry waivers, ADRs, investigation documents). The evidence (`evidence`) is an exact
place a reader can open and check — a section name, a manifest key, an array name, a repository path.

**Not written**: guesses. Something suspected but without evidence goes into an investigation document, not
into this file.

**Relation to registry waivers**: a waiver is an *exemption from a validation rule* (it has a deadline and
is a FAIL once it expires); a limitation entry is a *statement about the data* (it has no deadline). The two
may overlap and are linked with `related`. A limitation can remain after its waiver closes — filling a field
does not give an insole a magnetometer.

## 5. First application

On 2026-09-14 the existing prose caveats of five bundles (`prism_faithful_full_v1`, `hknu_unified8_v1`,
`gaitex_unified8_v1`, `amass_faithful_full_v2`, `addbio_unified8_v2`) were moved into this format. The prose in
each bundle's `DATA_DESCRIPTION_EN.md` was left as it was — it is the description for people, and this file
is for machines and comparison. If the two disagree, this file and the configuration are authoritative.

Later the same day the version suffixes disappeared from bundle names (parent project retention rules §1).
The configuration's block keys are lineage names (`prism_faithful_full`, `hknu_unified8`, `gaitex_unified8`,
`amass_faithful_full`, `addbio_unified8`), and when a lineage is regenerated the same block is rendered into
the new bundle — so a fixed entry is not deleted but kept with `status: fixed` and `fixed_in`. The generation
pipeline runner renders it right after generation (`GENERATION_PIPELINE_STANDARD.md` §9), and existing
bundles are rendered with `scripts/write_known_limitations.py`.

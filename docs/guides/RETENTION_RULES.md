# Retention rules for synthetic datasets

> This document states, in this repository's words, the data retention rules soma-synth follows. The
> authority is the data retention rules of the parent project (SOMA Synthetic IMU).
> If this document and those rules disagree, **the parent project's rules win.** The numbering (1.1–1.7)
> and the meaning of each rule are the same as section 1 of the parent project's rules.
>
> In this repository's messages, comments and documents, "retention rule 1.x" means the rule with that
> number below. For example, "retention rule 1.5" in a runner refusal is 1.5 (deletion needs the owner's
> approval).
>
> Data paths are relative to the data root (`<SOMA_DATA_ROOT>`).

---

## 1. Retention rules for synthetic datasets (enacted 2026-09-07, revised 2026-09-14)

### 1.1 One directory per lineage, no version in the name

**lineage** = the series of outputs that share the three values `(source, stage, spec_id)`.
The directory name is `<source>_<stage>` and **carries no version suffix** (revised 2026-09-14).

| lineage directory | stage |
|---|---|
| `hknu_smpl24_paired` | HKNU retarget corpus (with the fit correction) |
| `hknu_unified8` | HKNU unified8 bundle |
| `prism_faithful_full` | PRISM measured-small bundle |
| `amass_faithful_full` | AMASS unified8 bundle |
| `gaitex_smpl24` / `gaitex_unified8` | GAITEX retarget corpus / bundle |
| `addbio_smpl24_raw` / `addbio_smpl24_synth` / `addbio_unified8` | two AddBiomechanics corpora / bundle |

Generations are told apart not by name but by `INDEX.json.generated_utc` inside the bundle, the run
record `_runs/run_<identity>/`, and the evidence of earlier generations under `_superseded/`.

### 1.2 A regeneration replaces the old one under the same name after it passes validation

- Regenerating a lineage goes into **the same directory name**. It is not kept side by side with the
  previous generation.
- Order: (1) copy the previous generation's evidence files (1.3(b)) to `_superseded/` and write
  SUPERSEDED.md, (2) delete the previous payload with the owner's approval (1.5), (3) generate under the
  same name and pass L4 `--full` validation. If the new generation fails validation, that fact stays in
  `_runs/`, and the evidence must still explain the previous generation — that is why (1) comes before (2).
- Generation is done with `soma-synth run`. The runner leaves the run record, `KNOWN_LIMITATIONS.json`,
  validation, README and registration together ([generation pipeline standard](GENERATION_PIPELINE_STANDARD.md)).

### 1.3 Three preconditions that must hold before any deletion

No payload is deleted before **all** three are confirmed.

**(a) No live code reference.** Searching the whole repository for the directory name must find zero
references in code, configuration and tests. If there are references, **first move them to the lineage
name**, and delete only after that change passes. Since the 2026-09-14 revision every reference in the
repository uses a lineage name; mentions of `_vN` in documents and reports remain as historical records
of that generation.

**(b) Keep the evidence apart from the payload.** What is deleted is the take payload; the small files
that explain what that generation did are kept. Files to keep:

`INDEX.json` (and `INDEX_shard_*.json`), `VALIDATION_REPORT.json`, `validation_ledger.json`,
`SUMMARY.json`, `_run.json`, `DATA_DESCRIPTION_EN.md`, `DATASET_DESCRIPTION_EN.md`,
`README.md`, `KNOWN_LIMITATIONS.json`, `PROVENANCE_RETROFIT.json`, `PUSH_RECEIPT.txt`,
and any other small policy or record files at the bundle root.

Copy them under `_superseded/<name of the directory being deleted>/`, compare them by sha256, then delete
the payload. A generation that had a version suffix keeps that name (`_superseded/hknu_unified8_v1/`); a
generation after the revision goes to `_superseded/<lineage>_<generated_utc date>/`.

> Why this clause is needed: the previous generation's `INDEX.json` may be the only record of what was
> left out and why. For example, for a generation that excluded takes because of a defect (say, angle
> wrapping), only its `INDEX.json` lists the excluded takes and the reasons. Deleting the directory
> wholesale removes the only means of explaining what was left out and why.

**(c) Record the deletion.** In `_superseded/<name>/SUPERSEDED.md`, write the deletion date, the
replacement (or "none" and why), the reason for deletion, the file count and size before deletion, and the
commit that re-pointed the references.

### 1.4 Never delete

- **All source data** under `<SOMA_DATA_ROOT>/raw_archives/` and `<SOMA_DATA_ROOT>/extracted/` (and, when
  the source folders are kept elsewhere through `SOMA_SOURCE_ROOT`, everything under that too). No cleanup
  relaxes this rule.
- `<SOMA_DATA_ROOT>/README.md`, and the record files the data root may keep, `MASTER.md` (the data root's
  index) and `state/local_archive_inventory.json` (the list of source archives). Overwriting is forbidden
  too. A file that does not exist is left not existing.
- `.bak-*` files.
- `_superseded/`, `_runs/`, `_manifest_backfill/` — evidence and run records, not payload.
- Nothing on a team shared drive is deleted under these rules. Cleaning up shared copies is a separate
  approval matter.

### 1.5 Deletion needs the owner's approval

Deleting a payload cannot be undone. Deletion happens only after the data owner approves it: after the
preconditions (1.3) are confirmed and the evidence is moved, show the owner the **exact list and sizes** of
the directories to be deleted and get approval.

### 1.6 Naming rules

- A new lineage is `<source>_<stage>`. No version number goes in the name.
- A regeneration of the same lineage reuses the same name, in the order of 1.2.
- Directories that are obviously experiment leftovers, such as `_dryrun_*` and `xrun_*`, are not treated as
  lineages, and whoever made them is responsible for cleaning them up.

### 1.7 Record of past cleanups

The record of a past cleanup (for example, deleting the old generations with version suffixes after the
2026-09-14 revision) stays in the data root's evidence folder, that is, in the evidence files and
`SUPERSEDED.md` (1.3(b), (c)) under `_superseded/<name of the deleted directory>/`, and a deleted
generation is explained by that evidence. Every cleanup goes through the preconditions of 1.3 and the
approval of 1.5, and copies on a team shared drive are outside its scope (1.4).

# soma-synth

> **INTERNAL-ONLY.** Every bundle this code makes is `experimental_non_candidate` · `internal_only` and
> **stays on the PC that made it.** The licences of the source datasets apply.

soma-synth generates the SOMA synthetic IMU datasets (spec `qmd_unified8_smpl18`). From motion-capture
sources (AMASS, PRISM, HKNU, GAITEX, AddBiomechanics) it builds bundles of 8-channel virtual IMU data (Small)
and an 18-joint SMPL reference (Large), and validates, documents and registers them with one command,
`soma-synth run`.

The full specification is [`docs/guides/GENERATION_PIPELINE_STANDARD.md`](docs/guides/GENERATION_PIPELINE_STANDARD.md).
See also [`KNOWN_LIMITATIONS.md`](docs/guides/KNOWN_LIMITATIONS.md), [`RETENTION_RULES.md`](docs/guides/RETENTION_RULES.md)
and the viewer guide [`small_large_viewer.md`](docs/guides/small_large_viewer.md) (manual: `docs/manual/viewer_manual.docx`).

## Access

You need read access to two private repositories; send the owner your GitHub account:
`JeongHyunho/soma-synth` (this repository) and `JeongHyunho/smpl18` (the SMPL conversion rules, mounted
here as the submodule `packages/smpl18`). Without `smpl18` access, `git clone --recurse-submodules` fails
at the submodule with `repository not found`.

## Install

Baseline: CPython 3.13.5 (3.12+ works). Library versions are pinned in [`constraints.txt`](constraints.txt).

```powershell
# Windows (PowerShell)
git clone --recurse-submodules https://github.com/JeongHyunho/soma-synth.git
cd soma-synth
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -c constraints.txt -e packages/smpl18 -e ".[dev]"
.venv\Scripts\soma-synth --help
```

```bash
# macOS / Linux
git clone --recurse-submodules https://github.com/JeongHyunho/soma-synth.git
cd soma-synth
python3.13 -m venv .venv
.venv/bin/python -m pip install -c constraints.txt -e packages/smpl18 -e ".[dev]"
.venv/bin/soma-synth --help
```

If `packages/smpl18` is empty, run `git submodule update --init`. Tests: `python -m pytest -q`.
The examples below assume the venv is activated.

## Environment variables

| Variable | Meaning | Default |
|---|---|---|
| `SOMA_DATA_ROOT` | output root: bundles, corpora, run records, catalog | **required**; an existing absolute path |
| `SOMA_SOURCE_ROOT` | folder holding the source folders `amass`, `prism`, `gaitex`, `hknu_fullbody`, `addbiomechanics` | `<SOMA_DATA_ROOT>/extracted` |
| `SOMA_BODY_MODEL_DIR` | folder holding `SMPL_{MALE,FEMALE,NEUTRAL}_clean.npz` | `<SOMA_DATA_ROOT>/body_models/smpl` |

```powershell
$env:SOMA_DATA_ROOT = "E:\soma_data"            # persist with: setx SOMA_DATA_ROOT "E:\soma_data"
```

```bash
export SOMA_DATA_ROOT="$HOME/soma_data"         # put it in ~/.zshrc or ~/.bashrc to keep it
```

## Get the sources and body models

The sources are in the team Google Drive folder **`SOMA_AI_SharedData/00_Dataset`**, one folder per source,
with a `SHA256SUMS` list from the owner. Copy them onto a local disk with `stage-sources`: it copies only
the listed files that are missing, verifies each against the list, and never deletes or overwrites anything.

```bash
soma-synth stage-sources --from "<local path of 00_Dataset>" --sums SHA256SUMS --sources prism,hknu
soma-synth stage-sources --sums SHA256SUMS --sources prism,hknu --verify-only   # check only
```

Sizes: `amass` 12.1 GB, `prism` 5.0 GB, `gaitex` 17.7 GB, `hknu_fullbody` 27.9 GB, `addbiomechanics` 568 GB.
Fetch `addbiomechanics` with [rclone](https://rclone.org) (a Google Drive remote on the shared drive), not
through the Drive desktop app, which is unreliable at that size; then stage from the download:

```bash
rclone copy "<remote>:00_Dataset/addbiomechanics" "<download folder>/addbiomechanics" --progress
soma-synth stage-sources --from "<download folder>" --sums SHA256SUMS --sources addbiomechanics
```

Body models: copy `SMPL_MALE_clean.npz`, `SMPL_FEMALE_clean.npz` and `SMPL_NEUTRAL_clean.npz` from
`00_Dataset/body_models/smpl` into `SOMA_BODY_MODEL_DIR`. All three are needed. They fall under the MPI SMPL
non-commercial licence.

## First sample run

Build one PRISM subject into a scratch folder, validate it, and keep the run record out of the production
record folder:

```powershell
$S = "$env:SOMA_DATA_ROOT\tmp\first_sample"
soma-synth run --source prism "$S\prism_faithful_full" --bundle-arg=--only --bundle-arg=prism_subj001 `
    --stop-after validate --runs-root "$S\_runs"
```

```bash
S="$SOMA_DATA_ROOT/tmp/first_sample"
soma-synth run --source prism "$S/prism_faithful_full" --bundle-arg=--only --bundle-arg=prism_subj001 \
    --stop-after validate --runs-root "$S/_runs"
```

Keep the lineage name as the folder name (`prism_faithful_full`): the runner finds the bundle's limitations
block by it. The summary lists each stage (`ok` or `FAIL`) and the run record; the result is in the bundle's
`VALIDATION_REPORT.json`. Delete the scratch folder when you are done.

## Full runs

```bash
soma-synth run --source prism                       # prism_faithful_full
soma-synth run --source hknu                        # corpus hknu_smpl24_paired -> bundle hknu_unified8
soma-synth run --source gaitex                      # gaitex_smpl24 -> gaitex_unified8
soma-synth run --source amass --jobs 8              # amass_faithful_full
soma-synth run --source addbiomechanics --jobs 8    # addbio_smpl24_raw -> addbio_unified8 (hours)
```

`run` does generate → validate → readme → register and stops at the first failure. Every run builds into an
empty folder; a non-empty one needs `--replace-existing` (and `--rebuild-corpus` for a corpus). Re-validate
an existing bundle with `soma-synth run --source hknu --start-at validate --stop-after validate --full`.

## Where outputs go

| What | Where |
|---|---|
| Bundles and corpora | `<SOMA_DATA_ROOT>/runs/experimental_generation_poc_demo/<lineage>/` |
| Run records (config, environment, source hashes, logs) | `<bundle parent>/_runs/run_<identity>/` (or `--runs-root`) |
| Catalog | `<SOMA_DATA_ROOT>/experimental/catalog/` |

Keep the run records: if a source licence is withdrawn, they identify the affected bundles.

## Rules

- **Bundles stay on your PC.** Sharing them, or publishing anything, needs the owner's separate approval.
  soma-synth has no push command.
- **Do not redistribute the sources or the SMPL models.** Their licences apply; copy them only to people
  covered by the same licences.
- **Never modify or delete the sources** (`extracted/`, `raw_archives/`, `SOMA_SOURCE_ROOT`). `stage-sources`
  only adds missing files. Evidence folders (`_superseded/`, `_runs/`, `_manifest_backfill/`) are never deleted.
- **Synchronised folders only warn.** Output inside Dropbox, OneDrive, Google Drive, iCloud Drive, Synology
  Drive or a team shared drive (`SOMA_SHARED_DRIVE_NAMES`) prints a `warning:` and continues. Use a local disk.

## Troubleshooting

- Exit codes of `run`: 0 success, 1 a stage failed, 2 paths or registry unresolved (for example
  `SOMA_DATA_ROOT` unset), 3 refused (the reason is on standard error).
- `import smpl18` fails: the submodule is empty; `git submodule update --init`, then reinstall.
- Refusals, locks, `.replaced-*` / `.failed-*` folders and exit codes are explained in
  [the standard, §11](docs/guides/GENERATION_PIPELINE_STANDARD.md#11-operator-reference).
- Commands and options: `soma-synth --help`, `soma-synth <command> --help`.

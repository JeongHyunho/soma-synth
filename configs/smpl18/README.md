# smpl18 profiles kept by the SOMA project

This directory holds the SOMA project's own `smpl18` profiles: the datasets that are internal
to the project (PRISM, HKNU) and the correspondence tables they need. Public datasets
(AddBiomechanics, GAITEX, AMASS) ship as example profiles inside the `smpl18` package under
`packages/smpl18/configs/profiles/`; profiles of datasets internal to the project are kept here.

```
configs/smpl18/
  profiles/prism.yaml                 SMPL parameters in per-take pickles
  profiles/hknu.yaml                  Visual3D joint centres and segment rotations in .mat files
  correspondence/visual3d_hknu.yaml   HKNU segment and landmark names -> SMPL-24 joints
```

A profile is data, never code: it says which source kind and file format a dataset is, how its
files are laid out, which field means what, its units and frame, its correspondence, repairs and
settings. The schema is `smpl18_profile_v1`, documented field by field in
`packages/smpl18/docs/profile-schema.md`. Every value in these files cites the driver or module
it was lifted from, so the profile can be checked against the code it replaces.

## Where `smpl18` finds them

`smpl18` resolves a profile name in this order: an existing path as given; the package's own
`configs/profiles/<name>.yaml`; then `$SHARED_DATASET_PATH/smpl18/profiles/<name>.yaml`. The
environment variable `SHARED_DATASET_PATH` names the shared drive where the project publishes
its datasets, and this directory is published there as `smpl18/` by the project's push
procedure (`docs/guides/SAMPLE_DATA_DISTRIBUTION.md`), so a teammate with the drive mounted can
run

```bash
export SHARED_DATASET_PATH=/path/to/<team shared drive>
smpl18 profile validate hknu
smpl18 profile show prism
```

without a checkout of this repository. From a checkout the same profiles are reachable by path:

```bash
smpl18 profile validate configs/smpl18/profiles/hknu.yaml
```

Files a profile references (`correspondence`, `settings`, ...) resolve relative to the profile's
directory first, then to the package's `configs/`, then to `$SHARED_DATASET_PATH/smpl18/`. That
is why `prism.yaml` and `hknu.yaml` name `settings/default.yaml` without a `../`: the package's
shared engine settings are found in the package, and the same reference keeps working once both
are published side by side on the drive. `smpl18 profile show` prints every resolved path with
its SHA-256, and the converter writes those hashes into every corpus manifest.

## Rules

- No motion data, body models or subject tables live here; profiles only describe where those
  are under the `--input` root the converter is given.
- A change to a value in a profile changes what a corpus would contain. Cite the reason next to
  the value, as the existing entries do, and regenerate under the project's hold and approval
  rules (the parent project's generation hold and approvals); a profile authorises no generation.
- Nothing in `packages/smpl18/src/` may name a dataset. If a dataset needs something the schema
  cannot say, the schema or a source kind grows; a special case in code is a defect.

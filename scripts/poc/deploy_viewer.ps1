<#
  deploy_viewer.ps1 — deploys the SOMA Small/Large viewer (copies it into the folder given with -Dest).

  * Run data never goes into the viewer. The viewer reads the bundles of the PC it runs on:
    the bundles under $SOMA_DATA_ROOT\runs\experimental_generation_poc_demo, or a folder given with
    --folder <folder> (SOMA_VIEWER_DATASET). Bundles stay on the PC that made them (ADR-0041).
  * The licence-gated SMPL models (SMPL_MALE/FEMALE/NEUTRAL_clean.npz) are excluded by default (MPI
    non-commercial, no redistribution + ADR-0005 INTERNAL-ONLY). The models belong to the viewer, not
    to a dataset, so only whether they are included is gated.
    All three models are needed: a subject's betas live in the shape space of their own model, and
    GAITEX, which records no sex, was fitted on the neutral model — on another model it is not that
    subject's body.
  * Copying happens only with -Execute (before that the plan is printed only = dry-run).
  * Including the models needs both -IncludeRestricted and -IUnderstandLicense. Models missing from
    the bundle's data\ are then filled first from -ModelDir (default $SOMA_BODY_MODEL_DIR, else
    $SOMA_DATA_ROOT\body_models\smpl), so that the bundle has all three.
  * -Dest is required. There is no default destination — no shared drive is searched for. Deploying
    to a shared drive is a separate decision, made by naming that folder as -Dest.
  * -Source is required. It is the portable viewer bundle (viewer scripts, Blender, copies of the
    presentation package render_amass and gear_kit). This repository does not build that bundle:
    point at the bundle you received separately.

  Examples:
    # 1) preview what would be copied (moves nothing)
    pwsh scripts\poc\deploy_viewer.ps1 -Source <viewer bundle> -Dest <deploy folder>
    # 2) deploy code + Blender + manual only (models excluded — licence-safe)
    pwsh scripts\poc\deploy_viewer.ps1 -Source <viewer bundle> -Dest <deploy folder> -Execute
    # 3) include the SMPL models too (only when the recipients hold the same SMPL licence!)
    pwsh scripts\poc\deploy_viewer.ps1 -Source <viewer bundle> -Dest <deploy folder> -Execute `
      -IncludeRestricted -IUnderstandLicense
    # 4) temporary: also ship fallback run copies (this moves run data, so it needs the same approval as sharing a bundle)
    pwsh scripts\poc\deploy_viewer.ps1 -Source <viewer bundle> -Dest <deploy folder> -Execute `
      -IncludeRestricted -IUnderstandLicense -FallbackRun
#>
param(
  # Deployment source (the portable viewer bundle). Required — there is no default.
  [string]$Source = "",
  # Destination. Required — there is no default (no shared drive is searched for).
  [string]$Dest   = "",
  # Local source of the licensed SMPL models. Models missing from the bundle's data\ are filled from here (only with -IncludeRestricted).
  # Empty means $SOMA_BODY_MODEL_DIR, else $SOMA_DATA_ROOT\body_models\smpl.
  [string]$ModelDir = "",
  # The repository's manual folder (docs\manual of the checkout holding this script).
  [string]$ManualDir = (Join-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) "docs\manual"),
  [switch]$Execute,
  [switch]$IncludeRestricted,
  [switch]$IUnderstandLicense,
  # STOPGAP: ship data\runs\ with the viewer after all. Needed only while the published dataset
  # carries no smpl_trans — without it the viewer finds no viewable run on a machine without any
  # bundle. Drop this switch once the dataset publishes smpl_trans.
  [switch]$FallbackRun
)

$ErrorActionPreference = "Stop"
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}

if (-not $Source) { throw "pass -Source <portable viewer bundle>; there is no default source" }
if (-not (Test-Path $Source)) { throw "source not found: $Source" }

# The destination must be named. There is no default, and whoever writes a folder as -Dest decides whether it is a shared drive.
if (-not $Dest) { throw "pass -Dest <folder to deploy the viewer into>; there is no default destination" }

if (-not $ModelDir) {
  if ($env:SOMA_BODY_MODEL_DIR) { $ModelDir = $env:SOMA_BODY_MODEL_DIR }
  elseif ($env:SOMA_DATA_ROOT) { $ModelDir = Join-Path $env:SOMA_DATA_ROOT "body_models\smpl" }
}

# Restricted (licensed) assets: excluded by default. Including them requires both switches.
$restricted = $false
if ($IncludeRestricted) {
  if (-not $IUnderstandLicense) {
    Write-Warning "To include the licensed SMPL models, also pass -IUnderstandLicense."
    Write-Warning "Those models are under a no-redistribution licence. Only when you have confirmed that the recipients hold the same SMPL licence."
    throw "aborted: -IncludeRestricted without -IUnderstandLicense"
  }
  $restricted = $true
}

# Run data is excluded by default — the viewer reads the bundles of the PC it runs on (under $SOMA_DATA_ROOT, or --folder).
# -FallbackRun is a temporary exception that ships run copies.
# The three licensed SMPL models. A subject's betas live in their own model's shape space, so all three are needed —
# drawing a female subject on the male model, or a GAITEX subject fitted on neutral on the male model, gives
# another person's body, not an approximation.
$modelFiles = @("SMPL_MALE_clean.npz", "SMPL_FEMALE_clean.npz", "SMPL_NEUTRAL_clean.npz")
$exclDirs  = @()
$exclFiles = @()
if (-not $FallbackRun) { $exclDirs += "runs" }
if (-not $restricted) { $exclFiles = $modelFiles }
if ($FallbackRun -and -not $restricted) {
  throw "aborted: -FallbackRun ships run data, so it needs -IncludeRestricted -IUnderstandLicense."
}

# First refresh the bundle's scripts from the repository. The deployment source is a bundle received
# separately; without this step, code fixed in the repository could be left out and old scripts shipped.
# The bundle's scripts\ is a curated subset, so only files already in the bundle are refreshed, by name —
# except that modules the repository's viewer has started to import ($requiredScripts) are added if the bundle lacks them.
$repoPoc = Join-Path (Split-Path $PSScriptRoot -Parent) "poc"
$bundleScripts = Join-Path $Source "scripts"
$requiredScripts = @("viewer_paths.py")
$refreshed = @()
if ((Test-Path $repoPoc) -and (Test-Path $bundleScripts)) {
  foreach ($name in $requiredScripts) {
    $repoFile = Join-Path $repoPoc $name
    $bundleFile = Join-Path $bundleScripts $name
    if ((Test-Path $repoFile) -and -not (Test-Path $bundleFile)) {
      if ($Execute) { Copy-Item $repoFile $bundleFile -Force }
      $refreshed += ($name + " (new)")
    }
  }
  foreach ($f in Get-ChildItem (Join-Path $bundleScripts "*.py")) {
    $repoFile = Join-Path $repoPoc $f.Name
    if ((Test-Path $repoFile) -and
        ((Get-FileHash $repoFile -Algorithm MD5).Hash -ne (Get-FileHash $f.FullName -Algorithm MD5).Hash)) {
      if ($Execute) { Copy-Item $repoFile $f.FullName -Force }
      $refreshed += $f.Name
    }
  }
}

# Fill licensed models missing from the bundle's data\ from the local source (meaningful only when deploying the models).
$modelsAdded = @()
$modelsMissing = @()
if ($restricted) {
  $bundleData = Join-Path $Source "data"
  if ($Execute) { New-Item -ItemType Directory -Force -Path $bundleData | Out-Null }
  foreach ($name in $modelFiles) {
    $bundleFile = Join-Path $bundleData $name
    if (Test-Path $bundleFile) { continue }
    $local = if ($ModelDir) { Join-Path $ModelDir $name } else { "" }
    if ($local -and (Test-Path $local)) {
      if ($Execute) { Copy-Item $local $bundleFile -Force }
      $modelsAdded += $name
    } else {
      $modelsMissing += $name
    }
  }
}

Write-Output "==================== SOMA viewer deployment ===================="
Write-Output " source : $Source"
Write-Output " dest   : $Dest"
Write-Output " manual : $ManualDir\viewer_manual.docx"
Write-Output (" mode   : {0}" -f ($(if ($Execute) { "EXECUTE (copies)" } else { "DRY-RUN (preview; run with -Execute)" })))
Write-Output (" assets : {0}" -f ($(if ($restricted) { "* licensed SMPL models included — licence confirmed" } else { "code + Blender + manual only (SMPL models excluded)" })))
Write-Output (" runs   : {0}" -f ($(if ($FallbackRun) { "* included temporarily (-FallbackRun) — this moves run data, so it needs the same approval as sharing a bundle" } else { "excluded — the viewer reads the bundles of the PC it runs on (SOMA_DATA_ROOT, --folder)" })))
if ($exclFiles) { Write-Output ("  excluded files   : {0}" -f ($exclFiles -join ", ")) }
if ($exclDirs)  { Write-Output ("  excluded folders : {0}" -f ($exclDirs -join ", ")) }
Write-Output (" scripts: {0}" -f ($(if ($refreshed) { ("refreshed from the repository -> " + ($refreshed -join ", ")) } else { "the bundle already matches the repository" })))
if ($restricted) {
  Write-Output (" models : {0}" -f ($(if ($modelsAdded) { ("added to the bundle <- $ModelDir : " + ($modelsAdded -join ", ")) } else { "all three are in the bundle's data\" })))
  if ($modelsMissing) { Write-Warning ("models in neither the bundle nor locally: " + ($modelsMissing -join ", ") + " — subjects of that sex are shown on a remaining model instead (recorded in the notice)") }
}
Write-Output "================================================================"

# robocopy arguments (/L = list only = dry-run)
$rcArgs = @($Source, $Dest, "/E", "/NFL", "/NDL", "/NJH", "/NP", "/R:1", "/W:1")
foreach ($d in $exclDirs)  { $rcArgs += @("/XD", $d) }    # name-based: excludes that dir at any depth
foreach ($f in $exclFiles) { $rcArgs += @("/XF", $f) }
if (-not $Execute) { $rcArgs += "/L" }

if (-not $Execute) {
  Write-Output "[dry-run] printing the robocopy plan only. Add -Execute to copy."
}

robocopy @rcArgs | Out-Null
$code = $LASTEXITCODE
Write-Output ("robocopy exit={0} (0-7 = normal)" -f $code)

if ($Execute) {
  # ship the manual docx + the distribution notice (reference.docx is the Quarto style template, so it is left out)
  $docx = Get-ChildItem (Join-Path $ManualDir "*.docx") -ErrorAction SilentlyContinue |
          Where-Object { $_.Name -ne "reference.docx" }
  if ($docx) {
    New-Item -ItemType Directory -Force -Path (Join-Path $Dest "manual") | Out-Null
    foreach ($m in $docx) {
      Copy-Item $m.FullName (Join-Path $Dest ("manual\" + $m.Name)) -Force
      Write-Output ("manual copied -> manual\" + $m.Name)
    }
  } else {
    Write-Warning "no manual docx in $ManualDir (build it first with build_manual.ps1)"
  }
  $notice = @"
SOMA Small/Large viewer — distribution notice (INTERNAL ONLY)

- The human body in this viewer is derived from the MPI SMPL body model (non-commercial research use, no redistribution).
- Use and keep it only as a lab-internal user holding the same SMPL licence.
- No public distribution or external sharing (ADR-0005 INTERNAL-ONLY).

Where the data is
- The viewer holds no data. It reads the bundles on this PC:
  runs\experimental_generation_poc_demo\<bundle>\ in the folder the SOMA_DATA_ROOT environment variable names.
- To look at another folder, put that folder in SOMA_VIEWER_DATASET or pass
  -- --folder <folder> as a Blender argument. Bundles stay on the PC that made them (ADR-0041).
"@
  if ($FallbackRun) {
    $notice += @"

Temporary note
- For PCs with no bundle to view, fallback run copies are included in data\runs\. Once this PC has a
  bundle, the viewer shows it first, and the fallback copies can be deleted.
"@
  }
  if (-not $restricted) {
    $notice += @"

- This distribution does not include the licensed SMPL models.
- To run it, add data\SMPL_MALE_clean.npz yourself.
- With data\SMPL_FEMALE_clean.npz as well, female subjects are shown with a female body; with
  data\SMPL_NEUTRAL_clean.npz as well, subjects fitted on neutral (all of GAITEX) are shown with that body.
  Without them the viewer falls back to the male model and says so on the console (the body is then not that subject's).
"@
  } elseif ($modelsMissing) {
    $notice += @"

- data\ holds only some of the SMPL models (missing: $($modelsMissing -join ', ')). The viewer
  chooses the model by the gender in anthro_reference; a subject of a missing sex is shown on a remaining
  model instead, and the console says so (the body is then not that subject's).
"@
  } else {
    $notice += @"

- data\ holds the three SMPL models (MALE / FEMALE / NEUTRAL). A subject's betas live in their own model's
  shape space, so the viewer chooses the model by the gender in anthro_reference — GAITEX records no sex,
  was fitted on neutral, and is drawn on the neutral model.
"@
  }
  Set-Content -Path (Join-Path $Dest "DISTRIBUTION_NOTICE.txt") -Value $notice -Encoding UTF8
  Write-Output "distribution notice written -> DISTRIBUTION_NOTICE.txt"

  # Check that the deployment actually imports. The refresh above only updates "files already in the bundle",
  # so when a file comes in that the repository changed to import a new module, that new module does not follow —
  # this is how the deployment broke silently once generate_prism_faithful.py started importing anthro_smpl.py.
  # A robocopy success does not catch that, so import it with the deployed Blender.
  $blender = Join-Path $Dest "blender\blender.exe"
  if (Test-Path $blender) {
    $expr = "import sys; sys.path.insert(0, r'$Dest\scripts'); import viewer_app; print('DEPLOY_IMPORT_OK')"
    $out = & $blender -b --python-expr $expr 2>&1 | Out-String
    if ($out -notmatch "DEPLOY_IMPORT_OK") {
      Write-Output $out
      throw "the deployment does not import — add the missing module to the bundle's scripts\ and deploy again."
    }
    Write-Output "deployment imports (viewer_app and every module it depends on are present)"
  } else {
    Write-Warning "no blender.exe in the deployment; skipping the import check: $blender"
  }
  Write-Output "Done. Target: $Dest"
} else {
  Write-Output "Preview finished. To deploy: add -Execute to the command above."
}

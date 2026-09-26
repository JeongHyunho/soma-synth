<#
  deploy_viewer.ps1 — SOMA Small/Large 뷰어 배포 스크립트 (-Dest 로 준 폴더로 복사).

  ★ run 데이터는 절대 뷰어에 넣지 않는다. 뷰어는 실행한 PC 의 번들을 읽는다:
    $SOMA_DATA_ROOT\runs\experimental_generation_poc_demo 아래의 번들, 또는 --folder <폴더>
    (SOMA_VIEWER_DATASET) 로 준 폴더. 번들은 만든 PC 에만 있다(ADR-0041).
  ★ 라이선스가 걸린 SMPL 모델(SMPL_MALE/FEMALE/NEUTRAL_clean.npz)은 기본 제외한다 (MPI 비상업·재배포금지 +
    ADR-0005 INTERNAL-ONLY). 모델은 데이터셋이 아니라 뷰어에 속하므로 포함 여부만 게이트한다.
    세 모델이 모두 필요하다: 피험자 betas 는 자기 모델의 shape space 에 있고, GAITEX 는 성별 기록이
    없어 neutral 모델로 피팅됐다 — 다른 모델 위에 올리면 그 피험자의 몸이 아니다.
  ★ 실제 복사는 -Execute 를 줄 때만 일어난다 (그 전에는 계획만 출력 = dry-run).
  ★ 모델을 포함하려면 -IncludeRestricted -IUnderstandLicense 를 둘 다 줘야 한다. 그때 번들 data\ 에
    없는 모델은 -ModelDir(기본 $SOMA_BODY_MODEL_DIR, 없으면 $SOMA_DATA_ROOT\body_models\smpl)에서
    먼저 채운다(번들이 셋 다 갖추도록).
  ★ -Dest 는 반드시 준다. 기본 배포처는 없다 — 공유 드라이브를 찾아 쓰지 않는다. 공유 드라이브로
    배포하는 것은 그 폴더를 -Dest 로 명시하는 별도의 결정이다.
  ★ -Source 는 반드시 준다. 휴대용 뷰어 번들(뷰어 스크립트, Blender, 발표 패키지 render_amass·
    gear_kit 의 사본)이다. 이 저장소는 그 번들을 만들지 않는다: 따로 받은 번들을 가리킨다.

  사용 예:
    # 1) 무엇이 복사될지 미리보기 (아무것도 안 옮김)
    pwsh scripts\poc\deploy_viewer.ps1 -Source <뷰어 번들> -Dest <배포 폴더>
    # 2) 코드+Blender+매뉴얼만 배포 (모델 제외 — 라이선스 안전)
    pwsh scripts\poc\deploy_viewer.ps1 -Source <뷰어 번들> -Dest <배포 폴더> -Execute
    # 3) SMPL 모델까지 포함 (수신자가 동일 SMPL 라이선스 보유자일 때만!)
    pwsh scripts\poc\deploy_viewer.ps1 -Source <뷰어 번들> -Dest <배포 폴더> -Execute `
      -IncludeRestricted -IUnderstandLicense
    # 4) 한시적: 예비 run 사본까지 동봉 (run 데이터를 옮기므로 번들 공유와 같은 승인이 필요하다)
    pwsh scripts\poc\deploy_viewer.ps1 -Source <뷰어 번들> -Dest <배포 폴더> -Execute `
      -IncludeRestricted -IUnderstandLicense -FallbackRun
#>
param(
  # 배포 원본(휴대용 뷰어 번들). 필수 — 기본값은 없다.
  [string]$Source = "",
  # 배포처. 필수 — 기본값은 없다(공유 드라이브를 스스로 찾지 않는다).
  [string]$Dest   = "",
  # 라이선스 SMPL 모델의 로컬 원본. 번들 data\ 에 빠진 모델을 여기서 채운다(-IncludeRestricted 일 때만).
  # 비우면 $SOMA_BODY_MODEL_DIR, 없으면 $SOMA_DATA_ROOT\body_models\smpl.
  [string]$ModelDir = "",
  # 리포지토리의 매뉴얼 폴더(이 스크립트가 든 체크아웃의 docs\manual).
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

# 배포처는 명시해야 한다. 기본 배포처는 없고, 공유 드라이브에 둘지는 -Dest 로 그 폴더를 적는 사람이 정한다.
if (-not $Dest) { throw "pass -Dest <folder to deploy the viewer into>; there is no default destination" }

if (-not $ModelDir) {
  if ($env:SOMA_BODY_MODEL_DIR) { $ModelDir = $env:SOMA_BODY_MODEL_DIR }
  elseif ($env:SOMA_DATA_ROOT) { $ModelDir = Join-Path $env:SOMA_DATA_ROOT "body_models\smpl" }
}

# 제한(라이선스) 자산: 기본 제외. 포함하려면 두 스위치를 모두 요구.
$restricted = $false
if ($IncludeRestricted) {
  if (-not $IUnderstandLicense) {
    Write-Warning "라이선스 SMPL 모델을 포함하려면 -IUnderstandLicense 도 함께 주세요."
    Write-Warning "그 모델은 재배포 금지 라이선스 대상입니다. 수신자가 동일 SMPL 라이선스 보유자임을 확인한 경우에만."
    throw "aborted: -IncludeRestricted without -IUnderstandLicense"
  }
  $restricted = $true
}

# run 데이터는 기본 제외 — 뷰어는 실행한 PC 의 번들($SOMA_DATA_ROOT 아래, 또는 --folder)을 읽는다.
# -FallbackRun 은 run 사본을 동봉하는 한시적 예외다.
# 라이선스 SMPL 모델 3종. 피험자 betas 는 자기 모델의 shape space 에 있으므로 셋 다 필요하다 —
# 여성 피험자를 남성 모델로, neutral 로 피팅된 GAITEX 피험자를 남성 모델로 그리면 근사가 아니라
# 다른 사람의 몸이 된다.
$modelFiles = @("SMPL_MALE_clean.npz", "SMPL_FEMALE_clean.npz", "SMPL_NEUTRAL_clean.npz")
$exclDirs  = @()
$exclFiles = @()
if (-not $FallbackRun) { $exclDirs += "runs" }
if (-not $restricted) { $exclFiles = $modelFiles }
if ($FallbackRun -and -not $restricted) {
  throw "aborted: -FallbackRun 은 run 데이터를 배포하므로 -IncludeRestricted -IUnderstandLicense 가 필요합니다."
}

# 번들 스크립트를 리포지토리 최신본으로 먼저 갱신한다. 배포 원본은 따로 받은 번들이라, 이 단계가 없으면
# 리포에서 고친 코드가 배포에 빠진 채 옛 스크립트가 나갈 수 있다.
# 번들의 scripts\ 는 선별된 부분집합이므로, 이미 번들에 있는 파일만 이름으로 맞춰 갱신한다 — 단,
# 리포의 뷰어가 새로 임포트하게 된 모듈($requiredScripts)은 번들에 없으면 추가한다.
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

# 번들 data\ 에 빠진 라이선스 모델을 로컬 원본에서 채운다(모델을 배포할 때만 의미가 있다).
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

Write-Output "==================== SOMA 뷰어 배포 ===================="
Write-Output " source : $Source"
Write-Output " dest   : $Dest"
Write-Output " manuals: $ManualDir\*.docx (KO + EN)"
Write-Output (" mode   : {0}" -f ($(if ($Execute) { "EXECUTE (실제 복사)" } else { "DRY-RUN (미리보기; -Execute 로 실행)" })))
Write-Output (" assets : {0}" -f ($(if ($restricted) { "★라이선스 SMPL 모델 포함 — 라이선스 확인됨" } else { "코드+Blender+매뉴얼만 (SMPL 모델 제외)" })))
Write-Output (" runs   : {0}" -f ($(if ($FallbackRun) { "★한시적 포함(-FallbackRun) — run 데이터를 옮기므로 번들 공유와 같은 승인이 필요" } else { "제외 — 뷰어는 실행한 PC 의 번들(SOMA_DATA_ROOT, --folder)을 읽습니다" })))
if ($exclFiles) { Write-Output ("  제외 파일 : {0}" -f ($exclFiles -join ", ")) }
if ($exclDirs)  { Write-Output ("  제외 폴더 : {0}" -f ($exclDirs -join ", ")) }
Write-Output (" scripts: {0}" -f ($(if ($refreshed) { ("리포 최신본으로 갱신 -> " + ($refreshed -join ", ")) } else { "번들이 이미 리포와 동일" })))
if ($restricted) {
  Write-Output (" models : {0}" -f ($(if ($modelsAdded) { ("번들에 추가 <- $ModelDir : " + ($modelsAdded -join ", ")) } else { "번들 data\ 에 3종 모두 있음" })))
  if ($modelsMissing) { Write-Warning ("번들에도 로컬에도 없는 모델: " + ($modelsMissing -join ", ") + " — 그 성별 피험자는 남아 있는 모델로 대체 표시됩니다 (고지문에 기록)") }
}
Write-Output "========================================================"

# robocopy 인자 구성 (/L = 목록만 = dry-run)
$rcArgs = @($Source, $Dest, "/E", "/NFL", "/NDL", "/NJH", "/NP", "/R:1", "/W:1")
foreach ($d in $exclDirs)  { $rcArgs += @("/XD", $d) }    # name-based: excludes that dir at any depth
foreach ($f in $exclFiles) { $rcArgs += @("/XF", $f) }
if (-not $Execute) { $rcArgs += "/L" }

if (-not $Execute) {
  Write-Output "[dry-run] robocopy 계획만 출력합니다. 실제 복사하려면 -Execute 를 추가하세요."
}

robocopy @rcArgs | Out-Null
$code = $LASTEXITCODE
Write-Output ("robocopy exit={0} (0-7=정상)" -f $code)

if ($Execute) {
  # 매뉴얼 docx + 배포 고지문 동봉 (reference.docx 는 Quarto 스타일 템플릿이라 제외)
  $docx = Get-ChildItem (Join-Path $ManualDir "*.docx") -ErrorAction SilentlyContinue |
          Where-Object { $_.Name -ne "reference.docx" }
  if ($docx) {
    New-Item -ItemType Directory -Force -Path (Join-Path $Dest "manual") | Out-Null
    foreach ($m in $docx) {
      Copy-Item $m.FullName (Join-Path $Dest ("manual\" + $m.Name)) -Force
      Write-Output ("매뉴얼 복사됨 -> manual\" + $m.Name)
    }
  } else {
    Write-Warning "매뉴얼 docx 없음: $ManualDir (먼저 build_manual.ps1 로 빌드하세요)"
  }
  $notice = @"
SOMA Small/Large 뷰어 — 배포 고지 (INTERNAL ONLY)

- 이 뷰어의 인체는 MPI SMPL body model(비상업 연구용, 재배포 금지) 파생물입니다.
- 동일 SMPL 라이선스를 보유한 연구실 내부 사용자만 사용/보관하십시오.
- 공개 배포·외부 공유 금지 (ADR-0005 INTERNAL-ONLY).

데이터 위치
- 뷰어 안에는 데이터가 없습니다. 뷰어는 이 PC 의 번들을 읽습니다:
  SOMA_DATA_ROOT 환경 변수가 가리키는 폴더의 runs\experimental_generation_poc_demo\<번들>\.
- 다른 폴더를 보려면 SOMA_VIEWER_DATASET 에 그 폴더를 넣거나 Blender 인자로
  -- --folder <폴더> 를 줍니다. 번들은 만든 PC 에 둡니다(ADR-0041).
"@
  if ($FallbackRun) {
    $notice += @"

한시적 안내
- 볼 번들이 없는 PC 를 위해 data\runs\ 에 예비 run 사본을 함께 넣어 두었습니다. 이 PC 에 번들이
  생기면 뷰어는 그것을 먼저 보여 주며, 예비 사본은 삭제해도 됩니다.
"@
  }
  if (-not $restricted) {
    $notice += @"

- 이 배포본에는 라이선스 SMPL 모델이 포함되어 있지 않습니다.
- 실행하려면 각자 data\SMPL_MALE_clean.npz 를 추가하세요.
- data\SMPL_FEMALE_clean.npz 까지 넣으면 여성 피험자를 여성 체형으로, data\SMPL_NEUTRAL_clean.npz
  까지 넣으면 neutral 로 피팅된 피험자(GAITEX 전체)를 그 체형으로 봅니다. 없으면 뷰어가 남성
  모델로 대체하고 그 사실을 콘솔에 알립니다 (그때 체형은 그 피험자의 것이 아닙니다).
"@
  } elseif ($modelsMissing) {
    $notice += @"

- data\ 에 SMPL 모델이 일부만 들어 있습니다 (없음: $($modelsMissing -join ', ')). 뷰어는
  anthro_reference 의 gender 로 모델을 고르며, 없는 성별의 피험자는 남아 있는 모델로 대체 표시하고
  콘솔에 그 사실을 알립니다 (그때 체형은 그 피험자의 것이 아닙니다).
"@
  } else {
    $notice += @"

- data\ 에 SMPL 모델 3종(MALE / FEMALE / NEUTRAL)이 들어 있습니다. 피험자 betas 는 자기 모델의
  shape space 에 있으므로 뷰어가 anthro_reference 의 gender 로 모델을 고릅니다 — GAITEX 는 성별
  기록이 없어 neutral 로 피팅됐고, neutral 모델로 그려집니다.
"@
  }
  Set-Content -Path (Join-Path $Dest "DISTRIBUTION_NOTICE.txt") -Value $notice -Encoding UTF8
  Write-Output "배포 고지문 작성됨 -> DISTRIBUTION_NOTICE.txt"

  # 배포본이 실제로 임포트되는지 확인한다. 위 갱신 단계는 "번들에 이미 있는 파일"만 최신화하므로,
  # 리포에서 새 모듈을 임포트하게 바뀐 파일이 들어오면 그 새 모듈은 따라오지 않는다 — 실제로
  # generate_prism_faithful.py 가 anthro_smpl.py 를 임포트하게 되면서 배포본이 조용히 깨졌다.
  # robocopy 성공은 그것을 잡지 못하므로, 배포된 Blender 로 직접 임포트해 본다.
  $blender = Join-Path $Dest "blender\blender.exe"
  if (Test-Path $blender) {
    $expr = "import sys; sys.path.insert(0, r'$Dest\scripts'); import viewer_app; print('DEPLOY_IMPORT_OK')"
    $out = & $blender -b --python-expr $expr 2>&1 | Out-String
    if ($out -notmatch "DEPLOY_IMPORT_OK") {
      Write-Output $out
      throw "배포본이 임포트되지 않습니다 — 누락된 모듈을 번들 scripts\ 에 추가한 뒤 다시 배포하세요."
    }
    Write-Output "배포본 임포트 확인됨 (viewer_app 및 의존 모듈 전부 존재)"
  } else {
    Write-Warning "배포본에 blender.exe 가 없어 임포트 확인을 건너뜁니다: $blender"
  }
  Write-Output "완료. 대상: $Dest"
} else {
  Write-Output "미리보기 종료. 실제 배포: 위 명령에 -Execute 추가."
}

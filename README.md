# soma-synth

> **INTERNAL-ONLY.** 내부 팀원 전용 저장소다. 이 코드로 만든 번들은 모두
> `experimental_non_candidate` · `internal_only` 이며 **만든 PC 에만 둔다.** 그 PC 밖으로 공유하거나
> 외부에 공개하려면 별도 승인이 필요하다. 원천 데이터셋의 라이선스와 이용 조건은 그대로 적용된다.

SOMA 합성 IMU 데이터셋 생성 파이프라인이다(스펙 `qmd_unified8_smpl18`). 모션캡처 원천
(AMASS, PRISM, HKNU, GAITEX, AddBiomechanics)에서 가상 IMU 8채널(Small)과 SMPL 18관절
참조(Large) 번들을 만든다. 생성기, 리타깃 코퍼스, 검증(L0–L4), 필드별 README 렌더링, 로컬
카탈로그 등록, 실행 기록을 `soma-synth` 명령 하나로 묶는다.

이 문서는 팀원이 자기 PC 에서 처음부터 번들을 만들기까지의 안내서다. 순서대로 따라가면 된다.

1. [저장소 접근 권한](#1-저장소-접근-권한)
2. [설치](#2-설치) — Windows, macOS·Linux
3. [환경 변수 세 개](#3-환경-변수-세-개)
4. [원천 데이터 받기](#4-원천-데이터-받기) — `stage-sources`, 목록(`SHA256SUMS`)
5. [바디 모델](#5-바디-모델)
6. [첫 표본 실행](#6-첫-표본-실행) — 한 피험자, scratch 폴더, 확인 후 삭제
7. [전체 lineage 실행](#7-전체-lineage-실행)
8. [검증 L4 --full](#8-검증-l4---full)
9. [결과는 어디에 남나](#9-결과는-어디에-남나)
10. [무엇이 허용되나 (ADR-0041)](#10-무엇이-허용되나-adr-0041)
11. [보존 규칙](#11-보존-규칙)
12. [문제 해결](#12-문제-해결)

규격과 동작의 정본은 [`docs/guides/GENERATION_PIPELINE_STANDARD.md`](docs/guides/GENERATION_PIPELINE_STANDARD.md),
번들별 한계 선언은 [`docs/guides/KNOWN_LIMITATIONS.md`](docs/guides/KNOWN_LIMITATIONS.md), 뷰어는
[`docs/guides/small_large_viewer.md`](docs/guides/small_large_viewer.md) 다. 의존 방향은 상위 프로젝트 →
soma-synth → smpl18 이고, soma-synth 는 상위 프로젝트를 import 하지 않는다.

---

## 1. 저장소 접근 권한

비공개 저장소 **두 개**에 읽기 권한이 있어야 한다. 소유자에게 GitHub 계정을 알려 요청한다.

| 저장소 | 무엇 |
|---|---|
| `JeongHyunho/soma-synth` | 이 저장소 |
| `JeongHyunho/smpl18` | SMPL 변환 규칙(성별별 모델 선택, 골격, 18관절 축소). 이 저장소의 `packages/smpl18` submodule |

`smpl18` 권한이 없으면 `git clone --recurse-submodules` 가 submodule 단계에서
`repository not found` 로 실패한다. HTTPS 로 받을 때는 GitHub 로그인(또는 개인 토큰)이, SSH 로 받을
때는 등록된 키가 필요하다. 토큰이나 키를 저장소 파일에 적지 않는다.

## 2. 설치

기준 환경은 **CPython 3.13.5** 이다(3.12 이상이면 돌지만, 산출의 바이트 동일성은 같은 PC·같은 고정
환경에서만 주장한다). 라이브러리 버전은 [`constraints.txt`](constraints.txt) 가 고정한다.

### Windows (PowerShell)

```powershell
git clone --recurse-submodules https://github.com/JeongHyunho/soma-synth.git
cd soma-synth
# 이미 clone 했는데 packages\smpl18 이 비어 있으면: git submodule update --init
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -c constraints.txt -e packages/smpl18 -e ".[dev]"
.venv\Scripts\soma-synth --help
```

### macOS · Linux (bash / zsh)

```bash
git clone --recurse-submodules https://github.com/JeongHyunho/soma-synth.git
cd soma-synth
python3.13 -m venv .venv            # 3.13 이 없으면 pyenv, Homebrew(python@3.13), 배포판 패키지로 설치
.venv/bin/python -m pip install -c constraints.txt -e packages/smpl18 -e ".[dev]"
.venv/bin/soma-synth --help
```

- 아래 예시는 venv 를 켠 상태(`.venv\Scripts\Activate.ps1` / `source .venv/bin/activate`)에서
  `soma-synth ...` 로 적는다. 켜지 않았다면 `.venv\Scripts\soma-synth` / `.venv/bin/soma-synth` 로 부른다.
- 테스트: `python -m pytest -q` (저장소 루트에서). 뷰어의 그림 테스트까지 돌리려면 `-e ".[dev,viewer]"`
  로 설치한다.
- 생성기의 프로세스 풀은 모든 플랫폼에서 `spawn` 으로 시작한다(Linux 의 기본 `fork` 를 쓰지 않는다).
- 경로 규칙은 운영체제와 상관없이 같다. 기록되는 경로(논리 id `extracted/<폴더>/<경로>`, data root
  기준 상대 경로)는 언제나 `/` 로 적힌다.

## 3. 환경 변수 세 개

| 변수 | 뜻 | 기본값 |
|---|---|---|
| `SOMA_DATA_ROOT` | 이 PC 의 데이터 루트(출력). 번들·코퍼스·실행 기록·카탈로그가 여기 쌓인다 | **필수, 기본값 없음.** 절대 경로, 이미 있는 폴더 |
| `SOMA_SOURCE_ROOT` | 원천 폴더들(`amass`, `prism`, `gaitex`, `hknu_fullbody`, `addbiomechanics`)을 바로 담은 폴더 | `<SOMA_DATA_ROOT>/extracted` |
| `SOMA_BODY_MODEL_DIR` | `SMPL_{MALE,FEMALE,NEUTRAL}_clean.npz` 가 있는 폴더 | `<SOMA_DATA_ROOT>/body_models/smpl` |

```powershell
# Windows: 이 창에서만
$env:SOMA_DATA_ROOT = "E:\soma_data"
New-Item -ItemType Directory -Force $env:SOMA_DATA_ROOT | Out-Null    # 없는 폴더는 거부된다
# 계속 쓰려면: setx SOMA_DATA_ROOT "E:\soma_data"  (새 창부터 적용)
```

```bash
# macOS / Linux: ~/.zshrc 나 ~/.bashrc 에 넣어 두면 계속 쓴다
export SOMA_DATA_ROOT="$HOME/soma_data"
mkdir -p "$SOMA_DATA_ROOT"
```

**데이터 루트는 로컬 디스크에 둔다.** 동기화 폴더 안에는 쓰지 않는다: Dropbox, OneDrive, Google Drive
(`My Drive`·`내 드라이브`·`Shared drives`·`공유 드라이브`, `/Volumes/GoogleDrive*`), Synology Drive,
iCloud Drive, macOS 의 `~/Library/CloudStorage/*`, Linux 의 gvfs `google-drive:` 마운트, 그리고
`SOMA_SHARED_DRIVE_NAMES` 에 적은 팀 공유 드라이브(폴더 이름을 쉼표로 나눠 적는다. 대소문자는 가리지
않는다. 기본값은 비어 있다). 동기화 클라이언트가 생성 중인 파일을 건드리고, 내부 전용 데이터를 PC 밖으로
옮기기 때문이다. 원천(`SOMA_SOURCE_ROOT`)과 바디 모델은 읽기만 한다.

```powershell
$env:SOMA_SHARED_DRIVE_NAMES = "<공유 드라이브 폴더 이름>"      # 예: 팀 공유 드라이브의 최상위 폴더 이름
```

데이터 루트는 이렇게 채워진다.

```
<SOMA_DATA_ROOT>/
├─ extracted/<원천 폴더>/            원천 (SOMA_SOURCE_ROOT 기본값; 읽기만)
├─ body_models/smpl/                 바디 모델 (SOMA_BODY_MODEL_DIR 기본값; 읽기만)
├─ runs/experimental_generation_poc_demo/
│   ├─ <lineage>/                    번들·코퍼스 (예: hknu_unified8, hknu_smpl24_paired)
│   ├─ _runs/run_<identity>/         실행 기록
│   └─ _superseded/                  교체한 세대의 증거
├─ experimental/catalog/             로컬 카탈로그
└─ tmp/                              표본 실행용 scratch (관례)
```

## 4. 원천 데이터 받기

원천 다섯 폴더는 소유자가 알려 주는 폴더(원천 다섯 폴더를 담은 공유 사본)에서 받는다. 소유자는 그와
함께 목록 파일 `SHA256SUMS` 를 준다. 목록은 GNU coreutils 형식(`<sha256>  <폴더>/<상대 경로>`)이라
`sha256sum -c` 로도 확인할 수 있다.

```powershell
# 원천 폴더들을 담은 폴더에서, 목록에 있고 이 PC 에 없는 파일만 복사한다
soma-synth stage-sources --from "<원천 폴더들을 담은 폴더>" --sums C:\lists\SHA256SUMS --sources prism,hknu
# 받은 뒤 대조만 (복사하지 않는다)
soma-synth stage-sources --sums C:\lists\SHA256SUMS --sources prism,hknu --verify-only
```

- `--to` 를 주지 않으면 `SOMA_SOURCE_ROOT`, 없으면 `<SOMA_DATA_ROOT>/extracted` 를 채운다.
- 파일마다 목적지 폴더의 임시 이름에 복사하고, 목록의 SHA-256 과 맞을 때만 제 이름으로 바꾼다. 이미
  있고 해시가 맞는 파일은 건너뛰므로 끊겼다가 다시 돌려도 된다.
- **아무것도 지우거나 덮어쓰지 않는다.** 이미 있는 파일의 해시가 목록과 다르면 전부 보고하고 아무것도
  복사하지 않은 채 종료 코드 3 으로 끝난다. 어느 쪽이 맞는지 확인한 뒤 손으로 정리한다.
- 드라이브가 아직 동기화 중이면 `--from` 의 파일이 목록과 다를 수 있다. 그 파일은 쓰지 않고 보고하며
  종료 코드 1 이다. 동기화가 끝난 뒤 다시 돌린다.
- `--sources` 는 폴더 이름(`hknu` 는 `hknu_fullbody`)이다. 필요한 원천만 받아도 된다.
- 반입이 중간에 끊기면 검증 전의 복사본이 `.<이름>.<무작위>.staging` 으로 남을 수 있다. 원천 파일이
  아니므로 목록과 원천 추적에서 빠지고, `stage-sources`(`--verify-only` 포함)가 `LEFTOVER` 로 알려 준다.
  이 명령은 지우지 않으니 손으로 지운다.

원천 크기(참고; GB 는 10^9 바이트, `hash-sources` 가 적는 단위): `amass` 12.1 GB, `prism` 5.0 GB,
`gaitex` 17.7 GB, `hknu_fullbody` 27.9 GB, `addbiomechanics` 567.6 GB(528.6 GiB). 원천마다 라이선스·이용 조건이 따로 있다. 받기 전에 확인하고 따른다.

목록을 만드는 쪽(소유자)은 원천을 읽기만 하는 `hash-sources` 를 쓴다. 목록은 원천 폴더나
`extracted/`·`raw_archives/`, 증거 폴더(`_superseded`·`_runs`·`_manifest_backfill`), lineage 컨테이너
(`runs/experimental_generation_poc_demo`), 번들 밖에 쓴다. 이미 있는 파일이 이 명령이 쓴 목록이 아니면
덮지 않는다(`--force` 로만).

```powershell
soma-synth hash-sources --out C:\lists\SHA256SUMS                  # 다섯 폴더 전부
soma-synth hash-sources --sources hknu,prism --out C:\lists\hknu_prism.SHA256SUMS
```

## 5. 바디 모델

`SOMA_BODY_MODEL_DIR` 에 `SMPL_MALE_clean.npz`, `SMPL_FEMALE_clean.npz`, `SMPL_NEUTRAL_clean.npz` 세 개를
둔다. 생성기는 피험자 성별로 하나를 고르므로 셋 다 있어야 한다(GAITEX 는 성별 기록이 없어 neutral).

- 소유자가 알려 주는 공유 사본의 `body_models/smpl` 에서 받는다.
- 또는 MPI SMPL 라이선스를 받아 원본 `.pkl` 로 직접 만든다(chumpy 없이 읽는 변환기, pkl 은 읽기만 한다):

  ```powershell
  python scripts/poc/prepare_smpl_clean_npz.py --src <SMPL .pkl> --out <모델 폴더>\SMPL_FEMALE_clean.npz
  ```

SMPL 모델은 MPI 의 비상업 연구 라이선스 대상이다. 같은 라이선스를 가진 사람에게만 옮긴다.

## 6. 첫 표본 실행

한 피험자만 scratch 폴더에 만들어 설치와 경로가 맞는지 본다. `--stop-after validate` 로 README 와 카탈로그
등록은 건너뛴다. 실행 기록도 scratch 폴더(`--runs-root`)에 남겨 운영 기록 폴더(`_runs`)에는 아무것도 쌓이지
않게 하고, 확인이 끝나면 scratch 폴더를 통째로 지운다.

```powershell
$S = "$env:SOMA_DATA_ROOT\tmp\first_sample"
$R = "$S\_runs"
# PRISM(원천 5.0 GB, 코퍼스 없음) 한 피험자
soma-synth run --source prism "$S\prism_faithful_full" --bundle-arg=--only --bundle-arg=prism_subj001 --stop-after validate --runs-root $R
# GAITEX(코퍼스가 있는 소스, 원천 17.7 GB) 한 피험자: 코퍼스도 scratch 에 만든다
soma-synth run --source gaitex "$S\gaitex_unified8" --corpus "$S\gaitex_smpl24" --corpus-arg=--subjects --corpus-arg=austra --bundle-arg=--subjects --bundle-arg=austra --stop-after validate --runs-root $R
```

```bash
S="$SOMA_DATA_ROOT/tmp/first_sample"
R="$S/_runs"
# HKNU(코퍼스가 있는 소스) 한 피험자: 코퍼스도 scratch 에 만든다
soma-synth run --source hknu "$S/hknu_unified8" --corpus "$S/hknu_smpl24_paired" \
    --corpus-arg=--subjects --corpus-arg=S01 --bundle-arg=--subjects --bundle-arg=S01 \
    --stop-after validate --runs-root "$R"
# GAITEX 한 피험자
soma-synth run --source gaitex "$S/gaitex_unified8" --corpus "$S/gaitex_smpl24" \
    --corpus-arg=--subjects --corpus-arg=austra --bundle-arg=--subjects --bundle-arg=austra \
    --stop-after validate --runs-root "$R"
```

- 번들 폴더 이름은 **lineage 이름 그대로** 둔다(`prism_faithful_full`, `hknu_unified8` 등). 러너가 그
  이름으로 `KNOWN_LIMITATIONS.json` 블록을 찾고, 검증이 그 파일을 요구한다.
- 선택 인자(`--only`, `--subjects`, `--trials` 등)는 `=` 형태로 넘긴다(`--bundle-arg=--only`). 운영
  lineage 폴더(`runs/experimental_generation_poc_demo/<lineage>`)로 향하는 선택 인자는 거부된다. 표본은
  언제나 scratch 폴더로 간다.
- 피험자를 고르는 값(대소문자 구별): PRISM 의 `--only` 는 원천 폴더 이름이 아니라 **피험자 id**
  `prism_subj001` 을 받는다(원천 폴더는 `subj001`; `subj001` 을 주면 "no PRISM take of that subject" 로
  거부된다). HKNU 와 GAITEX 의 `--subjects` 는 원천 폴더 이름을 그대로 받는다(`S01`, `austra`).
- 끝나면 요약에 단계별 결과(`ok`·`FAIL`)와 실행 기록 위치가 나온다. 검증 결과는 번들의
  `VALIDATION_REPORT.json` 에, 생성기가 출력한 것은 실행 기록의 `logs/` 에 있다(§9).

확인이 끝나면 scratch 폴더를 지운다. 번들·코퍼스와 함께 `$R` 의 실행 기록도 지워진다: 표본 실행은 운영
기록으로 남기지 않는다.

```powershell
Remove-Item -Recurse -Force "$env:SOMA_DATA_ROOT\tmp\first_sample"
```

```bash
rm -rf "$SOMA_DATA_ROOT/tmp/first_sample"
```

## 7. 전체 lineage 실행

폴더를 주지 않으면 번들과 코퍼스는 데이터 루트의 lineage 폴더로 간다.

```powershell
soma-synth run --source prism                   # prism_faithful_full
soma-synth run --source hknu                    # 코퍼스 hknu_smpl24_paired → 번들 hknu_unified8
soma-synth run --source gaitex                  # gaitex_smpl24 → gaitex_unified8
soma-synth run --source amass --jobs 8          # amass_faithful_full (8 샤드 + 병합)
soma-synth run --source addbiomechanics --jobs 8   # addbio_smpl24_raw → addbio_unified8 (프로세스 8개)
```

- `run` 은 generate → validate → readme → register 를 차례로 하고 실행 기록을 남긴다. 한 단계가 실패하면
  거기서 멈춘다(fail-fast).
- 코퍼스가 있는 소스(`hknu`, `gaitex`, `addbiomechanics`)는 코퍼스가 없으면 먼저 만들고, 끝난 코퍼스가
  있으면 다시 쓰기만 한다. 코퍼스를 다시 만들려면 `--rebuild-corpus --replace-existing`.
- **모든 생성은 빈 폴더에 새로 한다.** 무엇이든 들어 있는 번들 폴더는 `--replace-existing` 없이는
  거부된다. 그 옵션을 주면 폴더를 `<폴더>.replaced-<run>` 으로 옮기고 빈 폴더에 만든다. 끝난 세대가 든
  운영 폴더를 교체할 때는 그 증거를 먼저 `_superseded/` 에 남겨야 한다(보존 규칙 1.2, 표준 §1.3).
- `--jobs` 는 `amass`(N 샤드 뒤 병합)와 `addbiomechanics`(프로세스 풀)에만 뜻이 있다.
- AddBiomechanics 원천은 약 568 GB(529 GiB)다. 전체 코퍼스를 새로 만드는 실행은 여러 시간 걸린다. 원천
  파일 전체의 SHA-256 을 실행 기록에 남기는 일(원천 추적)도 코퍼스를 만드는 실행만 한다: 그 시간은 원천
  크기에 비례하고, 코퍼스를 재사용하는 실행은 원천을 해시하지 않는다(표준 §4.1).

## 8. 검증 L4 --full

`run` 의 validate 단계는 L4 검증을 하되, 이전 PASS 가 원장(ledger)에 있는 take 는 다시 보지 않는다.
전부 다시 보려면 `--full` 을 준다. 검증기 코드가 바뀌면 원장의 PASS 가 한 번 무효가 되어 다음 검증이 모든
take 를 다시 본다(표준 §7).
`run` 의 보고서는 같은 실행이 검증 뒤에 쓰는 파일(보고서·원장, readme 단계가 있고 검증이 통과하면 `README.md`)을
없다고 경고하지 않는다. 검증이 실패하면 실행이 readme 단계 전에 멈추므로 `README.md` WARN 은 남는다. 기록 없이 `validate` 만 하면 번들에 없는 것은 그대로 WARN 이다.

```powershell
# 이미 만든 번들을 기록과 함께 다시 검증 (generate 는 하지 않는다)
soma-synth run --source hknu --start-at validate --stop-after validate --full
# 기록 없이 검증만
soma-synth validate $env:SOMA_DATA_ROOT\runs\experimental_generation_poc_demo\hknu_unified8 --level L4 --full --report C:\reports\hknu_unified8.json
```

`validate`·`readme`·`register`·`pipeline` 은 생성이 끝나지 않은 번들(`.generating` 이 남은 폴더)과
옆으로 옮긴 폴더(`*.replaced-*`, `*.failed-*`)를 거부한다(종료 코드 3). 증거 폴더(`_superseded`·`_runs`·
`_manifest_backfill`, 어느 데이터 루트의 것이든) 안의 폴더와 `extracted`·`raw_archives` 안의 폴더도 `run` 처럼
거부한다: `readme` 는 언제나, `validate` 는 보고서·원장을 그 번들에 쓸 때(`--report`·`--ledger` 없이 검증만
하면 읽기만 하므로 거부하지 않는다), `register` 는 보관된 증거를 살아 있는 자산으로 올리지 않도록. 번들 밖에
쓰는 `--report`·`--ledger` 와 `register` 의 `--catalog` 는 `run` 의 실행 기록·카탈로그와 같은 검사를 받는다.

## 9. 결과는 어디에 남나

| 무엇 | 어디 |
|---|---|
| 번들 | `<SOMA_DATA_ROOT>/runs/experimental_generation_poc_demo/<lineage>/` (`INDEX.json`, take 폴더, `KNOWN_LIMITATIONS.json`, `VALIDATION_REPORT.json`, `validation_ledger.json`, `README.md`) |
| 코퍼스 | 같은 컨테이너의 코퍼스 lineage (`hknu_smpl24_paired`, `gaitex_smpl24`, `addbio_smpl24_raw`) |
| 실행 기록 | `<번들의 부모>/_runs/run_<identity>/` (`--runs-root` 로 바꿀 수 있다): `resolved_config.yaml`, `manifest.json`, `environment.lock`, `source_assets.json`, `metrics.json`, `exclusions.json`, `logs/`, `source_manifest.json` |
| 실행 로그 | 실행 기록의 `logs/`: `runner.log`(러너가 출력한 것), 자식 단계마다 표준 출력·오류 전부 — `corpus.log`, `generate.log`(샤드마다 `generate.shard<i>of<N>.log`), `post_step.<스크립트>.log`. 콘솔에 나오는 것과 같고, 명령 줄로 시작해 종료 코드로 끝난다. 거부면 `refusal.txt`, 예기치 않은 실패면 `error.txt` |
| 원천 추적 | 실행 기록의 `source_manifest.json` — take manifest 가 적은 원천 참조와, 코퍼스가 있는 소스는 이번 실행의 단계가 읽을 수 있었던 원천 파일 전부의 SHA-256(`source_files`) |
| 카탈로그 | `<SOMA_DATA_ROOT>/experimental/catalog/` (`asset_catalog.json`, `<source>/source_manifest.json`). 쓸 때마다 바뀐 항목(옛 값·새 값)과 쓴 실행의 id 가 `_history/<파일>.journal/` 에 작은 기록으로 남고(전체 사본은 없다), 쓰는 동안 `.catalog.lock` 을 잡는다. 같은 번들을 다시 등록하면 그 번들의 이전 항목 가운데 지금 파일과 다른 것은 `superseded`(`registration_replaced`)로, 번들에 더는 없는 자산(`ok` 가 아니게 된 take, 없어진 파일)의 항목은 `superseded`(`registration_dropped`)로 표시되고 바뀐 자산만 더해진다. 바뀐 것이 없으면 쓰지 않는다(표준 §4.1). 자산 경로는 데이터 루트 기준이다: `register`·`pipeline` 은 `--data-root`, 없으면 `SOMA_DATA_ROOT` 를 쓰고, 둘 다 없거나 번들이 그 안에 없으면 종료 코드 2 로 거부한다 |

실행 기록에는 코드와 `smpl18` 리비전, 생성 근거(상시 허용 결정), 바디 모델 세트의 해시가 들어 있다.
실행 기록은 지우지 않는다. 원천의 라이선스가 철회되면 실행 기록을 모아 영향받은 번들을 찾는다.

## 10. 무엇이 허용되나 (ADR-0041)

- 소유자와 내부 팀원은 **자기 PC 에서** `experimental_non_candidate` · `internal_only` 번들을 만들 수 있다.
  실행마다 승인 기록을 받을 필요는 없다(상시 허용 결정). 러너는 이 등급만 만들고, 다른 등급은 거부한다.
- **번들은 만든 PC 에 둔다.** 공유 드라이브·클라우드·다른 PC 로 옮기거나 외부에 공개하려면 소유자의
  별도 승인이 필요하다. `soma-synth` 에는 push 명령이 없다.
- 원천 데이터셋의 라이선스와 이용 조건은 그대로 적용된다. SMPL 모델도 마찬가지다.
- 소유자 기록(`--authorized-by`)은 이 저장소가 상위 프로젝트에 `packages/soma-synth` 로 마운트됐을 때만
  읽힌다. 팀원의 실행에는 필요 없다.
- canonical·candidate 등급의 생성은 이 결정으로 열리지 않는다.

## 11. 보존 규칙

[`docs/guides/RETENTION_RULES.md`](docs/guides/RETENTION_RULES.md) 는 상위 프로젝트 데이터 보존 규칙
1절(1.1–1.7)을 같은 번호와 뜻으로 옮긴 것이다(상위 프로젝트 규칙이 권위). 러너와 경로 검사의 메시지가 적는
"retention rule 1.x" 가 이 번호다.
요점:

- `extracted/`·`raw_archives/` 의 원천은 지우지도, 고쳐 쓰지도 않는다. `stage-sources` 는 없는 파일을
  더하기만 한다.
- `_superseded/`·`_runs/`·`_manifest_backfill/` 은 증거와 실행 기록이다. 지우지 않는다.
- 데이터 루트의 `README.md`, 그리고 있으면 `MASTER.md`·`state/local_archive_inventory.json` 같은 데이터
  루트 기록 파일은 덮어쓰지 않는다.
- 운영 폴더의 payload 삭제는 소유자 승인 사항이다(1.5). 러너도 교체 후 운영 폴더의 옛 세대는 지우지 않고
  `<폴더>.replaced-<run>` 으로 남긴다.
- 카탈로그의 `_history/` 는 변경 기록이다. 이 도구는 고치지도 지우지도 않고, 이전 항목은 현재 파일과
  기록으로 복원된다(`catalog.catalog_as_of`). 기록 하나는 그 쓰기가 바꾼 항목만 담으므로, 크기는 바뀐 항목
  수를 따른다(카탈로그 크기와는 상관없다). `run`·`register`·`pipeline` 이 번들을 처음 등록하면 기록 하나에
  번들의 자산이 모두 들어간다: gzip 으로 자산 하나에 약 55 바이트, `gaitex`(360 자산) 약 20 KB, `prism`(900)
  약 50 KB, `hknu`(1,400) 약 80 KB, `amass`(41,305) 약 2.4 MB, `addbiomechanics`(202,300) 약 11 MB 다. 한 버전 전체를 대체(supersede)하는 쓰기는 옛 항목과 새 항목을
  함께 담아 그 두 배쯤이다. 다시 만든 번들을 다시 등록하면 바뀐 자산마다 표시(옛 항목과 새 항목)와 추가가 들어가
  그 세 배쯤이고, 바뀐 것이 없으면 쓰지 않으므로 기록도 없다. 자산 몇 개만 바꾼 쓰기는 수 KB 다.

## 12. 문제 해결

**종료 코드**

| 코드 | `run` | `stage-sources` |
|---|---|---|
| 0 | 모든 단계 성공 | 목록의 파일이 모두 있고 맞다 |
| 1 | 한 단계가 실패(검증 FAIL 포함) | 받지 못한 파일이 있다(`--from` 에 없음, 복사본 해시 불일치) 또는 `--verify-only` 에서 없거나 다른 파일 |
| 2 | 위치나 레지스트리를 풀 수 없다(`SOMA_DATA_ROOT` 없음·상대 경로·없는 폴더, 모르는 source) | 목록·폴더·인자를 쓸 수 없다 |
| 3 | **거부** — 사유가 표준 오류에 나온다 | 거부 — 목적지 파일이 목록과 다르다(아무것도 복사하지 않음), 또는 목적지 위치 |

**자주 보는 거부(종료 코드 3)**

| 메시지의 요지 | 풀어 주는 것 |
|---|---|
| `cloud-synchronised folder` / `team shared drive` | 데이터 루트·출력을 로컬 디스크로 |
| `inside ... which is read-only` (`extracted`, `raw_archives`, 소스 폴더, 바디 모델, `_superseded`, `_manifest_backfill`) | 출력을 lineage 폴더나 scratch 로 |
| `it already holds ...` (번들·코퍼스 폴더가 비어 있지 않다) | `--replace-existing`(코퍼스는 `--rebuild-corpus` 도), 또는 빈 폴더 |
| `selects part of the source ... production lineage directory` | 표본은 scratch 폴더로(§6) |
| `sets --out, which the runner controls` | 출력은 `dataset_dir`, 코퍼스는 `--corpus`, 원천은 `SOMA_SOURCE_ROOT` 로 |
| `no _superseded/<이름>_* folder ... holds a copy` | 운영 폴더를 교체하기 전에 증거를 `_superseded/` 에(보존 규칙 1.2) |
| `its generation did not finish -- .generating is there` | 그 번들은 끝나지 않았다. `--replace-existing` 으로 다시 만든다 |

**잠금** — generate 단계는 번들(과 코퍼스) 폴더 옆에 `<폴더>.lock` 을 만들고 끝나면 지운다. 다른 실행이
돌고 있으면 거부된다. 메시지가 그 잠금의 pid·호스트·시작 시각을 적는다. 그 프로세스가 없으면
(`tasklist /FI "PID eq <pid>"` 또는 `ps -p <pid>`) 죽은 실행이 남긴 것이니 손으로 지운다. 러너는 남의 잠금을
지우지 않는다. 카탈로그는 `.catalog.lock` 에 운영체제 잠금을 잡는다. 다른 등록이 잡고 있으면 최대 900 초
기다린 뒤 `CatalogLocked` 로 멈춘다. `.catalog.lock` 파일은 누구의 것이든 지우지 않는다(있는 것이 정상이다).

**`<폴더>.replaced-<run>` / `<폴더>.failed-<run>`** — `--replace-existing` 이 옮겨 둔 이전 세대와, 교체에
실패한 실행의 산출이다. 성공하면 scratch 폴더의 이전 세대는 지워지고, 운영 폴더의 것은 소유자 승인 전까지
남는다. 실패하면 새 산출은 `.failed-<run>` 이 되고 이전 세대가 제자리로 돌아온다. 둘 중 하나가 옆에 있는
동안 그 폴더는 다시 교체되지 않는다: 새 세대를 받아들이면 `.replaced-*` 를 지우고(운영 폴더는 승인 후),
되돌리려면 지금 폴더를 치우고 `.replaced-*` 의 이름을 되돌린다. `.failed-*` 는 살펴본 뒤 지운다.

**그 밖**

- `import smpl18` 실패: submodule 이 비어 있다. `git submodule update --init` 후 다시 설치한다.
- `SOMA_DATA_ROOT is not set` / `is not an absolute path` / `does not exist`: §3 대로 설정한다. 이 변수는
  기본값이 없다.
- 셸의 `python` 이 venv 가 아닌 인터프리터일 수 있다. venv 의 파이썬을 경로로 부르거나 venv 를 켠다.
- 명령과 옵션: `soma-synth --help`, `soma-synth <명령> --help`
  (`run`, `validate`, `readme`, `register`, `status`, `pipeline`, `pipeline-doc`, `provenance`,
  `hash-sources`, `stage-sources`).

## 뷰어와 도구

`scripts/poc` 에는 생성기 말고도 Small/Large 뷰어(`viewer_*.py`, `smpl_rig.py`, `smpl_model.py`,
`imu_plot.py`, `deploy_viewer.ps1`)와 그림·영상·보정 도구가 있다. 이 도구들은 생성 경로가 아니다. 뷰어는
이 PC 의 번들(`SOMA_DATA_ROOT` 아래)이나 `--folder` 로 준 폴더를 읽는다. 파이썬 의존성은 `-e ".[viewer]"`,
Blender 4.5 는 따로 설치해 `BLENDER_EXE` 로 가리킨다. 뷰어와 장면 빌더는 발표 패키지(`render_amass`·
`gear_kit`)가 **있어야** 돈다. 이 패키지는 이 저장소에 없고 휴대용 뷰어 번들과 함께 따로 배포된다: 받은 패키지
폴더를 `SOMA_ANIM_DIR` 로 가리키거나 뷰어 스크립트 옆에 둔다. 이 PC 에 내려받지 않은 온라인 전용
파일(Dropbox·OneDrive 의 자리표시자)은 열지 않고 건너뛴다. 자세한 것은
[`docs/guides/small_large_viewer.md`](docs/guides/small_large_viewer.md), 사용 설명서의 원본은
`docs/manual` 에 있다.

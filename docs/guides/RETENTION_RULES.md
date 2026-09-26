# 합성 데이터셋 보존 규칙

> 이 문서는 soma-synth 가 따르는 데이터 보존 규칙을 이 저장소의 말로 적은 것이다. 권위는 상위
> 프로젝트(SOMA Synthetic IMU)의 데이터 보존 규칙에 있다.
> 이 문서와 그 규칙이 어긋나면 **상위 프로젝트의 규칙이 이긴다.** 번호(1.1–1.7)와 각 규칙의 뜻은
> 상위 프로젝트 규칙 1절과 같다.
>
> 이 저장소의 메시지·주석·문서에서 "retention rule 1.x"(보존 규칙 1.x)는 아래 번호의 규칙을
> 뜻한다. 예를 들어 러너의 거부 메시지가 적는 "retention rule 1.5" 는 1.5(삭제는 소유자 승인
> 사항)이다.
>
> 데이터 경로는 데이터 루트(`<SOMA_DATA_ROOT>`) 기준이다.

---

## 1. 합성 데이터셋 보존 규칙 (2026-09-07 제정, 2026-09-14 개정)

### 1.1 lineage 하나에 디렉터리 하나, 이름에 버전은 없다

**lineage** = `(source, stage, spec_id)` 세 값이 같은 산출물의 계열.
디렉터리 이름은 `<source>_<stage>`이며 **버전 접미사를 붙이지 않는다** (2026-09-14 개정).

| lineage 디렉터리 | stage |
|---|---|
| `hknu_smpl24_paired` | HKNU 리타깃 코퍼스 (fit 수정 반영) |
| `hknu_unified8` | HKNU unified8 번들 |
| `prism_faithful_full` | PRISM 측정 small 번들 |
| `amass_faithful_full` | AMASS unified8 번들 |
| `gaitex_smpl24` / `gaitex_unified8` | GAITEX 리타깃 코퍼스 / 번들 |
| `addbio_smpl24_raw` / `addbio_smpl24_synth` / `addbio_unified8` | AddBiomechanics 코퍼스 둘 / 번들 |

세대를 구별하는 것은 이름이 아니라 번들 안의 `INDEX.json.generated_utc`, 실행 기록
`_runs/run_<identity>/`, 그리고 `_superseded/` 아래의 이전 세대 증거다.

### 1.2 재생성은 검증을 통과한 뒤 같은 이름으로 교체한다

- 같은 lineage를 다시 만들면 **같은 디렉터리 이름**으로 들어간다. 이전 세대와 나란히 두지
  않는다.
- 순서: (1) 이전 세대의 증거 파일(1.3(b))을 `_superseded/`로 복사하고 SUPERSEDED.md를 쓴다,
  (2) 소유자 승인으로 이전 payload를 지운다(1.5), (3) 같은 이름으로 생성하고 L4 `--full`
  검증을 통과시킨다. 새 세대가 검증에 실패하면 그 사실이 `_runs/`에 남고, 증거로 이전 세대를
  설명할 수 있어야 한다 — 그래서 (1)이 (2)보다 앞이다.
- 생성은 `soma-synth run`으로 한다. 러너가 실행 기록, `KNOWN_LIMITATIONS.json`, 검증, README,
  등록을 함께 남긴다 ([생성 파이프라인 표준](GENERATION_PIPELINE_STANDARD.md)).

### 1.3 삭제 전 반드시 통과해야 하는 세 가지 선행조건

세 가지를 **모두** 확인하기 전에는 어떤 payload도 지우지 않는다.

**(a) 살아 있는 코드 참조가 없을 것.** 저장소 전체에서 디렉터리 이름을 검색해 코드·설정·
테스트의 참조가 0건이어야 한다. 참조가 있으면 **먼저 lineage 이름으로 옮겨 붙이고**, 그
변경이 통과한 뒤에 지운다. 2026-09-14 개정 후 저장소의 참조는 모두 lineage 이름이며, 문서와
보고서의 `_vN` 언급은 그 세대를 가리키는 역사 기록으로 남긴다.

**(b) 증거는 payload와 분리해 보존할 것.** 삭제하는 것은 take payload이고, 그 세대가 무엇을
했는지 설명하는 작은 파일은 남긴다. 남길 파일:

`INDEX.json`(및 `INDEX_shard_*.json`), `VALIDATION_REPORT.json`, `validation_ledger.json`,
`SUMMARY.json`, `_run.json`, `DATA_DESCRIPTION_EN.md`, `DATASET_DESCRIPTION_EN.md`,
`README.md`, `KNOWN_LIMITATIONS.json`, `PROVENANCE_RETROFIT.json`, `PUSH_RECEIPT.txt`,
번들 루트의 그 밖의 작은 정책·기록 파일.

이들을 `_superseded/<지워지는 디렉터리 이름>/` 아래로 복사해 sha256으로 대조한 뒤 payload를
지운다. 버전 접미사가 있던 세대는 그 이름 그대로(`_superseded/hknu_unified8_v1/`), 개정 후의
세대는 `_superseded/<lineage>_<generated_utc 날짜>/`로 둔다.

> 이 조항이 필요한 이유: 이전 세대의 `INDEX.json` 이 무엇이 왜 빠졌는지의 유일한 기록일 수
> 있다. 예를 들어 결함(가령 각도 감김)으로 take 를 제외한 세대라면 그 `INDEX.json` 만이 제외된
> take 와 그 사유를 적고 있다. 디렉터리를 통째로 지우면 무엇이 왜 빠졌는지 설명할 수단이 사라진다.

**(c) 삭제를 기록할 것.** `_superseded/<이름>/SUPERSEDED.md`에 삭제 일자, 대체본(없으면
"없음"과 그 이유), 삭제 사유, 삭제 전 파일 수와 크기, 참조 재지정 커밋을 적는다.

### 1.4 절대 삭제 금지

- `<SOMA_DATA_ROOT>/raw_archives/` 및 `<SOMA_DATA_ROOT>/extracted/` 이하 **원천 데이터 일체**
  (`SOMA_SOURCE_ROOT` 로 원천 폴더를 따로 둔 경우 그 아래도). 이 규칙은 어떤 정리 작업으로도
  완화되지 않는다.
- `<SOMA_DATA_ROOT>/README.md`, 그리고 데이터 루트가 두는 경우의 기록 파일 `MASTER.md`(데이터 루트의
  색인)·`state/local_archive_inventory.json`(원천 아카이브 목록). 덮어쓰기도 금지. 없는 파일은 없는 대로 둔다.
- `.bak-*` 파일.
- `_superseded/`, `_runs/`, `_manifest_backfill/` — 증거와 실행 기록이며 payload가 아니다.
- 팀 공유 드라이브의 어떤 것도 이 규칙으로 지우지 않는다. 공유본 정리는
  별도 승인 사안이다.

### 1.5 삭제는 소유자 승인 사항

payload 삭제는 되돌릴 수 없다. 삭제는 데이터 소유자가 승인한 뒤에만 한다: 선행조건(1.3)을
확인하고 증거를 옮긴 다음, 지울 디렉터리의 **구체적 목록과 크기**를 소유자에게 보이고 승인을
받는다.

### 1.6 이름 규칙

- 새 lineage는 `<source>_<stage>`. 버전 번호를 이름에 넣지 않는다.
- 같은 lineage의 재생성은 1.2의 순서로 같은 이름을 다시 쓴다.
- `_dryrun_*`, `xrun_*`처럼 실험 흔적이 명백한 디렉터리는 lineage로 취급하지 않으며,
  그것을 만든 작업자가 정리 책임을 진다.

### 1.7 지난 정리 작업의 기록

지난 정리 작업(예: 2026-09-14 개정으로 버전 접미사가 붙은 옛 세대를 지운 일)의 기록은
데이터 루트의 증거 폴더, 곧 `_superseded/<지운 디렉터리 이름>/` 의 증거 파일과
`SUPERSEDED.md`(1.3(b)·(c))에 남으며, 지운 세대는 그 증거로 설명한다. 어떤 정리 작업이든
1.3의 선행조건과 1.5의 승인을 거치고, 팀 공유 드라이브의 사본은 그 범위 밖이다(1.4).

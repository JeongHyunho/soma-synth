# 생성 파이프라인 표준 — 실행 명세

| 항목 | 값 |
|---|---|
| Document ID | `SOMA-GENERATION-PIPELINE-STANDARD-001` |
| Version | `1.1.0` |
| 개정 이력 | `1.0.0` (2026-09-09) 최초. `1.1.0` (2026-09-25) `ADR-0041`(상위 프로젝트: `docs/adr/ADR-0041-soma-synth-split-and-teammate-generation.md`) 반영: 레지스트리 v2(코퍼스 단계·후처리·생성 정책·`selection_args`), §1.3 모든 생성은 빈 폴더에 하고 재개는 없다(비어 있지 않은 폴더는 `--replace-existing` 이 있어야 옆으로 옮기고 새 폴더에 만든다: 성공하면 운영 폴더의 것만 남기고 지우고, 실패하면 되돌린다; `force_arg` 없음; 죽은 실행은 처음부터 다시 만든다는 맞바꿈 선언), 끝난 세대를 담은 운영 폴더 교체에 `_superseded` 증거 요구, generate 단계의 잠금(`<폴더>.lock`), 읽기 전용·증거 폴더 거부, 다른 운영 폴더 거부, 쓸 곳 없는 인자 거부, 끝나지 않은 생성 단계의 `.generating` 표지, `pipeline`·`validate`·`readme`·`register`·push 의 같은 거부), §2.1 설치, §3.1 `resolved_config_sha256` 채움, §4.1 추적 기록과 그 빈틈, §6 동일성, §9 생성 정책(`pipeline` generate 거부, 모든 PC 에 같은 조건: 2026-09-25 결정), §10 불일치 두 줄 |
| 상태 | `ACTIVE_BASELINE` |
| 기준일 | `2026-09-25` |
| Authority | 봉인 문서 `PIPELINE_GOVERNANCE.md`(상위 프로젝트: `docs/guides/PIPELINE_GOVERNANCE.md`) §5.1·§10.1·§10.2 의 **실행 명세**. 그 문서를 개정하지 않으며, 어떤 hold 도 해제하지 않는다. 생성의 근거는 이 문서가 아니라 §9 가 적는 결정 기록이다 |
| 상위 문서 | `PIPELINE_GOVERNANCE.md`(상위 프로젝트: `docs/guides/PIPELINE_GOVERNANCE.md`), `MASTERPLAN.md`(상위 프로젝트: `docs/architecture/MASTERPLAN.md`) |
| 기계 판독본 | [`configs/datasets/source_pipelines_v2.yaml`](../../configs/datasets/source_pipelines_v2.yaml) (기본값). v1 [`source_pipelines_v1.yaml`](../../configs/datasets/source_pipelines_v1.yaml) 은 쓰인 설정이라 그대로 두며 여전히 읽힌다 |
| 짝 문서 | `ADR-0040`(상위 프로젝트: `docs/adr/ADR-0040-dataset-profiles-registry-and-envelope-conformance.md`) — 산출물 쪽 규격 |
| 근거 | `ADR-0041`(상위 프로젝트: `docs/adr/ADR-0041-soma-synth-split-and-teammate-generation.md`) |

> **이 문서는 새 정책이 아니다.** `PIPELINE_GOVERNANCE.md` 가 이미 요구하는 것을 **어떻게
> 이행하는지** 적는다. 그 문서는 M0 evaluator 에 sha-pin 된 봉인 파일이므로 여기서 고치지
> 않는다. 둘이 어긋나면 봉인 문서를 따르고 이 문서를 고친다.
>
> **이 저장소.** 이 문서는 soma-synth 저장소에 있다. 2026-09-25 분리 때 생성 파이프라인과
> 함께 상위 프로젝트에서 옮겨 왔다. 봉인 문서(`PIPELINE_GOVERNANCE.md`, `MASTERPLAN.md`,
> `LOCAL_DATA_PLANE.md`), ADR, `research/decisions` 의 결정 기록, 보존 규칙은 상위 프로젝트에
> 남아 있다. 이 문서는 그것들을 링크하지 않고 "상위 프로젝트: `<경로>`" 처럼 상위 프로젝트 루트 기준
> 경로를 글로 적는다. 이 저장소가 상위 프로젝트에 `packages/soma-synth` 로 마운트돼 있어도 이 문서에서
> 상위 프로젝트 파일로 가는 상대 링크는 열리지 않기 때문이다. 링크는 이 저장소 안의 파일만 가리키며,
> `src/soma_synth/`, `scripts/`, `configs/`, `tests/`, `packages/smpl18` 은 이 저장소 기준이다. 보존 규칙은
> 같은 번호와 뜻으로 [`RETENTION_RULES.md`](RETENTION_RULES.md) 에 옮겨 둔다(상위 프로젝트 규칙과 어긋나면 그
> 규칙이 이긴다). 이 저장소의 메시지가 적는 "retention rule 1.x" 는 그 번호의 규칙이다.

---

## 0. 왜 이 문서가 필요한가

`PIPELINE_GOVERNANCE.md` 는 2026-08-03 부터 §5.1 에서 adapter 의 7단계 분리를, §10.1 에서 run
디렉터리 7종을, §10.2 에서 provenance 10항목을 요구해 왔다. 2026-09-09 에 실측한 결과는 다음과
같다.

| 규정 | 실측 |
|---|---|
| §5.1 7단계 분리 | 구현 0. 생성기 11개가 각자 흐름을 갖는다 |
| §10.1 run 디렉터리 7종 | 5개 번들 전부 **0종** |
| §10.2 provenance 10항목 | **4항목**만 존재 |
| §10.1 `run_id` = config·code·source 해시 | `run_id` 는 take 라벨(`gaitex-austra-gwo-unified8-marker`) |

**기록된 것만으로는 어떤 번들도 재현할 수 없다.** 어느 코드 리비전으로, 어느 SMPL 모델로
만들었는지가 없다. 이 문서와 그 짝인 코드가 그 공백을 메운다.

### 0.1 왜 지금까지 이행되지 않았는가

구현 의지의 문제가 아니라 **구현할 자리가 없었다.** 생성기들은 공유 코드를 import 하지 않고
파일 경로로 런타임 로드한다(`spec_from_file_location`, 12개 파일). 한 생성기가 다른 생성기의
라이브러리다 — `generate_amass_faithful_all.py` 는 `generate_amass_faithful.py` 를 경로로 읽어
실행한다. 따라갈 import 그래프가 없고, 단계를 걸 모듈 경계가 없다.

이 문서는 그 구조를 **당장 걷어내지 않는다**. `amass` 와 `prism` 은 지금 검증을 통과한 두
번들을 만든 코드이고, 옮기다 산출이 달라지면 재현성을 잃는다. 대신 러너가 기존 진입점을
**감싸고**, 표준이 요구하는 것을 그 바깥에서 채운다.

---

## 1. 정규 단계

앞의 일곱은 `PIPELINE_GOVERNANCE.md` §5.1 의 이름을 그대로 쓴다. 뒤의 여덟은 이 lineage 가 그
뒤에 더하는 합성·방출이다.

| # | 단계 | 하는 일 | 오늘 어디에 있나 |
|---|---|---|---|
| 1 | `register_asset` | 원천 자산 등록, SHA-256 확보 | `SourceIdentity.asset_sha256` |
| 2 | `inspect_native` | 디코드 없이 형태·개수 확인 | 생성기 `main()` 의 순회 |
| 3 | `validate_license` | 이용 조건·gate 확인 | `configs/gates/source_gates_v1_3.yaml`(생성기 밖) |
| 4 | `load_native` | 원천 형식으로 읽기 | 생성기 `convert()` 앞부분 |
| 5 | `validate_native` | 원천 QC | 부분적, 소스마다 다름 |
| 6 | `convert_intermediate` | SMPL-24 리타깃 | 별도 코퍼스 생성 단계 — 레지스트리 v2 의 `corpus`, 러너가 실행(§1.3) |
| 7 | `emit_source_qc_and_provenance` | 원천 QC·provenance 방출 | `SourceIdentity`(31필드) |
| 8 | `fit_shape` | betas·rest skeleton | `addbio_retarget.shape_fit` |
| 9 | `build_large` | 18관절 kinematics | `unified8_emit.build_bundle`; 축소는 `pipeline.reduced_model`(§2.2) |
| 10 | `build_anthro` | rest 상수 | `unified8_emit.build_bundle` |
| 11 | `synthesise_small` | 8채널 IMU | **소스별로 갈리는 유일한 단계** |
| 12 | `resample_100hz` | 100 Hz 격자 | `unified8_emit.build_bundle` |
| 13 | `emit_take` | 5종 산출물 + take manifest | `unified8_emit.write_bundle` |
| 14 | `emit_bundle` | INDEX·설명서·README | `unified8_emit.write_data_description` |
| 15 | `close_run` | run 디렉터리 7종 확정 | **없음 — 이 표준이 신설** |

### 1.1 소스별로 갈리는 단계는 하나뿐이다

`synthesise_small` 만 `small_mode` 에 따라 경로가 다르다.

| `small_mode` | 소스 | 경로 |
|---|---|---|
| `measured_physical` | prism | 착용 센서의 실측값 |
| `synthetic_from_smpl` | amass · hknu · addbiomechanics | SMPL forward kinematics |
| `synthetic_from_markers` | gaitex | 마커 클러스터 강체 적합 (ADR-0039) |

나머지 열네 단계는 모든 소스가 같은 일을 한다. **규격화의 여지는 여기에 있다.**

### 1.2 `unified8_emit` 이 이미 공통 척추다

`scripts/poc/unified8_emit.py` 가 `build_bundle`·`write_bundle`·`write_data_description` 과
`SourceIdentity`(31필드)를 제공하고, `addbiomechanics`·`gaitex`·`hknu` 가 그것을 쓴다.
`amass` 와 `prism` 은 쓰지 않는다 — 그쪽이 먼저 있었기 때문이다.

그 모듈의 docstring 이 이유를 이미 적어 두었다: `generate_amass_faithful.build` 를 리팩터링하지
않은 것은 "그 함수가 카탈로그에 등록된 L4-PASS 데이터셋 둘을 만들고, 편의를 위해 그것을 위험에
빠뜨릴 수 없기" 때문이며, 대신 **동등성 테스트**(`tests/poc/test_unified8_emit.py`, 6건)가 두
경로를 같은 SMPL 입력에 돌려 배열을 비교한다.

**이 표준은 그 판단을 뒤집지 않고 채택한다.** 통합의 조건은 "동등성을 먼저 증명한다"이고, 그
장치는 이미 있다.

### 1.3 생성 단계는 코퍼스 → 번들 → 후처리다 (레지스트리 v2, 2026-09-25)

`hknu`·`gaitex`·`addbiomechanics` 는 원천을 먼저 SMPL-24 리타깃 코퍼스로 바꾸고, 번들은 그
코퍼스에서 만든다. 코퍼스는 지금까지 주어진 입력이었다. 새 PC 에는 코퍼스가 없고, 있던 코퍼스도
지워질 수 있다. 그래서 레지스트리 v2 가 각 소스의 `corpus`
블록(진입점, lineage, 출력 플래그, 지문 파일, 번들에 넘기는 플래그)을 선언하고, 러너가 그것을
실행한다.

| 순서 | 무엇 | 언제 |
|---|---|---|
| 1 | 코퍼스 진입점 → `<data root>/runs/experimental_generation_poc_demo/<corpus lineage>` (또는 `--corpus`) | 그 폴더에 끝난 코퍼스(지문 파일, `.generating` 없음)가 없을 때, 또는 `--rebuild-corpus`. 끝난 코퍼스는 재사용한다(읽기만) |
| 2 | 번들 진입점, 코퍼스는 선언된 플래그(`--paired`·`--retarget`·`--raw`)로 | 항상 |
| 3 | 후처리(`post_steps`) — `prism` 의 `synthesize_insole_heading.py <번들>` | 번들이 끝난 뒤, `KNOWN_LIMITATIONS.json` 렌더 전 |

**모든 생성은 빈 폴더에 한다. 재개는 없다.** 번들 진입점과 코퍼스 진입점은 언제나 없거나 빈 폴더에 쓴다.
러너는 폴더를 제자리에서 덮지도, 끝나지 않은 폴더를 이어 만들지도 않는다(아래 "교체는 새 폴더에 만든다").
그래서 몇 시간 돈 실행이 죽거나 멈추면 **처음부터 다시 만든다.** 이것이 선언한 맞바꿈이다. 한 번들에 두
실행(또는 두 코퍼스)의 take 가 섞이지 않고, 생성기마다 다른 재개 판정(§7)과 재개의 손실(`prism` 은 건너뛴
take 를 `"skipped"` 로 적어 색인의 `ok` 에서 뺀다)을 러너가 알 필요가 없는 대신, 끊긴 긴 실행의 시간을 잃는다.
생성기 자신의 `--resume`·`--force` 는 운영자가 `--corpus-arg`·`--bundle-arg` 로 넘길 수 있지만, 빈 폴더에는
건너뛸 것이 없다. 러너는 다음을 실행 전에 거부한다.

| 거부 | 조건 | 풀어 주는 것 |
|---|---|---|
| 비어 있지 않은 번들 폴더 | 번들 폴더가 있고 무엇이든 들어 있다: 끝난 번들(`INDEX.json`, `.generating` 없음), 끝나지 않은 생성(`.generating`), 그 밖의 파일. 거부는 무엇이 들어 있는지(표지를 남긴 실행 기록까지) 적는다 | `--replace-existing`(폴더를 옆으로 옮기고 빈 폴더에 만든다), 또는 없거나 빈 폴더 |
| 비어 있지 않은 코퍼스 폴더 | 코퍼스를 만들어야 하는데(끝난 코퍼스가 없거나 `--rebuild-corpus`) 코퍼스 폴더에 무엇이든 들어 있다: 끝난 코퍼스, 끝나지 않은 생성, 그 밖의 파일. `--rebuild-corpus` 없는 끝난 코퍼스는 재사용하므로 거부하지 않는다 | `--rebuild-corpus --replace-existing` 둘 다, 또는 `--corpus` 로 없거나 빈 폴더 |
| 운영 폴더의 끝난 세대 교체 | 교체하려는 폴더가 운영 폴더(어느 루트든 `runs/experimental_generation_poc_demo` 바로 아래의 폴더)이고 끝난 세대(`INDEX.json` 또는 코퍼스 지문, `.generating` 없음)를 담고 있다 | 그에 더해 옆의 `_superseded/<이름>_*` 한 곳에 그 `INDEX.json`(또는 지문)과 SHA-256 이 같은 사본과 `SUPERSEDED.md` (상위 프로젝트 보존 규칙 1.2 (1)). 끝나지 않은 생성에는 대조할 끝난 세대가 없으므로 묻지 않는다 |
| 한 번에 한 세대 | 폴더를 옆으로 옮겨야 하는데 그 옆에 `<폴더>.replaced-*`·`<폴더>.failed-*` 가 있다 | 거부가 방법을 적는다. `.replaced-*` 는 지금 폴더의 세대를 받아들이면 지우고(운영 폴더의 것은 소유자 승인으로, 상위 프로젝트 보존 규칙 1.5), 옛 세대로 돌아가려면 지금 폴더를 치우고 `.replaced-*` 의 이름을 되돌린다. `.failed-*` 는 살펴본 뒤 지운다 |
| 잠금 | 번들 폴더 옆에, 또는 (러너가 쓸 수 있는 곳의) 코퍼스 폴더 옆에 `<폴더>.lock` 이 있다: 다른 generate 단계가 그 폴더를 만들거나 읽는 중이다 | 그 실행이 끝나기를 기다린다. 거부가 잠금의 주인(pid·호스트·run identity)과 나이, 그 프로세스가 살아 있는지 확인하는 명령(`tasklist /FI "PID eq <pid>"`·`ps -p <pid>`)을 적는다. 죽은 실행이 남긴 잠금은 손으로 지운다. 러너는 잠금을 지우지 않는다 |
| 쓸 곳 없는 인자 | `--corpus-arg` 인데 소스에 코퍼스 단계가 없다(`amass`·`prism`), 또는 명시적 생성 명령(`generate_cmd`)과 함께 `--corpus-arg`·`--bundle-arg` 를 받았다. 버려지면 표본 선택이 사라져 원천 전체가 돈다 | 번들 인자는 `--bundle-arg` 로(선택 인자는 운영 lineage 검사를 거친다) |
| 표본 인자 | 출력이 운영 lineage 폴더(이 run 의 data root 아래이든, 다른 루트의 `runs/experimental_generation_poc_demo/<lineage>` 모양이든)인데 레지스트리의 `selection_args`(`--subjects`·`--trials`·`--only` 등, `--subjects=S04`·축약형 포함)를 받았다 | 출력을 scratch 폴더로 |
| 표본 코퍼스 → 운영 번들 | 번들 폴더가 운영 lineage 인데 코퍼스가 그 옆의 자기 코퍼스 lineage 가 아니다 | 번들도 scratch 폴더로 |
| 표본 실행 → 운영 코퍼스 생성 | 코퍼스 폴더가 운영 코퍼스 lineage 이고 이번 실행이 그것을 만들거나 다시 만드는데(지문 없음, 또는 `--rebuild-corpus`), 번들이 그 옆의 운영 번들 lineage 가 아니거나 선택 인자가 있다. 끝난 운영 코퍼스를 **재사용**만 하는 표본은 거부하지 않는다 | `--corpus <scratch 폴더>` |
| 러너가 정하는 플래그 | `--corpus-arg`·`--bundle-arg` 가 코퍼스 출력(`--out`), 번들 출력(`--out`·`--out-root`), 번들에 넘기는 코퍼스 플래그(`--paired`·`--retarget`·`--raw`), 원천 위치(`--source-root`·`--root`·`--hknu-root`·`--amass-root`·`--extracted`), PRISM `--data-root` 를 준다(`--x=v`·argparse 축약 포함; argparse 가 선언된 다른 플래그로 읽는 축약은 통과) | 번들은 `soma-synth run` 의 dataset_dir, 코퍼스는 `--corpus`, 데이터 루트는 `--data-root`, 원천은 `SOMA_SOURCE_ROOT` |
| 동기화 폴더(**거부 아님, 경고만**) | 번들 폴더·코퍼스 폴더(이번 실행이 만들거나 다시 만들 때)·실행 기록 폴더(`--runs-root`)·카탈로그(register 단계가 있을 때)가 Dropbox·OneDrive·Google Drive(`My Drive`·`내 드라이브`·`Shared drives`·`공유 드라이브` 등)·Synology Drive·iCloud Drive 폴더나 `SOMA_SHARED_DRIVE_NAMES` 가 이름을 적은 팀 공유 드라이브(쉼표로 나눈 폴더 이름, 기본값 없음; `paths.on_shared_drive`) 안이다. 폴더 이름은 운영체제와 상관없이 대소문자 없이 맞춘다(`~/Dropbox`·`~/OneDrive*`·`~/Google Drive`·`/Volumes/GoogleDrive*` 포함). macOS 의 `~/Library/CloudStorage/*`(File Provider 동기화 클라이언트 전부)·`~/Library/Mobile Documents`(iCloud), Linux 의 gvfs `google-drive:` 마운트는 POSIX 경로에서만 본다(`paths.synced_location`). 이 경우 실행은 거부되지 않고 계속되며, 경로마다 한 번 표준 오류에 `warning:` 줄이 나온다(`paths.warn_if_synced`; 2026-09-30 소유자 결정). 동기화 클라이언트가 쓰는 중인 파일을 잠그거나 늦추거나 일부만 올릴 수 있고, 그곳의 번들은 이 PC 밖으로 공유될 수 있어(공유는 별도 승인) 로컬 디스크를 권한다. `readme`·`register`·`validate`·`hash-sources --out`·`stage-sources --to` 도 같다 | 경고를 없애려면 로컬 디스크의 `SOMA_DATA_ROOT` 아래로 |
| 읽기 전용·증거 폴더 | 번들 폴더, 그리고 이번 실행이 만들거나 다시 만드는 코퍼스 폴더가 이 run 의 data root 나 `SOMA_DATA_ROOT` 의 `extracted`·`raw_archives`·바디 모델 폴더(`SOMA_BODY_MODEL_DIR` 포함)·`_superseded`·`_manifest_backfill`·`_runs` 안이거나, `SOMA_SOURCE_ROOT` 의 소스 폴더 안이거나, 컨테이너(data root·`runs`·`runs/experimental_generation_poc_demo`·`SOMA_SOURCE_ROOT`) 자체다. 번들은 읽는 코퍼스 안이나 그것을 담은 폴더일 수 없다(`paths.check_output_dir(root=...)`). 두 루트가 아닌 다른 루트의 증거 폴더(`runs/experimental_generation_poc_demo/{_superseded,_runs,_manifest_backfill}`) 안도 거부한다(`paths.evidence_folder`). 실행 기록 폴더와 카탈로그는 `_runs` 에는 쓸 수 있고 그 밖의 위 폴더 안에는 쓸 수 없으며, 컨테이너 자체와 lineage 폴더(`runs/experimental_generation_poc_demo/<lineage>`, 번들이나 코퍼스가 든다) 안에도 쓸 수 없다(`paths.check_record_dir`) | scratch 폴더나 lineage 폴더로(실행 기록은 `<컨테이너>/_runs`, 카탈로그는 `<data root>/experimental/catalog`) |
| 다른 운영 폴더 | 번들 폴더나 (이번 실행이 읽는) 코퍼스 폴더가 어느 루트든 `runs/experimental_generation_poc_demo` 바로 아래에 있는데 이 소스의 자기 번들·코퍼스 lineage 가 아니다: 레지스트리가 선언한 다른 lineage(`prism` 을 `amass_faithful_full` 에, `hknu` 번들을 `hknu_smpl24_paired` 에, `hknu` 코퍼스로 `gaitex_smpl24` 를)이든, 아무도 선언하지 않은 폴더(`_dryrun_*`, `my_experiment` 등)이든 (`runner.production_directory_refusal`) | 자기 lineage 폴더나 scratch 폴더로 |
| 교체 흔적 | 번들 폴더나 코퍼스 폴더 자체가 옆으로 옮긴 폴더나 실패한 교체의 산출(`*.replaced-*`·`*.failed-*`)이다 | 원래 폴더(이름에서 `.replaced-…`·`.failed-…` 를 뗀 것)로 |

마지막 네 줄의 거부는 실행 기록을 열기 **전에** 한다. 실행 기록이 바로 그 폴더나 그 옆에 쓰이기 때문이다(기본값은
`<번들의 부모>/_runs`). 그래서 기록은 남지 않고 `soma-synth run` 이 종료 코드 3 으로 끝나며 사유를 표준 오류에
쓴다. 나머지 거부는 실행 기록의 `logs/refusal.txt` 에 남는다. 읽기 전용·증거 폴더 거부가 필요한 까닭은 러너
자신의 쓰기다. 폴더를 옆으로 옮기는 일, 실행 기록, validate·readme 가 쓰는
`VALIDATION_REPORT.json`·`validation_ledger.json`·`README.md`, `KNOWN_LIMITATIONS.json` 은 생성기의 가드보다
먼저이거나 생성기 없이(`--skip-generate`, `--start-at validate`) 일어나고, 생성기의 가드는 러너가 넘긴 data
root 만 본다.

옛 `soma-synth pipeline` 명령도 같은 거부를 한다. validate·readme 가 번들 폴더에 쓰면 그 폴더에
`paths.check_output_dir(root=<data root>)`, 번들 밖에 쓰는 `--ledger`·`--report` 에는 `paths.check_record_file`
(그 폴더에 `check_record_dir`, 그리고 data root 의 `README.md` 와, 있으면 `MASTER.md`·`state/local_archive_inventory.json`
은 덮지 않는다), register 가 있으면 카탈로그에 `paths.check_record_dir`, readme 의 `--top-level-root` 에는
`paths.check_readme_root`(data root 나 `SOMA_SOURCE_ROOT` 자체, 읽기 전용·증거 폴더 안은 거부하고 lineage
컨테이너는 허용한다. `<data root>/README.md` 는 덮지 않는다, [보존 규칙](RETENTION_RULES.md) 1.4)를 걸고,
`--source` 의 번들이 아닌 다른 lineage 도 거부한다. 거부되면 아무것도 쓰기 전에 종료 코드 3 으로 끝난다.
`soma-synth readme --top-level-root` 도 같은 `check_readme_root` 를 건다.

`soma-synth validate`·`readme`·`register` 를 따로 부를 때도 같다. 번들 폴더에 쓰는 명령 —
`readme`, 그리고 `--report`·`--ledger` 가 번들 안을 가리키는 `validate` — 은 그 폴더에
`paths.check_bundle_dir(root=<data root>)` 를 건다: `check_output_dir` 전부에 더해, 어느 루트든
`runs/experimental_generation_poc_demo/{_superseded,_runs,_manifest_backfill}` 안이면(`paths.evidence_folder`;
`SOMA_DATA_ROOT` 가 그 루트를 가리키지 않아도) 거부한다. `_superseded/<이름>/` 에는 교체한 세대의
`INDEX.json`·`README.md`·검증 보고서가 보관돼 있으므로 `readme` 는 그 `README.md` 를 덮지 않는다.
`register` 는 번들에 쓰지 않지만 보관된 증거를 살아 있는 자산으로 올리지 않도록 같은
증거 폴더와 읽기 전용 폴더(`extracted`·`raw_archives`·소스·바디 모델 폴더)를 거부하고
(`check_bundle_dir(writes=False)`), `--catalog` 에 `check_record_dir` 를 건다. `pipeline` 도 register 만 하는
실행(`--start-at register`)에서 같은 번들 검사를 한다. 번들 밖에 쓰는 `--report`·`--ledger` 는 `check_record_file`
을 받고, 그때 번들은 읽기만 하므로 거부하지 않는다. `--report`·`--ledger` 없는 `validate` 는 아무것도 쓰지
않으므로 위치를 묻지 않는다. 거부는 종료 코드 3 이고, 아무것도 쓰기 전이다.

**끝나지 않은 번들은 넘겨받지 않는다.** `soma-synth validate`·`readme`·`register`·`pipeline`, 러너의
validate·readme·register 단계, 상위 프로젝트의 `scripts/push_sample_data.py` 는 다음을 거부한다(종료 코드 3, 다시 만드는 명령을
적는다): `.generating` 이 있는 번들(`soma-synth run --source <소스> <번들> --replace-existing` 으로 처음부터 다시
만든다), 그리고 이름이 `*.replaced-*`(옆으로 옮긴 옛 세대)·`*.failed-*`(실패한 교체의 산출)인 폴더(번들이
아니다. 원래 폴더를 쓴다). `register`·`pipeline` 은 `--source` 의 번들 lineage 가 아닌, 레지스트리가 선언한 다른
lineage 도 거부한다.

`--compress`·`--resume`·`--force` 처럼 일부를 고르지 않는 인자는 운영 lineage 에도 넘어간다. 운영 lineage 를
바꾸는 것은 어느 PC 에서든 [보존 규칙](RETENTION_RULES.md) 1.2 의 순서(증거를 `_superseded/` 로 → 소유자 승인 삭제 →
재생성)를 따른다. 그 순서대로 해서 삭제 뒤의 폴더가 없거나 비어 있으면 `--replace-existing` 이 필요 없다.
`--replace-existing` 은 그 순서를 대신하지 않는다. 끝난 세대를 담은 운영 폴더에 주면 러너는 그 폴더 옆
`_superseded/<이름>_*` 가운데 한 곳이 지금 바꾸려는 `INDEX.json`(또는 코퍼스 지문)과 SHA-256 이 같은 사본과
`SUPERSEDED.md` 를 가질 때만, 곧 1.2 (1) 을 이 세대에 대해 마쳤을 때만 진행한다. 코퍼스 지문에는
`generated_utc` 가 없으므로 폴더 날짜가 아니라 내용으로 찾는다. scratch 폴더는 이 확인 없이 바꾼다. 재사용한
코퍼스는 `--corpus-arg` 를 받지 않는다. 로그와 `manifest.json` `corpus.corpus_args_used: false` 가 그것을
적는다(같은 명령으로 표본 실행을 다시 돌릴 수 있게 거부하지는 않는다).

**교체는 새 폴더에 만든다.** 비어 있지 않은 폴더를 `--replace-existing`(코퍼스는 `--rebuild-corpus` 와 함께)으로
바꿀 때, 러너는 폴더 전체를 같은 볼륨의 `<폴더>.replaced-<run>` 으로 이름만 바꾸고(`os.replace`, 복사하지
않는다), `os.replace` 가 돌아오는 즉시 그 옆 폴더를 이 실행이 되돌릴 목록에 올린다. 그다음 작은 증거
파일(`INDEX.json`·`INDEX_shard_*.json`·`SUMMARY.json`·`_run.json`·`VALIDATION_REPORT.json`·
`validation_ledger.json`·`KNOWN_LIMITATIONS.json`·`README.md`·`DATA_DESCRIPTION_EN.md`·`reduced_model_fit.json`·
코퍼스 지문 등, `runner.REPLACED_EVIDENCE_FILES`)을 실행 기록의 `replaced/<bundle|corpus>/` 에 복사해
`manifest.json` `replaced.<bundle|corpus>` 에 SHA-256 과 폴더가 담고 있던 것(`found`), 옮긴 곳과 함께 적은 뒤,
진입점을 빈 폴더에 돌린다. `<run>` 은 run identity 앞 12자다(같은 identity 를 반복한 실행이면 기록 폴더처럼
`-2` 가 붙는다). 그래서 옛 세대의 take·샤드 색인·검증 보고서·코퍼스 파일이 새 세대에 남을 수 없고, 진입점에
건너뛴 take 를 다시 만들라는 플래그를 줄 필요도 없다. 레지스트리 v2 는 그런 플래그(`force_arg`)도 재개에 관한
필드도 두지 않으며, 연산자가 `--bundle-arg=--force` 로 준 것은 다른 인자처럼 넘어간다. 빈 폴더에 만들므로
`amass` 가 `reduced_model_fit.json` 의 subject fit 을 다시 쓰던 일도 없다. 번들 폴더는 코퍼스 단계와 그 등급
검사가 끝난 뒤, 번들 진입점 바로 앞에서 옮긴다. 코퍼스 단계가 실패하면 번들은 손대지 않은 그대로다.

- **성공하면** 생성 단계 전체(진입점, 후처리, 등급 검사, `KNOWN_LIMITATIONS.json`, `source_manifest.json`)가
  끝난 뒤 옆으로 옮긴 폴더를 지운다(`replaced.<what>.outcome: deleted`). 단 운영 폴더 — 어느 루트든
  `runs/experimental_generation_poc_demo` 바로 아래의 폴더, 레지스트리가 선언한 lineage 만이 아니다 — 의 것은
  지우지 않는다. 운영 payload 삭제는 소유자 승인 사항이므로([보존 규칙](RETENTION_RULES.md) 1.5) 그 자리에 두고, generate 단계의
  결과가 그 경로를 적는다(`kept_production_directory`).
- **실패하면** 이번 실행이 쓴 표지 하나 말고는 아무것도 지우지 않는다. 새 폴더는 `<폴더>.failed-<run>` 으로
  이름을 바꾸고(표지밖에 없으면 지운다), 옆으로 옮긴 폴더를 `<폴더>` 로 되돌린다. 이전 상태가 바이트 그대로
  돌아온다(`outcome: restored`, `failed_output_relative`). 코퍼스와 번들을 함께 옮긴 실행이면 둘 다 되돌린다.
  단계가 예외로 죽어도 같다. 옆으로 옮긴 직후 증거 파일을 복사하다 예외가 나도 같다. 되돌리지 못하면(파일이
  잠겼다 등) 결과가 옛 세대가 있는 경로를 적고 손으로 되돌리라고 한다.
- **한 번에 한 세대.** `<폴더>.replaced-*` 나 `<폴더>.failed-*` 가 옆에 있는 동안 그 폴더는 다시 교체하지
  않는다(위 표). 운영 폴더의 교체가 성공하면 옛 세대가 옆에 남으므로, 다음 교체는 소유자가 그것을 승인해 지운
  뒤에만 된다.
- **죽은 실행.** 프로세스가 죽어 실패 처리를 못 하면 새 폴더에 `.generating` 이, 그 옆에 `<폴더>.replaced-<run>`
  과 잠금 파일이 남는다. 어떤 실행도 그것을 넘겨받거나 이어 만들지 않는다. 잠금이 있는 동안은 잠금 때문에
  거부되고, 잠금을 손으로 지운 뒤에는 폴더가 끝나지 않은 생성이라 `--replace-existing` 없이 거부되며, 주어도 옆
  폴더 때문에 거부된다. 그 거부가 되돌리는 법(지금 폴더를 치우고 `.replaced-*` 의 이름을 되돌린다)과 정리하는
  법을 적는다.

**잠금.** generate 단계는 폴더를 살피기 전에 번들 폴더 옆, 그리고 코퍼스 단계가 있으면 코퍼스 폴더 옆에
`<폴더>.lock` 을 `O_CREAT | O_EXCL` 로 만든다. 먼저 만든 실행 하나만 성공한다. 내용은 pid, 호스트, 시작 시각,
run identity, 실행 기록 위치다. 단계가 어떻게 끝나든(성공, 실패, 예외, 거부) 지운다. 단 그 파일이 이 실행이 쓴
그대로일 때만 지운다(운영자가 지운 뒤 다른 실행이 잡은 잠금은 그 실행의 것이다). 코퍼스는 읽기만 하는
실행도 잠근다. 번들이 읽는 동안 다른 실행이 그것을 다시 만들지 못하게 하려는 것이다. 그래서 같은 코퍼스를
읽는 두 표본 실행도 동시에 돌지 않는다. 어떤 실행도 쓸 수 없는 곳(증거·읽기 전용 폴더,
`paths.check_output_dir`)의 코퍼스는 아무도 다시 만들 수 없으므로 잠그지 않고, 그 옆에 아무것도 쓰지 않는다.
러너는 남의 잠금을 지우지 않는다. 오래된 잠금의 거부 메시지가 주인과 나이, 그리고 그 프로세스가 살아 있는지
확인하는 명령을 적는다.

**끝나지 않은 생성 단계.** 생성기는 `INDEX.json` 을 쓰고 끝나지만, 생성 단계는 그 뒤에 후처리(`prism` 의
`synthesize_insole_heading.py`), `KNOWN_LIMITATIONS.json`, `source_manifest.json` 이 남아 있다. 그래서 러너는 빈
폴더에 만들기 시작할 때(번들 진입점 앞) `.generating`(어느 실행이 만드는지: 실행 기록 위치, run identity, 시작
시각)을 쓰고, 이 모두가 끝난 뒤에만 지운다. 코퍼스 단계도 코퍼스 진입점 앞에 코퍼스 폴더에 같은 표지를 쓰고,
지문과 등급 검사가 끝난 뒤에 지운다. 표지는 무엇을 이어 만들지 적지 않는다. 쓰임은 거부뿐이다. 표지가 있는
코퍼스는 재사용하지 않는다. 후처리가 실패하면 번들에는 새 `INDEX.json` 과 `.generating` 이 함께 남는다. 이
번들은 끝난 번들이 아니다. validate·readme·register(러너의 단계든 `--skip-generate`·`--start-at validate` 든
`soma-synth` 명령이든)·`pipeline`·push(상위 프로젝트의 스크립트) 가 다시 만드는 명령을 적어 거부하고(`logs/refusal.txt`, 종료 코드 3),
다음 generate 단계도 `--replace-existing` 없이는 거부한다. 주면 폴더를 옆으로 옮겨 처음부터 만든다. 그
`INDEX.json` 은 끝난 실행이 쓴 것이 아니므로 `_superseded` 와 대조하지 않는다(운영 폴더면 옮긴 폴더는 남기고
결과에 적는다). 생성기가 시작도 못 하고 실패해 폴더에 표지만 남으면 표지를 지우고, 이번 실행이 만든 폴더면
폴더째 지운다(코퍼스도 같다).

**코퍼스 등급.** 코퍼스 지문 파일이 `artifact_class`·`distribution_scope` 를 적으면 번들과 같은 검사를 한다.
`hknu`·`gaitex` 의 지문은 두 필드를 적는다. `addbiomechanics` 의 `SUMMARY.json` 은 적지 않으므로(각 artifact 가
적는다) `corpus.class_check.declared: false` 로 남는다.

---

## 2. 단계 간 계약

- 각 단계는 **앞 단계의 산출만** 입력으로 받는다. 전역 상태를 읽지 않는다.
- 각 단계는 실패 시 **부분 산출을 남기지 않는다**. take 디렉터리는 완성되거나 존재하지 않는다.
- `close_run` 을 제외한 모든 단계는 take 단위이며, `close_run` 은 run 단위다.
- 단계는 건너뛸 수 있으나(재개), **건너뛴 사실이 `metrics.json` 에 남는다.**

### 2.1 실행 환경 — `smpl18` 이 경로에 있어야 한다 (2026-09-16)

SMPL 변환의 규칙(성별별 모델 파일, 골격 정의, 순운동학)은 **별도 저장소**가 소유하고, 이
저장소는 그것을 `packages/smpl18` 에 **submodule** 로 마운트한다(비공개
`JeongHyunho/smpl18`, 2026-09-16 교체). 러너와 생성기가 `import smpl18` 로 불러 쓰므로,
submodule 이 채워져 있어야 하고 그 경로가 실행 환경에 있어야 한다.

패키지 이름은 2026-09-16 `smpl24` 에서 `smpl18` 로 바뀌었다 — 그것이 내놓는 코퍼스가 18관절
축약 모델이기 때문이며(전체 24관절 SMPL 포즈를 거친 뒤 네 관절을 피험자별 상수로 고정하고 손
관절 둘을 버린다), 원격 저장소도 `JeongHyunho/smpl18` 로 옮겼다. 이전 이름으로 된 경로·import
는 더 이상 없다.

```bash
# 한 번: submodule 을 채운다 (새로 clone 할 때는 git clone --recurse-submodules)
git submodule update --init

# 설치해 둔다 (둘 다 편집 가능 설치, 기준 환경의 버전은 constraints.txt)
python -m pip install -c constraints.txt -e packages/smpl18 -e .
soma-synth run --source <name> ...

# 설치 없이 개발 실행: 두 소스 트리를 경로에 둔다 (macOS / Linux 는 구분자가 ':')
PYTHONPATH="src:packages/smpl18/src" python -m soma_synth.cli run --source <name> ...
```

```powershell
# 설치 없이 개발 실행 (Windows PowerShell, 구분자는 ';')
$env:PYTHONPATH = "src;packages/smpl18/src"; python -m soma_synth.cli run --source <name> ...
```

실행 환경의 기준은 CPython 3.13.5, numpy 2.2.6, scipy 1.16.2 다(§6). 셸의 `python` 이 venv 가 아닌
인터프리터일 수 있으므로 venv 의 파이썬을 경로로 부르거나 venv 를 켠다. 러너는
`SOMA_DATA_ROOT` 가 있어야 한다(`--data-root` 로 줄 수도 있다). 원천은 `SOMA_SOURCE_ROOT`(기본
`<SOMA_DATA_ROOT>/extracted`), 바디 모델은 `SOMA_BODY_MODEL_DIR`(기본
`<SOMA_DATA_ROOT>/body_models/smpl`)에서 읽는다.

submodule 이 비어 있으면 `import smpl18` 이 실패한다 — 채우지 않은 것과 규칙이 갈라진 것을
구별하려면 그 실패가 드러나야 한다. 어느 커밋을 쓰는지는 상위 저장소의 gitlink 가 정하며,
그 값이 곧 재현의 기준이다(§5.2 의 모델 세트 해시와 같은 규율). 러너는 실제로 import 한
`smpl18` 의 리비전을 코드 리비전 옆에 기록한다: 체크아웃(`packages/smpl18/.git`)이 있으면 그
HEAD·dirty 여부·패치 해시, 설치본이면 버전과 패키지 파일별 SHA-256(§4.1).

pytest 는 `pyproject.toml` 의 `pythonpath` 로 같은 두 경로를 잡는다. 경로에 없으면
`import smpl18` 이 실패하며, 이는 조용한 대체 경로 없이 드러나는 편이 낫다 — 모델 선택 규칙이
두 벌로 갈라졌던 것이 애초의 문제였다.

submodule 을 갱신할 때는 그 저장소에서 커밋·push 한 뒤 상위에서 gitlink 를 옮긴다:

```bash
cd packages/smpl18 && git commit -am "..." && git push && cd ../..
git add packages/smpl18 && git commit -m "bump the smpl18 pin"
```

### 2.2 18관절 축소 — `smpl18` 이 정하고, 피험자마다 한 번 적합한다 (2026-09-16)

`large_reference.npz` 의 18관절은 SMPL-24 에서 `spine1`·`spine2`·양 `collar` 를 피험자 상수로
고정하고 손 관절 둘을 버린 것이다. 무엇을 고정하는지, 그 아래 관절(`spine3`, 양 어깨)이 제거된
회전을 어떻게 흡수하는지, 네 상수를 어떻게 적합하는지는 모두 `smpl18.reduce` 가 정한다. 이
저장소는 2026-09-16 까지 같은 계산을 `scripts/poc/anthro_smpl.py` 에 따로 갖고 있었고, 다섯
소스 중 네 곳(hknu·addbiomechanics·gaitex·amass)은 상수를 **take 마다** 적합했다.

이제는 `soma_synth.pipeline.reduced_model` 하나를 거친다. 이 모듈이 하는 일은 패키지가
호출자에게 남긴 두 가지뿐이다.

| 무엇 | 어떻게 |
|---|---|
| 어떤 프레임을 모으나 | 한 피험자의 **모든 take** 를 이어 붙인 것처럼 고르게 뽑는다(긴 take 는 길이만큼 반영). 서로 다른 rest skeleton 에 올라간 take 는 섞지 않고 묶음을 나눈다 |
| 숫자는 어디서 오나 | 레지스트리의 `smpl18_profile` 이 가리키는 프로파일의 `reduce` 절(`sample_frames`·`optimiser`·`max_evaluations`). 코드에 기본값이 없다 |

소스별로 모으는 단위는 다음과 같다.

| source | 적합 묶음 | 적합에 쓰는 프레임 |
|---|---|---|
| `hknu` | 피험자 | 코퍼스의 모든 trial, 원 표본율, 뼈 길이로 보정된 skeleton |
| `addbiomechanics` | 피험자 | 읽히는 모든 trial, 원 표본율 |
| `gaitex` | 피험자 | 모든 take 의 pair window 행 — 번들이 싣는 바로 그 행 |
| `amass` | 피험자 × 성별 모델 × MoSh betas | 묶음의 모든 motion 시퀀스, 원 표본율. 샤드 여럿이 동시에 떠도 적합은 한 번이며(잠금 파일), 나머지 샤드는 그 결과 파일을 읽는다 |
| `prism` | 피험자 | 모든 take 의 전체 프레임 |

기록은 두 곳에 남는다. 번들 루트의 `reduced_model_fit.json` 은 묶음마다 네 상수(`constants_wxyz`),
흡수 관절, 적합 전후 잔차(RMS·최대, m), 관절별 RMS, 쓴 프레임 수, 수렴 여부, 설정과 프로파일
해시를 담고, 각 take 가 어느 묶음의 상수를 썼는지 적는다. 모든 번들이 이 파일을 가져야 한다
(`dataset_profiles_v1.yaml` `universal.bundle_files`). 각 take 의 `manifest.json` 에는 같은 기록이
`anthro_reconstruction.reduced_model_fit` 으로 들어가고, 그 take 자체의 잔차는 전과 같이
`validation.reduced_model_fit_residual_m` 에 있다. 묶음 없이 한 take 만으로 적합한 경우(생성기를
단독으로 부른 경우)는 기록의 `scope` 가 `take` 로 남는다.

### 2.3 원천 준비 — 목록과 반입 (`hash-sources`, `stage-sources`)

원천 폴더 다섯(`amass`·`prism`·`gaitex`·`hknu_fullbody`·`addbiomechanics`)은 복사로 PC 에 온다. 다섯
폴더를 담은 폴더(예: 읽기 전용 공유 사본)에서 가져온다. 명령 둘이 그 일을 한다
(`pipeline/source_files.py`).

```bash
# 목록: 원천 폴더의 모든 파일, GNU coreutils 형식("<sha256>  <폴더>/<상대 posix 경로>", 이름 순, 한 줄에 하나)
soma-synth hash-sources [--source-root <root>] [--sources hknu,prism] --out <SHA256SUMS> [--force]

# 반입: 목록의 파일 중 없는 것만 복사한다. --verify-only 는 복사 없이 대조만 한다
soma-synth stage-sources --from <원천 폴더들을 담은 폴더> [--to <source root>] [--sources ...] --sums <SHA256SUMS> [--verify-only]
```

- `hash-sources` 는 원천을 **읽기만** 한다. `--source-root` 의 기본은 `SOMA_SOURCE_ROOT`, 없으면
  `<SOMA_DATA_ROOT>/extracted`. `--sources` 는 폴더 이름(`hknu` 는 `hknu_fullbody`), 기본은 다섯 모두.
  목록은 원천 루트 안, `SOMA_DATA_ROOT` 의 `extracted/`·`raw_archives/` 안, 증거 폴더(`_superseded`·`_runs`·
  `_manifest_backfill`, 어디에 있든 이름으로) 안, lineage 컨테이너 `runs/experimental_generation_poc_demo`
  안(`SOMA_DATA_ROOT` 의 것, 그리고 폴더 이름이 그렇게 이어지는 어느 경로든), 번들이나 코퍼스 안(위쪽
  폴더 어딘가에 `INDEX.json` 이 있으면), data root 자신의 파일(`README.md`, 그리고 있으면 `MASTER.md`·
  `state/local_archive_inventory.json`)에는 쓰지 않는다(종료 코드 3). 동기화 폴더는 경고만 한다. 이미 있는 파일은 이
  명령이 쓴 목록으로 읽힐 때만(모든 줄이 원천 폴더의 파일을 적은 `sha256sum` 줄이고, 이름이 겹치지 않고
  정렬돼 있고, 줄바꿈으로 끝난다) 새 목록으로 바꾼다. 그렇지 않은 파일은 `--force` 가 있어야 덮는다
  (`--force` 도 위치 거부는 풀지 않는다). `sha256sum -c` 로도 확인할 수 있다.
- `stage-sources` 는 목록에 있고 `--to`(기본 `SOMA_SOURCE_ROOT`, 없으면 `<SOMA_DATA_ROOT>/extracted`)에
  **없는** 파일만 `--from` 에서 복사한다. 복사본은 목적지 폴더의 임시 이름(`.<이름>.*.staging`)에 쓰고,
  목록의 SHA-256 과 맞을 때만 제 이름으로 바꾼다(덮어쓰지 않는 rename). 이미 있고 해시가 맞는 파일은
  건너뛴다. 이미 있는데 해시가 **다른** 파일이 하나라도 있으면 모두 보고하고 **아무것도 복사하지 않고**
  종료 코드 3 으로 끝난다. 이 명령은 아무것도 지우거나 덮어쓰지 않는다. `--from` 에 없거나 `--from`
  의 파일이 목록과 다르면(동기화가 덜 된 드라이브) 그 파일은 쓰지 않고 보고하며 종료 코드 1 이다.
  `--verify-only` 는 복사 없이 없는 것·다른 것을 보고한다(있으면 종료 코드 1).
- 중단된 반입이 남긴 임시 이름의 복사본(`.<이름>.<무작위>.staging`)은 원천 파일이 아니다. 목록과 원천
  추적의 열거(`iter_source_files`)가 빼고, `stage-sources` 는 `--verify-only` 를 포함해 매번 목적지의 원천
  폴더(`--sources` 로 고른 것) 아래 남은 것을 `LEFTOVER` 로 보고한다. 지우지는 않는다 — 소유자가 손으로
  지운다. 종료 코드에는 영향이 없다.
- `--from` 은 동기화 폴더나 공유 드라이브여도 된다(읽기만 한다). `--to` 는 원천 루트가 받는 가드를
  받는다(동기화 폴더·공유 드라이브는 경고만 한다): 증거 폴더(`_superseded`·`_runs`·`_manifest_backfill`)
  안이 아니고, data root 자체나 그 `runs`·`raw_archives`·바디 모델 폴더 안이 아니며, `--from` 과 겹치지
  않는다. 반입은 원천 폴더에 **없던 파일을 더하는 일**뿐이다. 이미 있는 원천은 보존 규칙 1.4 대로
  바꾸거나 지우지 않는다.
- 운영체제가 남기는 파일(`.DS_Store`·`Thumbs.db`·`desktop.ini`·`._*`)은 원천 파일이 아니어서 목록에
  넣지 않는다. 실행 기록의 원천 추적(§4.1)도 같은 열거를 쓴다.

---

## 3. run identity

### 3.1 `run_id` 는 그대로 두고 `run_identity` 를 더한다

현재 `run_id` 는 take 라벨이며 manifest 필수 필드다. 의미를 바꾸면 기존 5개 번들의 manifest 가
달라진다. 따라서 **`run_id` 는 건드리지 않고**, run 단위 식별자를 새 이름으로 더한다.

```
run_identity = sha256(
    canonical_json({
        "spec_id":            <spec_id>,
        "spec_version":       <spec_version>,
        "source_name":        <source>,
        "entrypoint":         <script path>,
        "resolved_config":    <resolved_config.yaml 의 sha256>,
        "code_revision":      <git HEAD sha + dirty 여부>,
        "input_corpus":       <입력 코퍼스 경로와 그 SUMMARY/INDEX sha256>,
        "body_model_sha256":  <SMPL 모델 파일 sha256>,
    })
)[:32]
```

§10.1 이 요구하는 "config/code/source identity 를 포함한 hash" 를 그대로 만족한다. 같은 입력은
같은 `run_identity` 를 준다.

`resolved_config` 항목(`resolved_config_sha256`)은 레지스트리 v2(2026-09-25) 전까지 채워지지 않았다. 그래서
출력 폴더나 표본 선택, 단계만 다른 두 실행이 같은 identity 를 받았다. 이제 `resolved_config.yaml` 의 `config`
블록을 canonical JSON 으로 해시해 채운다. data root 와, data root 기준
상대 경로가 있는 `dataset_dir` 의 절대 경로는 해시에서 뺀다 — 다른 PC 에서 같은 실행은 같은 identity 를
받는다. 완전히 같은 실행을 되풀이하면 identity 도 같으므로, 러너는 첫 실행 폴더를 덮지 않고
`run_<identity>-2` 에 기록한다.

> **남는 불일치.** §10.1 의 문구는 `run_id` 자체가 그 해시이기를 요구한다. 이 명세는 호환을
> 위해 새 필드를 쓰므로 문구와 완전히 일치하지 않는다. 봉인 문서를 고칠 수 없으므로 이 차이를
> 여기 적어 둔다.

---

## 4. run 디렉터리

`runs/<run_identity>/` 에 §10.1 의 7종을 남긴다.

| 파일 | 내용 | 쓰는 시점 |
|---|---|---|
| `resolved_config.yaml` | 해석된 전체 설정(기본값 포함) | run 시작 |
| `manifest.json` | run 수준 manifest — `run_identity`, 소스, 단계 목록, 결과 | `close_run` |
| `environment.lock` | Python·의존성 버전, 플랫폼 | run 시작 |
| `source_assets.json` | 입력 자산 경로와 SHA-256 | `register_asset` 누적 |
| `metrics.json` | take 수, 건너뛴 수, 소요 시간, 단계별 집계 | `close_run` |
| `exclusions.json` | 제외된 take 와 사유 | `close_run` |
| `logs/` | 실행 로그: `runner.log`(러너가 출력한 것), 자식 단계마다 표준 출력과 표준 오류 — `corpus.log`(코퍼스 진입점), `generate.log`(번들 진입점과 병합; 샤드마다 `generate.shard<i>of<N>.log`), `post_step.<스크립트>.log`(후처리). 명령 줄(`$ ...`)로 시작해 종료 코드(`[exit N]`)로 끝나고, 콘솔에도 그대로 나온다(`pipeline.generate` 가 tee 한다). 거부되면 `refusal.txt`, 예기치 않은 실패면 `error.txt` | 전 구간 |

take 디렉터리의 `manifest.json`(계약 §12.2)과 run 디렉터리의 `manifest.json`(§10.1)은 **다른
것**이다. 전자는 take 의 필드 계약이고 후자는 실행의 기록이다.

### 4.1 추적 기록 — 라이선스 철회에 대비한다 (ADR-0041, 2026-09-25)

팀원이 만든 번들은 그 PC 에만 있고 중앙 카탈로그에 없다. 원천의 라이선스가 철회되면 실행 기록을
모아 영향받는 산출물을 찾는다(`PIPELINE_GOVERNANCE.md` §12). 그래서 번들을 만든 실행은 일곱 가지
외에 다음을 남긴다.

| 무엇 | 어디 |
|---|---|
| 원천 참조 목록 — take manifest 가 적은 모든 `relative_path`(또는 `source_asset_id`)와 그 옆의 해시(`sha256`·`source_asset_sha256`·`source_asset_sha256_take`), 참조별 take·필드, 개수, 합산 해시 | `source_manifest.json` |
| 원천 파일 전체 해시(코퍼스가 있는 소스) — 이번 실행의 코퍼스 단계·번들 단계가 읽을 수 있었던 모든 원천 파일의 논리 id(`extracted/<폴더>/<경로>`), 크기, **파일 전체**의 SHA-256, 그 파일을 읽을 수 있었던 단계, 단계별 선택 인자, 개수·총 크기·합산 해시·해시에 걸린 시간 | `source_manifest.json` `source_files`, 요약은 `manifest.json` `tracing.source_manifest.source_files` |
| 코퍼스 — lineage, 이번 실행이 만들었는지(`built`·`rebuilt`·`reused`), 코퍼스 인자를 썼는지(`corpus_args_used`), 지문 파일과 그 SHA-256, 열 때 폴더가 담고 있던 것(`found_at_open`: 없음이면 `null`, 끝난 코퍼스, 끝나지 않은 생성, 그 밖의 파일), 그것을 옆으로 옮겼는지(`moved_aside`) | `manifest.json` `corpus`, `source_assets.json` |
| `smpl18` 리비전 — 체크아웃의 HEAD·dirty·패치 해시, 또는 버전과 파일별 해시 | `environment.lock`, `manifest.json` `provenance.smpl18_revision` |
| 생성 근거 — 기준(`standing_decision`·`owner_record`), ADR, 결정 기록 경로와 그 SHA-256, 등급. `--authorized-by` 기록이 허가했으면 그 기록의 SHA-256(`authorization_record_sha256`)과 실행 기록 안의 사본(`authorization/`) | `manifest.json` `generation_decision` |
| 의도한 생성 근거 — 결정을 묻기 전에 적는다(`generation_basis_intended`; generate 단계가 없거나 `--skip-generate` 면 `null`). 건넨 `--authorized-by` 기록의 SHA-256(`authorized_by_sha256`) | `resolved_config.yaml` |
| 위 셋의 요약(원천 목록 개수와 합산 해시, 코드·`smpl18` 리비전, 결정 기록 경로와 SHA-256, 실제 근거 `generation_basis` 와 의도한 근거 `generation_basis_intended`). 거부된 실행은 실제 근거가 `null` 이거나 거부 사유(`artifact_class`·`gates`)다 | `manifest.json` `tracing` |
| 원천 루트 — 생성기가 읽는 곳(`SOMA_SOURCE_ROOT`, 없으면 `<data root>/extracted`)과 그 출처(`source_root_from`) | `resolved_config.yaml` `source_root_relative`·`source_root` |
| 번들 폴더 — 생성 전에 `INDEX.json` 이 있었는지(`index_present`), 열 때 폴더가 담고 있던 것(`found_at_open`: 없음이면 `null`, 끝난 번들, 끝나지 않은 생성과 그 표지를 남긴 실행, 그 밖의 파일), 그것을 옆으로 옮겼는지(`moved_aside`), 생성 단계를 끝내고 `.generating` 을 지웠는지(`generating_marker_removed`) | `manifest.json` `bundle` |
| 카탈로그 위치, 그리고 기록된 경로 각각의 data root 기준 상대 경로. 레지스트리 v2 와 함께 더한 경로(카탈로그, `_runs`, 바디 모델 폴더, 파일 단위 모델 재지정, 코퍼스 폴더, `source_manifest.json` 의 번들)는 data root 안이면 **상대 경로만** 적고, 밖일 때만 절대 경로를 적는다. 팀원의 실행 기록은 모아서 읽히므로 그 PC 의 절대 경로를 늘리지 않는다. `dataset_dir`·`data_root` 는 전처럼 절대 경로도 남는다 | `resolved_config.yaml`, `source_assets.json`, `provenance.body_model` |
| 교체한 것 — 옆으로 옮긴 번들·코퍼스 폴더마다 원래 위치, 옮긴 곳(`aside_relative`·`aside_name`), 담고 있던 것(`found`), 운영 폴더인지(`production_directory`), 작은 증거 파일의 사본 위치와 SHA-256(`evidence`), 그리고 그 결말(`outcome`: 성공하면 `deleted`·`kept_production_directory`, 실패하면 `restored` 와 실패한 산출의 `.failed-*` 위치) | `replaced/`, `manifest.json` `replaced` |

코퍼스에서 만든 번들(`hknu`·`gaitex`·`addbiomechanics`)의 manifest 는 원천으로 **코퍼스 파일**을
적는다. 그 코퍼스가 어느 원천 파일에서 왔는지는 코퍼스 자신의 기록에 있고, 실행 기록은 코퍼스의
지문을 함께 남긴다. `gaitex` 는 small 을 합성한 원천 마커·IMU 파일(`small_synthesis.inputs`)도 적으므로
그것도 목록에 들어간다.

**코퍼스 너머의 원천 해시는 실행 기록에 있다.** 코퍼스 자신의 기록은 원천을 이렇게만 적는다.

| 코퍼스 | 원천을 적는 방식 |
|---|---|
| `hknu_smpl24_paired` (`generate_hknu_faithful.py`) | `SUMMARY.json` 과 코퍼스 npz 가 피험자·trial 이름만 적는다. 원천 파일의 해시는 없다 |
| `gaitex_smpl24`·`addbio_smpl24_raw` | 각 artifact 의 `adapter_hash` 가 원천 파일 **첫 1 MiB** 의 SHA-256 앞 16자리다. 파일 전체의 해시가 아니다 |

이 기록은 npz 멤버·manifest·`pair_id` 로 이어지므로 바꾸지 않는다. 대신 러너가 실행 기록의
`source_manifest.json` 에 `source_files` 블록을 더한다(`pipeline/source_trace.py`). 파일 전체의 해시는
거기에 있다.

- **어느 파일인가.** 실행의 원천 루트(`SOMA_SOURCE_ROOT`, 없으면 `<data root>/extracted`) 아래 그 소스의
  폴더에서, 이번 실행의 단계가 읽을 수 있었던 파일 전부다. 선택 인자가 있으면 선택된 피험자로 좁힌다.
  어느 피험자의 것도 아닌 파일(워크북, README, `_provenance`)은 늘 들어간다.

  | source | 코퍼스 단계(이번 실행이 코퍼스를 만들 때만) | 번들 단계 |
  |---|---|---|
  | `hknu` | `hknu_fullbody`, `--subjects` 로 좁힘(`Dataset_Processed/<S>`·`Dataset_Raw/C3D/<S>`) | `hknu_fullbody`(워크북을 읽는다), 번들의 `--subjects` 로 좁힘 |
  | `gaitex` | `gaitex`, `--subjects` 로 좁힘(`<subject>/…`) | `gaitex`(take 의 마커·IMU 파일을 읽는다), 번들의 `--subjects` 로 좁힘 |
  | `addbiomechanics` | `addbiomechanics`, `--only <study>/<subject>` 로 좁히고 `--per-study N` 이면 생성기와 같은 순서·같은 규칙으로 study 마다 비어 있지 않은 `.b3d` 첫 N 개(생성기는 파일을 먼저 담고 개수를 비교하므로 N 이 0 이하여도 첫 1 개) | 원천을 읽지 않는다(코퍼스만 읽는다) |

  재사용한 코퍼스의 원천은 그것을 만든 실행의 기록에 있고, 이번 기록에는 그 코퍼스의 지문이 있다.
  `amass`·`prism` 은 코퍼스가 없고 take manifest 가 원천 파일을 파일 전체의 SHA-256 과 함께 적으므로
  `source_files` 는 `status: not_recorded` 와 그 이유를 적는다. 명시적 생성 명령(`generate_cmd`)은
  무엇을 읽었는지 알 수 없어 역시 `not_recorded` 다.
- **무엇을 적나.** 파일마다 논리 id `extracted/<폴더>/<상대 경로>`, 크기, SHA-256(1 MiB 씩 읽어 파일
  전체), 그 파일을 읽을 수 있었던 단계. 블록에는 단계별 선택 인자와 좁혔는지(`narrowed`), 원천 루트의
  data root 기준 위치(`source_root_relative`, 밖이면 절대 경로 `source_root`), 개수·총 바이트·합산 해시
  (`<relative_path>\t<sha256>` 줄을 정렬해 이은 것의 SHA-256)·해시에 걸린 초가 있다. 읽지 못한 파일은
  `unreadable` 에 이유와 함께 남고, 원천 폴더가 없으면 `status: unavailable` 이다. 파일은 한 번만 해시한다.
  열거는 `hash-sources` 와 같다(§2.3): 같은 파일이 같은 순서로, 같은 해시로 나온다.
- **언제 해시하나.** 생성 단계 끝, `.generating` 을 지우기 전이다. **코퍼스 단계의 원천은 그 코퍼스를
  만드는 실행에서만 해시한다.** 코퍼스를 재사용하는 실행은 그 원천을 다시 해시하지 않는다: 그 코퍼스를
  만든 실행이 이미 추적했고(그 실행 기록의 `source_files`), 이번 기록에는 재사용한 코퍼스의 지문이 있다.
  번들 단계가 원천을 읽는 `hknu`(워크북)·`gaitex`(take 의 마커·IMU 파일)는 번들만 다시 만드는 실행도
  번들 단계가 읽을 수 있었던 원천을, 선택한 피험자로 좁혀 해시한다.
- **비용.** 해시 시간은 원천 크기에 비례한다. 여기서 GB 는 10^9 바이트다(`hash-sources` 가 적는 단위).
  `hknu_fullbody` 는 843 파일 27.88 GB 이고, `hknu` 코퍼스나 번들을 전체로 만드는 실행은 그만큼을 더 읽는다.
  `gaitex` 는 17.7 GB 다. `addbiomechanics` 의 `.b3d` 1,137 개, 약 568 GB(529 GiB)는 **AddBio 코퍼스를 전체로
  새로 만드는 실행에서만** 해시된다(그 실행 자체가 몇 시간이므로 받아들인다). `--only`·`--per-study` 표본
  실행은 고른 `.b3d` 만 해시하고, 코퍼스를 재사용해 번들만 만드는
  AddBio 실행은 원천을 해시하지 않는다(AddBio 번들 단계는 코퍼스만 읽는다).

카탈로그의 기본값은 `<data root>/experimental/catalog` 다. 이전 기본값 `<번들의 부모>/catalog` 는
lineage 컨테이너 안이라 카탈로그가 아니었다. 자산의 `relative_path` 는 **데이터 루트 기준**이다: `run` 은
러너의 데이터 루트, `pipeline`·`register` 는 `--data-root`, 없으면 `SOMA_DATA_ROOT` 를 쓰고, 둘 다 없으면
종료 코드 2 로 멈춘다. 데이터 루트 안에 있지 않은 번들도 종료 코드 2 로 거부한다. `<번들의 부모>` 를
기준으로 하면 경로가 `gaitex_unified8/...` 로 나와 `run` 이 등록한 항목
(`runs/experimental_generation_poc_demo/gaitex_unified8/...`)과 맞지 않으므로 기준으로 쓰지 않는다.

카탈로그 파일(`asset_catalog.json`, `<source>/source_manifest.json`)은 이렇게 쓴다(`datasets/catalog.py`).

- **잠금.** 쓰는 쪽은 읽고-고치고-쓰는 동안 카탈로그 폴더의 `.catalog.lock` 을 잡는다. 잠금은 그 파일의
  **존재가 아니라** 운영체제 잠금이다(Windows 는 첫 바이트의 `msvcrt.locking`, 그 밖은 `fcntl.flock`). 파일이
  없으면 만들고, 있으면(다른 writer 가 만든 빈 잠금 파일이 이미 있을 수 있다) 그대로 쓴다.
  어느 쪽이든 **지우지 않는다**: 남의 잠금일 수 있고, 누군가 막 잡으려는 잠금 파일을 지우면 두 writer 가
  함께 들어온다. 잠금은 그 프로세스가 끝나면 풀린다. 다른 writer 가 잡고 있으면 기다리다가(기본 900 초)
  `CatalogLocked` 로 멈춘다. 자산 해시는 잠금 밖에서 먼저 계산한다.
- **원자적 교체.** 새 내용은 같은 폴더의 임시 파일(`.<이름>.*.tmp`)에 쓰고 flush·fsync 한 뒤 `os.replace` 로
  바꾼다. 읽는 쪽은 옛 파일이나 새 파일을 보고, 반쯤 쓰인 파일은 보지 않는다. 실패하면 파일은 그대로고
  임시 파일은 지운다(프로세스가 죽으면 임시 파일이 남을 수 있지만 카탈로그 파일은 멀쩡하다).
- **변경 기록(journal).** 카탈로그 파일마다 옆에 `_history/<stem>.journal/` 이 있고, 쓸 때마다 그 폴더에
  작은 gzip JSON 파일 하나 `<순번 8자리>.<UTC 시각>Z.json.gz` 가 더해진다(임시 이름으로 쓰고 fsync 한 뒤
  아무도 쓰지 않은 이름으로만 링크한다: 덮어쓰기 없음). 기록 하나에는 쓴 시각(`utc`), 쓴 주체
  (`registered_by`: 도구 이름, 러너가 등록하면 그 실행의 `run_<identity>`), 작업(`operation`), 쓰기 전후 파일의
  SHA-256·크기(`before`·`after`), 그리고 **바뀐 항목만** 담는다 — 추가·교체·삭제된 항목마다 키(JSON Pointer:
  자산은 `/assets/<index>`, 그 밖의 필드는 `/<field>`), 옛 항목(추가면 null), 새 항목(삭제면 null). 카탈로그
  전체 사본은 남기지 않는다: 자산 몇 개를 등록하면 기록은 수 KB 이고, 카탈로그가 수백 MB 여도 같다(자산은
  위치별로 비교한다. 이 도구의 writer 는 자산을 뒤에 붙이거나 제자리에서 고치기만 한다). `run`·`register`·
  `pipeline` 이 번들을 처음 등록하면 기록이 번들의 자산 수를 따른다: gzip 으로 자산 하나에 약 55 바이트, `gaitex`
  360 자산이 약 20 KB, `addbiomechanics` 202,300 자산이 약 11 MB 다. 한 버전 전체의 `supersede_spec_version` 은 옛 항목과 새 항목을 함께 담아 그 두 배쯤이다. 다시 등록은 바뀐
  자산만큼이고(아래), 바뀐 것이 없으면 기록이 없다. 이 도구는 기록을
  고치거나 지우지 않는다. 현재 파일과 기록으로 이전 항목이 모두 복원된다: `catalog.catalog_as_of(path, n)`
  은 n 번 기록이 적용된 직후의 파일을 현재 파일에서 그 뒤 기록들을 거꾸로 되돌려 만들고, 바이트가 기록의
  SHA-256 과 같은지 알려 준다.
- **순서와 죽은 쓰기.** 새 내용을 임시 파일에 fsync → 기록을 fsync·링크 → `os.replace`. 기록 전에 죽으면
  아무것도 바뀌지 않는다. 기록과 교체 사이에 죽거나 교체가 실패하면 파일은 그대로고, 파일이 가진 적 없는
  `after` 를 적은 기록이 남는다. 그 기록의 `before` 는 현재 파일의 해시이고 다음 쓰기의 `before` 도 같으므로,
  `catalog.journal_story(path)` 가 적용되지 않은 기록으로 가려낸다(기록 어느 것으로도 설명되지 않는 변경은
  "기록 밖에서 바뀜"으로 표시한다). 반대 순서(교체 먼저)는 교체된 항목의 옛 값이 기록되기 전에 죽으면
  영영 잃는다. 이 순서는 아무것도 잃지 않는다. 기록이 없던 때(이 도구 이전)의 카탈로그는 첫 기록의
  `before` 가 출발점이다.
- **다시 등록하면 맞춘다.** 제자리에서 다시 만든 번들은 자산 id 와 경로가 그대로고 바이트만
  바뀐다. 등록이 뒤에 붙이기만 하면 id 마다 살아 있는 항목이 둘 남고, 옛 항목의 SHA-256 은 파일과
  맞지 않는다.
  이제 등록은 그 데이터셋의 이전 항목과 맞춘다(`catalog.merge_registration`). 그 데이터셋의 항목이란 살아 있는
  (`superseded` 가 아닌) 항목 가운데 같은 번들 폴더 아래의 같은 자산 id, 또는 같은 lineage(source·spec_id·
  spec_version)의 같은 상대 경로를 가진 것이다. 최신 것부터 보아, 지금 등록하는 자산과 똑같은(`catalog.ASSET_FIELDS`
  여덟 필드가 같다; 다른 writer 가 곁에 더한 `registration_evidence`·`metadata` 는 보지 않는다) 첫 항목은 그대로
  두고 그 자산은 다시 붙이지 않는다. 나머지는 모두 제자리에서 기존 카탈로그가 이미 쓰는 규약대로 표시한다:
  `superseded: true`, `supersession_reason: "registration_replaced"`, `superseded_by` 새 항목의 자산 id,
  `superseded_by_sha256` 새 항목의 SHA-256. 항목은 지우거나 옮기지 않는다. 같은 번들 폴더 아래 같은
  lineage 의 살아 있는 항목 가운데 지금 등록하는 자산 어느 것도 대신하지 않는 것 — 새 INDEX 에서 더는 `ok`
  가 아닌 take, 없어진 take 폴더나 역할 파일의 항목 — 도 같은 쓰기에서 표시한다: `superseded: true`,
  `supersession_reason: "registration_dropped"`, `superseded_by: null`. 제자리 교체 뒤에는 그 파일이 없으므로,
  살아 있게 두면 없는 파일을 가리킨다. 번들 폴더가 데이터 루트 자체이면 이 규칙은 쓰지 않는다. 표시한
  항목은 그 쓰기의 기록에 `replaced` 변경(옛 항목 전체와 새 항목)으로, 붙인 항목은 `added` 로 들어가고,
  기록의 `registered_by` 가 그 실행을 적는다. `operation` 에 `added`·`unchanged`·`superseded`·`dropped`
  개수가 있다. 아무것도 바꾸지 않을 등록은 쓰지
  않는다(기록도 없고 파일도 그대로다; `<source>/source_manifest.json` 도 같다). 그래서 뒤에 붙이기만 하던
  때 등록된 카탈로그는 같은 번들을 다시 등록하면 **기록 하나**로 정리된다: 옛 항목이 표시되고, 살아 있는 id
  중복이 없어지며, 살아 있는 항목의 SHA-256 이 모두 파일과 맞고, 한 번 더 등록하면 쓰지 않는다. 그 정리
  명령은 둘 중 하나다. 실행 기록을 남기고 기록의 `registered_by` 에 그 실행 id 가 들어가는 쪽이 앞의 것이다.

  ```bash
  soma-synth run --source gaitex --start-at register --stop-after register
  soma-synth register "$SOMA_DATA_ROOT/runs/experimental_generation_poc_demo/gaitex_unified8" \
      --catalog "$SOMA_DATA_ROOT/experimental/catalog" --source gaitex --data-root "$SOMA_DATA_ROOT"
  ```

  어느 쪽이든 `SOMA_DATA_ROOT` 가 그 번들을 담은 데이터 루트여야
  하고, 출력의 `... earlier entries superseded` 가 대체한 항목 수, `... dropped` 가 번들에 더는 없어 표시한
  항목 수다. 자산 id 는 폴더를 적지 않으므로 다른 폴더(scratch 표본)의 같은
  take 는 다른 데이터셋이다: 서로의 항목을 대체하지 않는다. 같은 폴더를 다른 spec_version 으로 등록한 항목도
  대체하지 않는다(그것은 `supersede_spec_version` 의 일이다).

---

## 5. provenance 10항목

| # | §10.2 항목 | 출처 | 현재 |
|---|---|---|---|
| 1 | 코드 리비전 + dirty 해시 | `git rev-parse HEAD` + `git status --porcelain` 해시 | **없음** |
| 2 | 계약·schema·config 버전 | `spec_id`/`spec_version`/`canonicalization_config` | 있음 |
| 3 | source asset SHA-256·라이선스 | `SourceIdentity.asset_sha256` + gate 설정 | 있음 |
| 4 | spec snapshot 해시 | `spec_documents` | 있음 |
| 5 | Python·의존성 버전 | `importlib.metadata` | prism 만 |
| 6 | model/checkpoint 해시 | 바디 모델 **세트**의 파일별 SHA-256 — §5.2 | 러너가 기록(2026-09-16) |
| 7 | seed | — **§5.1 참조** | 해당 없음 |
| 8 | 호스트 환경(비민감) | OS·아키텍처·CPU 수 | **없음** |
| 9 | 입출력 artifact 해시 | `emit_take` 가 쓰면서 누적 | **없음** |
| 10 | pass/warn/reject·quarantine | `quality_gate`·`validation`·`exclusions.json` | 있음 |

**credential·token·cookie·개인식별정보는 기록하지 않는다**(§10.2).

### 5.1 seed — 채울 것이 없다는 것을 기록한다

등록된 paired 생성 경로에는 난수가 **0건**이다. 검사는 `scripts/poc/generate_*.py`,
공통 emitter와 `src/soma_synth/`를 대상으로 하되, 별도 연구용인
`src/soma_synth/experimental/`만 제외한다. 이 연구 코드의 seed를 paired
생성의 seed로 혼동하지 않는다. 생성 경로가 이 namespace를 import하면 경계 검사에서
실패한다. 상대 import와 별칭도 검사하며, 파일 읽기·구문 분석 실패는 통과시키지 않는다.
이는 정적 소스 검사이지 임의의 동적 실행에 대한 증명은 아니다. 계산된 동적 import
대상 등은 별도 검토가 필요하다. 실제 생성 코드의 난수 금지 기준은 그대로 유지한다.

따라서 seed 를 발명하지 않고 부재를 명시한다.

```json
"seeds": {"present": false, "reason": "no stochastic step in the generation path"}
```

그리고 **그 주장을 테스트로 지킨다**. 빈칸으로 두면 "없음"과 "안 적음"을 구별할 수 없고,
나중에 확률적 단계가 들어와도 아무도 모른다.

### 5.2 6번 — 파일 하나가 아니라 세트를 기록한다 (2026-09-16)

§10.2 6번은 "model/checkpoint 해시"를 요구하지만, 다섯 생성기는 **피험자마다** 성별에 맞는
모델을 고른다. 파일 하나를 지목하면 남녀가 섞인 번들을 잘못 서술하므로, 실행 기록에는 **세트
전체**가 들어간다: 디렉터리, 선택 규칙, 파일별 SHA-256, 그리고 그 해시들을 정렬해 한 번 더
해시한 `set_digest`. `set_digest` 는 run identity 의 `body_model_sha256` 이기도 해서, 모델이
하나라도 바뀌면 다른 run identity 가 된다.

두 가지를 각각 한 곳에서 가져온다.

| 무엇 | 어디서 |
|---|---|
| 세트가 있는 위치 | `SOMA_BODY_MODEL_DIR` 가 있으면 그 폴더, 없으면 [`source_pipelines_v2.yaml`](../../configs/datasets/source_pipelines_v2.yaml) `body_model_sets.<이름>.relative_path`, run 의 data root 기준 — 곧 `pipeline.paths.body_model_dir()` |
| 성별 → 파일 이름 | `smpl18.model.select` (`packages/smpl18`) — 생성기도 같은 함수로 고른다 |

코드에서는 `soma_synth.pipeline.body_models` 가 그 둘을 잇고,
`addbio_retarget/shape_fit.py` 와 `scripts/poc/anthro_smpl.py` 는 파일 이름 규칙을 스스로
정하지 않고 같은 함수에 위임한다. 세트를 찾지 못하면 해시를 지어내지 않고 `unavailable` 과
그 이유를 남긴다(§5.1 과 같은 규율).

러너는 생성기와 같은 함수(`paths.body_model_dir`)로 바디 모델 폴더를 정하고, 생성기에 자기 data root 를
`SOMA_DATA_ROOT` 로 넘기며, 정한 그 폴더를 해시한다. 기록에는 폴더와 그 출처
(`directory_from`: `SOMA_BODY_MODEL_DIR` 또는 `data_root`)가 남는다. `scripts/poc/anthro_smpl.py`
(amass·prism 이 거친다)가 받는 파일 단위 재지정 `SOMA_SMPL_MODEL_MALE`·`SOMA_SMPL_MODEL_FEMALE` 는
그 폴더를 우회하므로, 설정돼 있으면 `resolved_config.yaml` 의 `body_model_file_overrides` 에 경로와
SHA-256 이 남는다.

---

## 6. 결정론

- 파일 순회는 항상 정렬한다. 디렉터리 열거 순서에 의존하지 않는다.
- 병렬 실행이 결과를 바꾸지 않는다. take 는 서로 독립이며 공유 가변 상태가 없다.
- 부동소수 연산 순서를 병렬도에 따라 바꾸지 않는다.
- 프로세스 풀은 모든 플랫폼에서 `spawn` 으로 워커를 시작한다(`multiprocessing.get_context("spawn")`,
  `ProcessPoolExecutor(mp_context=...)`). Windows 는 `spawn` 뿐이고 Linux 의 기본은 `fork` 여서, 부모의
  모듈·`sys.path` 편집·열린 파일을 물려받는 워커가 PC 에 따라 다르게 돌 수 있기 때문이다. 지금 풀을 쓰는
  것은 `addbiomechanics` 의 `--jobs` 두 곳(`generate_addbio_unified8.py`, `generate_addbio_faithful.py`)이고,
  `tests/pipeline/test_process_start_method.py` 가 `src/`·`scripts/` 를 AST 로 훑어 `spawn` 없이 만든 풀을
  거부한다. 러너의 `--jobs` 샤드는 `subprocess` 로 생성기 프로세스를 띄우므로 해당하지 않는다.

### 6.1 바이트 동일성은 같은 PC, 같은 고정 환경에서만 주장한다 (ADR-0041)

- **같은 PC 와 같은 버전 고정 환경**(CPython 3.13.5, numpy 2.2.6, scipy 1.16.2, 같은 `smpl18`
  리비전)에서는 산출이 바이트 단위로 같아야 한다. 이것을 하네스(`scripts/harness/`,
  `run_harness.py` → `hash_outputs.py` → `compare_hashes.py`)로 확인한다. npz 는 멤버별 배열 해시로,
  JSON 은 파싱한 뒤 시각 필드를 가리고 비교한다. 기준은 같은 PC 에서 수정 전 코드로 만든 기준 해시 파일이다
  (이 저장소에 싣지 않고, `compare_hashes.py` 에 경로로 준다).
- **PC 가 다르면 허용 오차로 판정한다.** CPU, BLAS, numpy, zlib, 줄바꿈이 결과에 영향을 준다. 기계를
  건너 약 1 float32 ulp 차이가 관측된 바 있다. 팀원 번들은 그 PC 에만 두므로(§9) PC 사이의 바이트
  동일성은 요구하지 않는다.
- **선언된 차이.** `code_hash` 의 뜻은 바꾸지 않았다(ADR-0041 결정 4). 그래서 생성 코드 파일이
  바뀌면 코퍼스의 `content_hashes` 중 `code=` 토큰이 바뀌고, 그 토큰에서 나오는 `pair_id` 도 바뀐다.
  soma-synth 분리 과정에서 `hknu`·`gaitex`·`addbiomechanics` 의 `code=` 와 `pair_id` 가 달라지는
  것은 결함이 아니라 선언된 차이다(분리 때 실제로 달라진 것은 `gaitex`·`addbiomechanics` 뿐이고,
  `amass`·`prism`·`hknu` 는 바이트 단위로 같았다). 하네스 비교(`compare_hashes.py`)는 다른 토큰이
  `code` 하나일 때만 이것으로 분류한다. `model`·`adapter`·`config` 가 다르면 실패다.
- 레지스트리 v2 로의 전환은 어떤 생성기 코드도 바꾸지 않았다. 생성기가 레지스트리에서 읽는 것은
  `smpl18_profile` 뿐이고 그 값은 v1 과 같다. 바뀐 것은 러너가 부르는 방식이다: `prism` 은
  `--data-root <root>` 대신 `--out <번들>` 로(원천은 `SOMA_SOURCE_ROOT`), 코퍼스가 있는 세 소스는
  코퍼스를 선언된 플래그로 명시해 넘긴다.

---

## 7. 재개와 부분 실패

현재 다섯 생성기가 **네 가지 서로 다른 재개 판정**을 쓴다.

| 소스 | 판정 | 반쯤 쓰다 만 take 를 알아채나 |
|---|---|---|
| amass | 4파일 존재 + `manifest.json` 파싱 성공 | **예** |
| prism | core 파일 존재 | 아니오 |
| addbiomechanics | `manifest.json` 존재 | 아니오 |
| gaitex · hknu | 없음(항상 재생성) | 해당 없음 |

**표준은 가장 강한 것을 채택한다** — `bundle_opens`: 모든 산출 파일이 존재하고 `manifest.json`
이 파싱되어야 건너뛴다. 나머지는 `source_pipelines_v2.yaml` 에 현재 값 그대로 기록해 두고,
러너가 통합할 때 이 판정으로 올린다.

건너뛴 take 수는 `metrics.json` 에 남긴다. 조용한 건너뛰기는 재개가 아니라 누락이다.

`soma-synth run` 은 어느 생성기도 재개시키지 않는다(§1.3): 모든 번들과 코퍼스를 빈 폴더에 만들므로 이
판정들이 건너뛸 take 가 없다. 위 판정은 생성기를 직접 돌리거나 운영자가 `--bundle-arg=--resume` 류의 플래그를
넘길 때의 동작이다.

검증에도 건너뛰기가 하나 있다. validation ledger 는 PASS 한 take 를 그 내용 digest, `spec_version`, 생성 코드
digest, **검증기 digest**(`catalog.default_validator_digest`: spec 과 `validation/checks.py`·`npz_io.py`·
`report.py` 의 바이트)와 함께 기억하고, 넷이 모두 같을 때만 다시 보지 않는다(`takes_trusted`). 그래서 검증기
코드가 바뀌면 모든 ledger 의 신뢰가 한 번 풀리고, 다음 검증은 모든 take 를 다시 본다(`--full` 과 같다).

L4 governance 의 절대 경로 검사는 드라이브 문자와 POSIX 절대 경로를 잡는다: `/`, 폴더 이름, 다시 `/` 로
이어지는 경로가 값의 시작이나 공백·따옴표·`=`·`:`·괄호·구분자 뒤에 오면 뿌리와 상관없이 FAIL 이다
(`/Users/`·`/home/`·`/Volumes/`·`/tmp/` 뿐 아니라 `/data/`·`/opt/`·`/srv/`·`/scratch/`·`/nfs/`·`/gpfs/`·
`/workspace/` 도). 논리 id(`extracted/…`, `runs/…/tmp/…`), 단위(`m/s^2`), URL(`https://host/home/…`,
`file:///home/…`: 경로 앞 글자가 `/`)은 걸리지 않고, 아래에 경로가 없는 `/tmp` 하나도 걸리지 않는다.

적합성 판정(`checks.check_dataset_profile_conformance`)은 번들이 배포 전에 가져야 할 파일(`README.md`·
`VALIDATION_REPORT.json`·`validation_ledger.json`)이 없으면 WARN 을 붙인다. 러너의 validate 단계는 이 판정을
보고서·원장을 쓰기 전에, readme 단계보다 먼저 하므로, 호출한 쪽이 검증 뒤에 쓸 파일을 `pending_files` 로 넘기면
그 파일은 없다고 적지 않는다. 러너는 결과와 상관없이 쓰는 자기 보고서·원장을 넘기고, `README.md` WARN 은 계획에
readme 단계가 있고 **검증이 통과했을 때만** 보고서에서 뺀다(`checks.missing_distribution_file`). 검증이
실패하면 실행이 readme 단계 전에 멈춰(fail-fast) 번들에 `README.md` 가 없으므로 그 WARN 이 남는다.
`--stop-after validate` 면 `README.md` WARN 하나는 남는다: 그 실행은 그것을 쓰지 않는다. `soma-synth validate`
는 번들 폴더에 바로 쓰는 `--report`·`--ledger` 의 이름만 넘기고, 아무것도 쓰지 않는 검증은 셋 다 경고한다.
번들이 만들어야 할 파일(`KNOWN_LIMITATIONS.json` 등)은 넘겨도 없으면 FAIL 이다.

---

## 8. 소스별 차이

**이 절은 [`configs/datasets/source_pipelines_v2.yaml`](../../configs/datasets/source_pipelines_v2.yaml)
에서 생성한다. 손으로 고치지 않는다.**

```
soma-synth pipeline-doc --render
```

손으로 관리하는 표는 낡는다. 그래서 이 절은 생성한다.

<!-- BEGIN GENERATED: source pipelines -->

> 생성물이다. 원본은 [`source_pipelines_v2.yaml`](../../configs/datasets/source_pipelines_v2.yaml) 이고,
> `soma-synth pipeline-doc --render` 가 이 블록을 다시 쓴다. 손으로 고치지 말 것.

| source | `small_mode` | emitter | 재개 판정 | 병렬 | 진입점 | 번들 lineage | 코퍼스 | smpl18 프로파일 |
|---|---|---|---|---|---|---|---|---|
| `addbiomechanics` | `synthetic_from_smpl` | `unified8_emit` | `manifest_exists` | `process_pool` | `generate_addbio_unified8.py` | `addbio_unified8` | `addbio_smpl24_raw` | `addbiomechanics` |
| `amass` | `synthetic_from_smpl` | `inline` | `bundle_opens` | `shard` | `generate_amass_faithful_all.py` | `amass_faithful_full` | — | `amass` |
| `gaitex` | `synthetic_from_markers` | `unified8_emit` | `none` | `none` | `generate_gaitex_unified8.py` | `gaitex_unified8` | `gaitex_smpl24` | `gaitex` |
| `hknu` | `synthetic_from_smpl` | `unified8_emit` | `none` | `none` | `generate_hknu_unified8.py` | `hknu_unified8` | `hknu_smpl24_paired` | `configs/smpl18/profiles/hknu.yaml` |
| `prism` | `measured_physical` | `inline` | `core_files_exist` | `none` | `generate_prism_measured_all.py` | `prism_faithful_full` | — | `configs/smpl18/profiles/prism.yaml` |

**재개 판정의 뜻**

- `bundle_opens` — skips when every deliverable exists and manifest.json parses
- `core_files_exist` — skips when every core deliverable file exists, without opening any of them
- `manifest_exists` — skips when manifest.json exists, even if the npz beside it are truncated
- `none` — no skip logic; every take is rebuilt on every run

**공유 emitter 사용**: `addbiomechanics`, `gaitex`, `hknu` — **미사용(자체 구현)**: `amass`, `prism`

**생성 단계 (코퍼스 → 번들 → 후처리)**

- `addbiomechanics` — 코퍼스 `generate_addbio_smpl24.py --out <addbio_smpl24_raw>` (지문 `SUMMARY.json`, 번들에 `--raw` 로 전달) → 번들 `generate_addbio_unified8.py --out <addbio_unified8>`
- `amass` — 번들 `generate_amass_faithful_all.py --out-root <amass_faithful_full>`
- `gaitex` — 코퍼스 `generate_gaitex_smpl24.py --out <gaitex_smpl24>` (지문 `_run.json`, 번들에 `--retarget` 로 전달) → 번들 `generate_gaitex_unified8.py --out <gaitex_unified8>`
- `hknu` — 코퍼스 `generate_hknu_faithful.py --out <hknu_smpl24_paired>` (지문 `SUMMARY.json`, 번들에 `--paired` 로 전달) → 번들 `generate_hknu_unified8.py --out <hknu_unified8>`
- `prism` — 번들 `generate_prism_measured_all.py --out <prism_faithful_full>` → 후처리 `synthesize_insole_heading.py {dataset_dir}`

**읽는 환경 변수**

- `addbiomechanics` — `SOMA_DATA_ROOT`, `SOMA_SOURCE_ROOT`, `SOMA_BODY_MODEL_DIR`, `ADDBIO_RETARGET_CORPUS`
- `amass` — `SOMA_DATA_ROOT`, `SOMA_SOURCE_ROOT`, `SOMA_BODY_MODEL_DIR`
- `gaitex` — `SOMA_DATA_ROOT`, `SOMA_SOURCE_ROOT`, `SOMA_BODY_MODEL_DIR`
- `hknu` — `SOMA_DATA_ROOT`, `SOMA_SOURCE_ROOT`, `SOMA_BODY_MODEL_DIR`
- `prism` — `SOMA_DATA_ROOT`, `SOMA_SOURCE_ROOT`, `SOMA_BODY_MODEL_DIR`

**바디 모델 세트**

- `smpl_clean` — data root 기준 `body_models/smpl` (`SOMA_BODY_MODEL_DIR` 가 있으면 그 폴더, `pipeline.paths.body_model_dir()`), 성별에 따른 파일 선택은 `smpl18.model.select` 가 정한다. 사용: `addbiomechanics`, `amass`, `gaitex`, `hknu`, `prism`

**생성 정책 (`generation_policy`)**

- 근거: `standing_decision` — ADR-0041 (`docs/adr/ADR-0041-soma-synth-split-and-teammate-generation.md`), 결정 기록 `research/decisions/2026-09-25_source_distribution_and_pipeline_split_decision.md` §4
- 등급 `experimental_non_candidate`, 배포 범위 `internal_only`, `releases_holds: false`

<!-- END GENERATED: source pipelines -->

---

## 9. 이 파이프라인이 하지 않는 것

- **push 하지 않는다.** `src/soma_synth/cli.py` 의 `_next_step_guide` 가 명시하듯 soma-synth 는
  번들을 어디에도 올리지 않는다. 번들은 만든 PC 에 남는다. 설계이지 미완성이 아니다.
- **`experimental_non_candidate` 가 아닌 것은 만들지 않는다.** 생성 정책은 아래 §9.1 이다.
  canonical·internal-candidate 생성은 hold 아래 그대로 있고(상위 프로젝트의 hold 규칙), 이 파이프라인은
  그것을 열지 않는다.
- **생성이 끝나면 `KNOWN_LIMITATIONS.json` 을 렌더한다.** 검증기가 모든 번들에 요구하는
  파일이므로 생성기가 아니라 러너가 `configs/datasets/known_limitations_v1.yaml` 에서 쓴다.
  번들 이름의 블록이 없으면 파일을 쓰지 않고 `metrics.json` 에 `limitations_block_missing` 을
  남기며, 이어지는 validate 가 그 부재를 FAIL 로 보고한다.
- **봉인 문서를 고치지 않는다.**
- **임계값을 코드 상수로 두지 않는다.** 임계값은 버전이 붙은 설정 파일에 둔다.

### 9.1 생성 정책 — 결정 기록 하나로 상시 허용한다 (ADR-0041, 2026-09-25)

**상시 허용.** 내부 사용자가 자기 PC 에서 `soma-synth run` 으로 `experimental_non_candidate` · `internal_only` 번들을
만드는 것은 `ADR-0041`(상위 프로젝트: `docs/adr/ADR-0041-soma-synth-split-and-teammate-generation.md`) 이 기록한
결정(`2026-09-25_source_distribution_and_pipeline_split_decision.md`(상위 프로젝트: `research/decisions/2026-09-25_source_distribution_and_pipeline_split_decision.md`)
§4) 하나로 허용된다. 실행마다 소유자 기록을 요구하지 않는다. 레지스트리 v2 의 `generation_policy`
가 이 결정을 이름으로 적고, `--authorized-by` 없이 실행한 `soma-synth run` 은 그것을 run
`manifest.json` 의 `generation_decision` 에 남긴다: 기준(`standing_decision`), ADR, 결정 기록 경로와
그 SHA-256, 등급, 배포 범위, `releases_holds: false`. 적재할 때 정책의 등급이
`experimental_non_candidate`, 범위가 `internal_only`, `releases_holds` 가 `false` 가 아니면
레지스트리가 거부된다.

**등급 강제.** 러너는 두 번 확인한다.

- 레지스트리가 소스에 `experimental_non_candidate` 가 아닌 `artifact_class` 를 선언하면 아무것도
  실행하기 전에 거부한다(소유자 기록을 건네도 마찬가지다).
- 생성이 끝난 번들의 `INDEX.json` 이 `artifact_class: experimental_non_candidate`,
  `distribution_scope: internal_only` 가 아니거나 `INDEX.json` 이 없으면 generate 단계를 실패로 끝낸다
  (`logs/artifact_class.txt`). 코퍼스 지문 파일이 두 필드를 적으면 그것도 같은 값이어야 한다(§1.3).

**`soma-synth run` 밖의 생성은 상시 허용의 범위가 아니다.** 상시 허용의 조건은 실행 기록(결정, 원천 해시,
코드 리비전)이고, 그것을 남기는 것은 `soma-synth run` 뿐이다. `soma-synth pipeline` 의 generate 단계는
거부되며(종료 코드 3) `soma-synth run` 을 가리킨다. `pipeline` 은 이미 있는 번들의 validate → readme →
register 에 쓴다(`--skip-generate` 또는 `--start-at validate`). 생성기 스크립트를 직접 실행하는 것은 러너가
막을 수 없으므로 여기서 범위 밖이라고 적어 둔다.

**누구의 PC 에서든 같다.** 결정 기록 §4 는 내부 사용자가 **자기 PC 에서** 생성하는 것을 다룬다
(ADR-0041 `1.2.0`). 러너는 누구의 PC 인지 가리지 않고 같은 조건(등급 강제, 실행 기록)을 건다. 어느 PC
에서도 [보존 규칙](RETENTION_RULES.md) §1 은 그대로다.
운영 lineage 를 바꾸는 것은 1.2 의 순서를 따르고, 러너는 그 증거가 `_superseded` 에 없으면 운영 lineage 의
`--replace-existing` 을 거부한다(§1.3). `_superseded`·`_runs`·`_manifest_backfill` 과 원천은 지우지 않으며
(1.4), 러너는 원천·바디 모델·증거 폴더 안에 쓰지 않는다(§1.3).

**거부는 성공이 아니다.** 거부된 실행은 실행 기록(`logs/refusal.txt`)을 남기고 `soma-synth run` 은 종료 코드
3 으로 끝난다(실패한 단계는 1, 경로·레지스트리를 풀지 못한 경우는 2). 읽기 전용·증거 폴더
거부는 실행 기록을 열기 전이라 기록 없이 종료 코드 3 으로 끝난다(§1.3).

**소스별 소유자 기록은 여전히 유효하다.** 소유자가 `research/decisions/` 에 날짜·소스·
`experimental_non_candidate` 범위를 적고 `releases_holds: false` 라고 명시한 승인 기록(2026-09-14
방식)을 `--authorized-by <기록>` 으로 건네면, 러너는 전과 같이 `gates.py` 로 그 기록을 읽어 소스와
등급을 확인하고, 게이트가 거부한 사유와 기록을 함께 `generation_decision`(기준 `owner_record`)과
`logs/authorization.txt` 에 남긴다. 기록 파일의 SHA-256 과 사본(`authorization/<기록>`)도 실행 기록에 남는다.
소스별 기록이 허가하는 것도 `experimental_non_candidate` 뿐이다(`gates.py` 는 다른 등급을 적은 기록을
거부한다). canonical 과 candidate 생성은 hold 와 그 승인된 전이(상위 프로젝트의 hold 규칙) 아래에 있고, 상시
허용도 소스별 기록도 그것을 열지 않는다. `gates.py` 의 거부 판정은 상위 프로젝트의 hold 검사다.

**governance root 는 상위 프로젝트에 마운트됐을 때만 있다.** `gates.py` 가 읽는 M0 evaluator
(`configs/evaluators`)·게이트 설정과 소유자 기록(`research/decisions`)은 상위 프로젝트의 것이다.
`gates.GOVERNANCE_ROOT` 는 이 저장소가 상위 프로젝트 안에 `packages/soma-synth` 로 마운트돼 있고 그 상위
프로젝트 루트에 `configs/evaluators` 와 `research/decisions` 가 둘 다 있을 때만 그 루트이고, 그 밖의 배치
(단독 체크아웃, 두 폴더 중 하나가 없는 상위 프로젝트)에서는 없다. 환경 변수나 플래그로 바꿀 수 없다
(fail-closed). evaluator, 그것이 가리키는 게이트 설정, 소유자 기록, 결정 기록의 SHA-256 은 모두 이 루트에서
읽고, 상대 경로로 준 `--authorized-by` 는 작업 폴더가 아니라 이 루트 기준으로 푼다. governance root 가 없으면
`--authorized-by` 를 준 실행과 정책이 없는 레지스트리(v1)로 기록 없이 한 실행이 결정 전에 거부되고, 거부
사유가 governance root 가 없다고 적는다(`logs/refusal.txt`, 종료 코드 3). `--authorized-by` 없이 레지스트리
v2 로 한 실행, 곧 상시 허용은 영향을 받지 않는다. 그때는 결정 기록 파일을 읽을 수 없으므로
`generation_decision.decision_record_sha256` 은 해시 대신
`{"status": "unavailable", "reason": "... is not in this checkout"}` 이다.

**추적 기록.** 모든 실행 기록(`_runs/run_<identity>/`)은 §4.1 의 추적 기록을 남긴다: 원천 해시
목록(`source_manifest.json`), 코드 리비전과 `smpl18` 리비전, 생성 근거가 된 결정 기록의 경로.
라이선스가 철회되면 이것으로 영향받는 산출물을 찾는다. 코퍼스에서 만든 번들(`hknu`·`gaitex`·`addbiomechanics`)은
코퍼스 자신의 기록이 원천을 피험자·trial 이름이나 부분 해시로만 적지만, 실행 기록의 `source_manifest.json`
`source_files` 블록이 그 실행의 코퍼스·번들 단계가 읽을 수 있었던 원천 파일 전부를 파일 전체의 SHA-256 과 함께
적는다(§4.1). 코퍼스를 재사용한 실행의 코퍼스 원천은 그 코퍼스를 만든
실행의 기록에 있고, 이번 기록에는 그 코퍼스의 지문이 있다.

**번들은 그 PC 에만 둔다.** 등록은 그 PC 의 카탈로그(`<data root>/experimental/catalog`)에만 한다.
공유하거나 외부에 공개하려면 별도 승인이 필요하다.

---

## 10. 알려진 불일치

봉인 문서와 현실이 어긋나는 지점을 기록한다. **고치지 않고 보고한다**(상위 프로젝트의 봉인 문서
규칙).

| 위치 | 봉인 문서의 서술 | 현재 실체 |
|---|---|---|
| `PIPELINE_GOVERNANCE.md` §0 | `AMASS`·`PRISM`·`AddBiomechanics` 세 소스 | **5개** (+`gaitex`, `hknu`) |
| `PIPELINE_GOVERNANCE.md` §0 | `6 IMU Small` | **8채널** |
| `PIPELINE_GOVERNANCE.md` §0 | `15관절 Large` | **18관절** |
| `PIPELINE_GOVERNANCE.md` §10.1 | `run_id` 가 identity 해시 | `run_id` 는 take 라벨, §3.1 이 `run_identity` 로 우회 |
| `PIPELINE_GOVERNANCE.md` §2 | 파이프라인 위치 `src/soma_synthetic_imu/` | 생성 파이프라인은 별도 저장소 `soma-synth`(패키지 `soma_synth`, 상위 프로젝트의 submodule `packages/soma-synth`)로 옮겨 갔다(ADR-0041 결정 1) |
| `LOCAL_DATA_PLANE.md` §1·§5·§8 | 고정된 단일 데이터 영역을 가정한다 | PC 마다 `SOMA_DATA_ROOT` 가 출력 영역이고 원천은 `SOMA_SOURCE_ROOT`(ADR-0041 결정 3). 공유 드라이브 원본 배포는 §8 갱신 뒤에만 |

해소는 M0 evaluator 승급을 포함하는 review set 사안이다. 위 두 줄(§2, `LOCAL_DATA_PLANE`)은 다음 M0
승격 review set 에서 고친다(ADR-0041).

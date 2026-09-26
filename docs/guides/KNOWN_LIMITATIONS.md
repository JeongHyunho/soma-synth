# KNOWN_LIMITATIONS.json — 번들이 약속하지 못하는 것을 적는 공통 형식

> 작성 2026-09-14 · 스키마 `dataset_limitations_v1` · 코드 `src/soma_synth/contracts/dataset_limitations.py`
> 값의 정본 `configs/datasets/known_limitations_v1.yaml` · 렌더러 `scripts/write_known_limitations.py`

## 1. 왜 있는가

모든 번들에는 한계가 있다 — 실제 센서가 없어 모델로 대신한 채널, 자기계가 없어 방위가
흐르는 인솔, 자세 파라미터만 주는 소스, 감긴 좌표를 고쳐야 했던 리타깃. 지금까지는 번들마다
`DATA_DESCRIPTION_EN.md` 의 다른 절에, 다른 목소리로 산문으로 적었다. 두 번들을 비교하려면
에세이 둘을 읽어야 했고, 프로그램은 어느 쪽도 읽을 수 없었다.

이 형식은 **모양을 하나로 고정**한다. 값은 저장소의 config 한 파일에 번들별로 살고, 번들
루트의 `KNOWN_LIMITATIONS.json` 으로 렌더된다. 검증기는 그 파일을 **요구**하고(레지스트리
`universal.bundle_files`), 스키마에 맞는지 보고, 저장소의 config 와 **같은지** 본다 — 손으로
번들 쪽만 고치면 FAIL 이다.

## 2. 항목 하나의 모양

```yaml
- id: prism.insole_heading_unreferenced        # <source>.<snake_case_topic>, 안정적으로
  kind: hardware_limit                          # 아래 표
  severity: caution                             # info | caution | do_not_use
  scope: {level: site, sites: [foot_l, foot_r]} # bundle | site | array | take | subject
  statement: >-                                 # 무엇이 제한되는가, 한두 문장
    The insoles are six-axis; their heading is the gyro integral and drifts.
  consequence: >-                               # 소비자가 그래서 무엇을 하면 안 되는가
    Do not use the measured insole heading as an absolute reference.
  evidence: small_reference.npz imu_orientation_absolute_heading = [T,T,T,T,T,T,F,F]
  status: mitigated                             # open | mitigated | fixed
  since: "2026-09-09"
  mitigation: heading_synth_reference.npz carries a heading transferred from the shank.
  related: [prism.foot_gyro_axis_relabel]       # 선택
```

| `kind` | 뜻 — **누가 없앨 수 있는가** |
|---|---|
| `hardware_limit` | 센서가 측정할 수 없다 (6축에는 방위 기준이 없다) |
| `source_defect` | 소스 데이터의 결함, 우리가 만들지 않았다 |
| `source_scope` | 소스가 애초에 기록하지 않았다 (GRF 없음, 실측 IMU 없음) |
| `pipeline_defect` | 우리 처리가 만들었다 — 우리 쪽에서 고칠 수 있다 |
| `proxy` | 없는 센서를 다른 채널이 대신한다 |
| `approximation` | 명시된 단순화 가정 아래 유도된 값 |
| `coverage` | 일부 take·사이트·피험자가 없거나 제외됐다 |
| `rights` | 이 데이터로 무엇을 해도 되고 안 되는가 |

| `severity` | 소비자의 행동 |
|---|---|
| `info` | 알아 두면 좋다, 사용에 변화 없음 |
| `caution` | 쓸 수 있으나 `consequence` 를 염두에 둘 것 |
| `do_not_use` | 명시된 목적에는 **쓰면 안 된다** |

`status: mitigated` 는 `mitigation` 을, `status: fixed` 는 `fixed_in` 을 요구한다. 스키마에 없는
필드는 **거부**된다 — `source_attribution` 이 번들마다 다른 모양으로 자라난 일이 다시 일어나지
않게 하기 위해서다.

이 스키마는 존재·열거 판정만 한다. 임계값·허용 오차를 담지 않는다
(`DETERMINISTIC_EXECUTION_CONFIG_HOLD`). 한계는 **진술·범위·근거**로 적지,
점수로 적지 않는다.

## 3. 바꾸는 절차

1. `configs/datasets/known_limitations_v1.yaml` 의 해당 번들 블록을 고친다. 새 번들이면 블록을
   추가한다 — 최소한 `rights` 항목 하나는 있어야 한다.
2. 렌더한다: `python scripts/write_known_limitations.py <bundle_dir> [...]`.
   내용이 같으면 `generated_utc` 를 보존하므로 무변경 재렌더는 바이트 동일하다.
3. 검증한다: `python scripts/write_known_limitations.py <bundle_dir> --check`, 또는 번들 검증
   (`soma-synth run ... --level L4`) — 파일 부재는 L0 FAIL, 스키마 위반·config 불일치는 L4 FAIL.
4. 공유본이 있으면 상위 프로젝트의 `push_sample_data.py` 가 바뀐 파일만 옮긴다(soma-synth 는 번들을
   올리지 않는다).

## 4. 무엇을 적고 무엇을 적지 않는가

**적는다**: 번들 자신이 문서화한 것(설명서·manifest·npz provenance 배열), 저장소가 기록한 것
(레지스트리 waiver, ADR, 조사 문서). 근거(`evidence`)는 읽는 사람이 열어 확인할 수 있는
정확한 지점 — 절 이름, manifest 키, 배열 이름, 저장소 경로.

**적지 않는다**: 추측. 의심되지만 근거가 없는 것은 조사 문서로 가지, 이 파일로 오지 않는다.

**레지스트리 waiver 와의 관계**: waiver 는 *검증 규칙의 면제*이고(기한이 있고 만료되면 FAIL),
한계 항목은 *데이터에 대한 진술*이다(기한이 없다). 둘은 겹칠 수 있고 `related` 로 잇는다.
waiver 가 닫혀도 한계는 남을 수 있다 — 필드가 채워졌다고 인솔에 자기계가 생기지는 않는다.

## 5. 첫 적용

2026-09-14 에 다섯 번들(`prism_faithful_full_v1`, `hknu_unified8_v1`, `gaitex_unified8_v1`,
`amass_faithful_full_v2`, `addbio_unified8_v2`)의 기존 산문 caveat 을 이 형식으로 옮겼다. 각
번들의 `DATA_DESCRIPTION_EN.md` 산문은 그대로 두었다 — 사람이 읽는 설명이고, 이 파일은
기계와 비교를 위한 것이다. 둘이 어긋나면 이 파일과 config 가 정본이다.

같은 날 늦게 번들 이름에서 버전 접미사가 사라졌다(상위 프로젝트 보존 규칙 §1). config 의 블록 키는
lineage 이름(`prism_faithful_full`, `hknu_unified8`, `gaitex_unified8`, `amass_faithful_full`,
`addbio_unified8`)이고, 한 lineage 가 다시 생성되면 같은 블록이 새 번들에 렌더된다 — 그래서
고쳐진 항목은 지우지 않고 `status: fixed` 와 `fixed_in` 으로 남긴다. 렌더는 생성 파이프라인
러너가 생성 직후에 하고(`GENERATION_PIPELINE_STANDARD.md` §9), 기존 번들은
`scripts/write_known_limitations.py` 로 한다.

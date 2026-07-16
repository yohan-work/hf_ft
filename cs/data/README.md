# 데이터 디렉터리

`raw/synthetic_cs_v1.csv`는 Phase 3에서 제작한 개인정보 없는 인공 한국어 CS 문의 후보 데이터다.

- 총 140개, 라벨별 20개
- 모든 행은 `source=synthetic`, `is_synthetic=true`다.
- 아직 Train/Validation/Test로 분할하지 않았다. 분할과 누수 검사는 Phase 4에서 수행한다.
- 실제 고객 데이터는 이 저장소에 추가하지 않는다.

필드:

| 필드 | 설명 |
|---|---|
| `id` | 데이터셋 내 고유 식별자 |
| `text` | 라벨링 대상 고객 문의 |
| `label` | `configs/labels.yaml`의 라벨 코드 |
| `source` | 문장 출처 |
| `is_synthetic` | 인공 생성 여부 |
| `review_status` | v1 라벨 가이드 기준 검수 상태 |
| `notes` | 검수·경계 관련 메모 |

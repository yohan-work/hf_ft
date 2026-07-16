# 인공 CS 데이터셋 검수 보고서 v1

검수 대상: `data/raw/synthetic_cs_v1.csv`  
검수일: 2026-07-16  
데이터 성격: 학습 흐름 검증용 인공 한국어 문의 데이터

## 구성

| 라벨 | 건수 |
|---|---:|
| payment | 20 |
| cancellation_refund | 20 |
| delivery | 20 |
| exchange_return | 20 |
| account | 20 |
| technical_issue | 20 |
| product_general | 20 |
| 합계 | 140 |

모든 행은 `source=synthetic`, `is_synthetic=true`, `review_status=approved`로 기록됐다.
필수 필드는 `id`, `text`, `label`, `source`, `is_synthetic`, `review_status`, `notes`다.

## 자동 검수 결과

- 고유 ID: 140개, 중복 없음
- 빈 필수값: 0개
- 라벨 분포: 클래스마다 20개로 균등
- 정규화 중복: 0개. Unicode NFKC 정규화, 소문자화, 공백·문장부호 제거 후 비교했다.
- 문자 길이: 최소 16자, 최대 55자, 평균 25.3자
- 40자 이상 문의: 12개
- 구어체 또는 경미한 오타 표기 문의: 4개
- 서로 다른 라벨 간 문자 bigram Jaccard 유사도 0.60 이상 쌍: 0개

이 유사도 검사는 표면적인 템플릿 반복을 찾는 빠른 검사일 뿐 의미적 중복을 보장하지 않는다.
Phase 4에서 split 전후에 템플릿 파생 관계와 더 강한 유사도 검사를 다시 수행한다.

## 검수 판단

- 통과: 필수 필드, 라벨 수, 균형 분포, 완전·정규화 중복 검사.
- 보완: 초안의 문장 길이가 고르게 짧고 정중한 표현에 치우쳐 있어 장문·구어체·경미한 오타 문장을
  추가했다.
- 제한: 여전히 소규모 인공 데이터라 결제·환불·배송 관련 대표 단어가 라벨을 쉽게 드러낼 수 있다.
  따라서 높은 baseline 또는 Transformer 점수는 실제 일반화 성능으로 해석하지 않는다.
- 제한: `product_general`은 기타 라벨로 사용하지 않았지만, 실제 데이터에서 이 라벨의 유입 비중과
  혼동 사례를 별도로 재검토해야 한다.

## 다음 단계 전제

이 파일은 아직 raw 후보 데이터다. Phase 4에서만 seed 42 기반 Train/Validation/Test 분할을 만들고,
Test를 이후 학습·설정 변경에 사용하지 않는다.

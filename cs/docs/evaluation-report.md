# 최종 Test 평가 보고서 v1

평가일: 2026-07-16  
평가 데이터: 고정 Test 21개, 라벨별 3개  
주의: 이 Test 데이터는 Phase 7에서 처음 사용했으며, 결과를 본 뒤 학습 설정을 변경하지 않는다.

## 모델 비교

| 모델 | Accuracy | Macro Precision | Macro Recall | Macro F1 | 추론 시간 | 모델 크기 |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | 0.429 | 0.514 | 0.429 | 0.432 | 0.058 ms/text | 0.35 MB |
| Fine-tuned `klue/roberta-base` | 0.714 | 0.645 | 0.714 | 0.664 | 6.6 ms/text | 443.5 MB |

추론 시간은 CPU에서 Test 문장 배치를 warm-up 뒤 5회 반복한 평균이며, 다른 하드웨어·배치 크기·
동시 요청 환경의 운영 지표는 아니다.

Transformer는 baseline보다 Macro F1이 `0.232` 높아, 권장 개선 기준인 `+0.05`는 충족했다.
그러나 목표 Macro F1 `0.80`에는 도달하지 못했다. 인공·소규모 데이터라는 제한 때문에 이 점수를
실제 운영 성능으로 해석하지 않는다.

## Transformer 클래스별 결과

| 라벨 | Precision | Recall | F1 |
|---|---:|---:|---:|
| payment | 0.750 | 1.000 | 0.857 |
| cancellation_refund | 0.000 | 0.000 | 0.000 |
| delivery | 0.500 | 0.667 | 0.571 |
| exchange_return | 0.667 | 0.667 | 0.667 |
| account | 0.600 | 1.000 | 0.750 |
| technical_issue | 1.000 | 1.000 | 1.000 |
| product_general | 1.000 | 0.667 | 0.800 |

중요 카테고리로 볼 수 있는 `cancellation_refund` Recall은 0.0이므로, 현재 모델은 취소·환불
자동 확정 용도로 사용할 수 없다.

## 산출물

- `artifacts/metrics/final-test-metrics.json`
- `artifacts/metrics/final-test-errors.csv`
- `artifacts/figures/tfidf-test-confusion-matrix.png`
- `artifacts/figures/transformer-test-confusion-matrix.png`

`scripts/evaluate.py`는 이미 최종 결과가 있으면 기본적으로 다시 실행하지 않는다. 명시적으로
승인된 재실행에서만 `--force`를 사용한다.

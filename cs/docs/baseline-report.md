# Baseline Validation 보고서 v1

실행 명령: `uv run python scripts/train_baseline.py`  
평가 데이터: Validation 21개(라벨별 3개)  
Test 데이터: 사용하지 않음

## 결과 요약

| 모델 | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| Majority Class (`payment`) | 0.143 | 0.020 | 0.143 | 0.036 |
| TF-IDF + Logistic Regression | 0.571 | 0.526 | 0.571 | 0.530 |

TF-IDF는 단어 n-gram(1–2)과 문자 n-gram(2–5)을 결합했다. `C=0.1`, `1.0`, `3.0`을
Validation Macro F1로 비교했고 각각 `0.530`, `0.420`, `0.420`이어서 `C=0.1`을 선택했다.

## 클래스별 TF-IDF 결과

| 라벨 | Precision | Recall | F1 |
|---|---:|---:|---:|
| payment | 0.500 | 0.667 | 0.571 |
| cancellation_refund | 0.000 | 0.000 | 0.000 |
| delivery | 0.500 | 0.333 | 0.400 |
| exchange_return | 0.750 | 1.000 | 0.857 |
| account | 0.600 | 1.000 | 0.750 |
| technical_issue | 1.000 | 0.667 | 0.800 |
| product_general | 0.333 | 0.333 | 0.333 |

## 해석과 주의점

- 최다 라벨 baseline은 균등 데이터에서 라벨 하나만 맞히므로 Accuracy 1/7, Macro F1 0.036이다.
- TF-IDF의 0.530은 단순 최다 예측보다 의미 있는 비교 기준이지만, 목표치 0.80에는 미달한다.
- `cancellation_refund`가 `payment` 또는 `product_general`으로, `delivery`가 `exchange_return` 또는
  `product_general`으로 혼동됐다. 이는 경계 표현과 학습 표본 수가 적다는 신호다.
- 계수 상위 특징에는 `결제`, `취소`, `환불`, `배송`, `교환`, `반품`, `계정`처럼 라벨을 직접 드러내는
  단어가 많다. 이 데이터는 키워드 편향 가능성이 있으므로 Transformer 성능을 일반화 성능으로 과장하지 않는다.
- 9개 오답의 confidence는 약 0.15–0.16으로 낮았다. 모델이 확신한 오답이 아니라, 단문·소규모
  데이터에서 경계를 잘 구분하지 못한 결과에 가깝다.

## 생성 산출물

- `artifacts/metrics/baseline-validation.json`
- `artifacts/metrics/tfidf-validation-errors.csv`
- `artifacts/figures/majority-validation-confusion-matrix.png`
- `artifacts/figures/tfidf-validation-confusion-matrix.png`
- `artifacts/models/tfidf-logistic-regression.joblib`

이 산출물은 실행으로 재생성되며 Git에는 포함하지 않는다.

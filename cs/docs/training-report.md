# Transformer 파인튜닝 실행 보고서 v1

실행 명령: `uv run python scripts/train_transformer.py`  
평가 데이터: Validation 21개(라벨별 3개)  
Test 데이터: 사용하지 않음

## 실행 설정

| 항목 | 값 |
|---|---|
| Base model | `klue/roberta-base` |
| 실제 revision | `02f94ba5e3fcb7e2a58a390b8639b0fac974a8da` |
| 라벨 수 | 7 |
| Seed | 42 |
| Epoch | 3 |
| Learning rate | 2e-5 |
| Train/Eval batch size | 8 / 8 |
| Weight decay | 0.01 |
| Max length | 128 |
| Best model 기준 | Validation Macro F1 |
| 장치 | CPU |
| Trainable parameters | 110,623,495 |
| 학습 시간 | 23.9초 |

현재 설치된 PyTorch에서 MPS backend는 빌드됐지만 사용 가능 상태가 아니어서 CPU로 실행했다.
이 작은 데이터셋에서는 CPU 학습이 가능했으나, 더 큰 데이터셋이나 반복 실험에는 Colab 등의 GPU
환경을 검토한다.

## Validation 결과

| 모델 | Accuracy | Macro F1 |
|---|---:|---:|
| Majority Class | 0.143 | 0.036 |
| TF-IDF + Logistic Regression | 0.571 | 0.530 |
| Fine-tuned `klue/roberta-base` | 0.714 | 0.664 |

Transformer는 TF-IDF baseline보다 Validation Macro F1이 `0.134` 높다. 이는 선택한 데이터와
분할에서의 개선일 뿐, 인공·소규모 데이터이므로 실제 고객 문의 일반화 성능의 증거는 아니다.

클래스별 F1은 payment `1.000`, cancellation_refund `0.500`, delivery `0.800`,
exchange_return `0.600`, account `0.750`, technical_issue `1.000`, product_general `0.000`이었다.
`product_general`이 여전히 전혀 맞지 않아, 다음 분석 단계에서 이 라벨의 문장 범위와 혼동 사례를
우선 검토한다.

## 저장·재로딩 확인

- 최고 checkpoint: `artifacts/checkpoints/klue-roberta-cs-v1/checkpoint-39`
- 최종 모델: `models/klue-roberta-cs-v1/` (`model.safetensors`, tokenizer, config)
- 최종 모델을 새로 로드한 뒤 “결제는 완료됐는데 주문 내역이 보이지 않아요.”를 추론해
  `payment`를 반환함을 확인했다. confidence는 `0.179`로 낮아, 작은 데이터에서 확률을
  운영 확정값으로 쓰면 안 된다.

전체 실행 메타데이터와 Validation 지표는 `artifacts/metrics/transformer-validation.json`에 저장된다.

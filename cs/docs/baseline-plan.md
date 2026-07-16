# Baseline 및 평가 계획 v1

## 비교 대상

| 모델 | 학습 데이터 | 목적 |
|---|---|---|
| Majority Class | 사용하지 않음 | 가장 흔한 라벨만 예측하는 최저 기준선 |
| TF-IDF + Logistic Regression | Train | 단어·문자 패턴 기반의 강한 고전 분류 기준선 |
| `klue/roberta-base` 파인튜닝 | Train, Validation으로 선택 | 문맥을 반영하는 한국어 Transformer 분류 모델 |

일반 LLM 프롬프트 분류는 v1의 필수 비교 대상이 아니다. API 비용·프롬프트 변화·재현성이라는
별도 변수가 있어, 분류 모델 완료 후 선택적으로 다룬다.

## 데이터 사용 규칙

- Train: 모델 파라미터 학습에만 사용한다.
- Validation: 전처리와 hyperparameter 선택, early stopping, 최고 checkpoint 선택에 사용한다.
- Test: 모든 설계와 학습 설정을 고정한 뒤 Phase 7에서 한 번만 최종 비교에 사용한다.
- split 전 중복·정규화 중복·템플릿 파생 문장·높은 유사도 문장을 검사한다.

## 재현성 고정값

- random seed: `42`
- 라벨 순서: `configs/labels.yaml`의 ID `0`부터 `6`
- 기본 split: stratified Train/Validation/Test = `70/15/15`
- Transformer 초기값: epoch `3`, learning rate `2e-5`, batch size `16`, weight decay `0.01`, max length `128`
- MPS 메모리가 부족하면 실제 batch size만 낮추고 gradient accumulation으로 유효 batch size를 기록한다.
- 모든 실행은 모델 ID/revision, 데이터셋 버전, 패키지 버전, 장치, 실행 시간, 설정 파일을 기록한다.

## 지표와 모델 선택

- 1차 지표: Validation **Macro F1**. 클래스 불균형에서 다수 클래스가 결과를 지배하지 않도록 한다.
- 병행 지표: Accuracy, Macro Precision, Macro Recall, 클래스별 Precision/Recall/F1, Confusion Matrix.
- 최고 Transformer checkpoint는 Validation Macro F1로 선택한다.
- Test 결과는 모델 선택에 되돌려 사용하지 않는다.
- 추론 시간은 warm-up 후 같은 장치에서 여러 문장을 반복 실행한 평균과 p95를 기록한다.
- 모델 크기는 배포에 쓰는 최종 safetensors·config·tokenizer 디렉터리의 합계로 기록한다.

## TF-IDF baseline의 고정 설계

- 입력: 최소 정규화된 원문 텍스트.
- 특징: 한국어 단어 n-gram `(1, 2)`과 문자 n-gram `(2, 5)`을 결합한다.
- 분류기: `LogisticRegression`, `max_iter=1000`, `class_weight="balanced"`, `random_state=42`.
- 조정은 Validation에서 `C` 후보 `0.1`, `1.0`, `3.0`만 비교한다. Test는 사용하지 않는다.
- 단일 키워드 또는 템플릿에 과도하게 의존하는지 상위 특징과 오분류를 함께 점검한다.

## 비정상 결과의 처리

- baseline Macro F1이 매우 높으면 Transformer 진행 전에 split 누수, 반복 템플릿, 라벨별 고정 키워드를 조사한다.
- 특정 클래스 Recall이 낮으면 곧바로 데이터를 늘리지 않는다. 라벨 정의, 데이터 분포, 경계 사례, 실제 오분류를 먼저 분류한다.
- 인공 데이터 결과는 학습 파이프라인의 검증 결과이며, 실제 고객 데이터 일반화 성능으로 주장하지 않는다.

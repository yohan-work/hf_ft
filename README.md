# Hugging Face 한국어 CS 파인튜닝 실습

한국어 고객 문의를 7개 카테고리로 분류하는 모델을 직접 만들며, Hugging Face의 데이터 처리·
파인튜닝·평가 흐름을 익히는 학습 프로젝트입니다.

이 저장소의 구현은 [`cs/`](cs/)에 있습니다. 완성형 고객센터가 아니라, 재현 가능한 머신러닝
실험 과정을 이해하는 것이 목적입니다.

## 현재 상태

Phase 6까지 완료했습니다.

| 단계 | 상태 | 결과 |
|---|---|---|
| 문제·라벨 설계 | 완료 | 7개 단일 라벨과 경계 사례 정의 |
| 인공 데이터셋 | 완료 | 라벨별 20개, 총 140개 |
| 데이터 파이프라인 | 완료 | seed 42 기반 70/15/15 split, tokenizer 검증 |
| Baseline | 완료 | TF-IDF + Logistic Regression Validation Macro F1 0.530 |
| Transformer 파인튜닝 | 완료 | `klue/roberta-base` Validation Macro F1 0.664 |
| 최종 Test 평가·오분류 분석 | 예정 | Test 데이터는 아직 사용하지 않음 |
| API·Model Lab·Hub 업로드 | 예정 | 별도 Phase에서 진행 |

현재 결과는 **인공 소규모 데이터**의 Validation 결과입니다. 실제 고객 문의 일반화 성능이나 운영
품질을 뜻하지 않습니다.

## 분류 라벨

| 코드 | 한글명 |
|---|---|
| `payment` | 결제 |
| `cancellation_refund` | 취소·환불 |
| `delivery` | 배송 |
| `exchange_return` | 교환·반품 |
| `account` | 계정 |
| `technical_issue` | 기술 오류 |
| `product_general` | 상품·일반 문의 |

라벨의 포함·제외 기준과 복합 문의 처리 규칙은 [`cs/docs/labeling-guide.md`](cs/docs/labeling-guide.md)에
있습니다.

## 빠른 실행

Python 3.11과 [`uv`](https://docs.astral.sh/uv/)가 필요합니다.

```bash
cd cs
uv sync --group dev
uv run python scripts/prepare_data.py
uv run python scripts/train_baseline.py
uv run python scripts/train_transformer.py
uv run pytest
```

실행 순서:

1. `prepare_data.py`: 원본 CSV를 Train/Validation/Test로 나누고, Hugging Face tokenizer로 token 길이를 확인합니다.
2. `train_baseline.py`: 단어·문자 패턴 기반 TF-IDF 모델을 학습합니다.
3. `train_transformer.py`: `klue/roberta-base`에 7개 CS 라벨 분류 헤드를 붙여 파인튜닝합니다.
4. `pytest`: 전처리, 라벨 매핑, split 재현성, 평가 지표를 검증합니다.

Hugging Face 모델 파일과 학습 산출물은 로컬 캐시·`models/`·`artifacts/`에 생성되며 Git에 포함하지
않습니다.

## 핵심 개념

```text
고객 문의 문장
  → tokenizer가 숫자(token ID)로 변환
  → 사전학습된 한국어 모델이 문맥을 해석
  → 분류 헤드가 7개 라벨의 점수를 계산
  → 가장 높은 점수의 라벨을 반환
```

- **사전학습 모델**: 대량의 한국어 텍스트로 언어 패턴을 이미 배운 모델
- **파인튜닝**: 이 모델에 CS 문의와 정답 라벨을 추가로 학습시키는 과정
- **Train / Validation / Test**: 학습용 문제집 / 중간 점검 / 마지막 시험
- **Macro F1**: 모든 라벨을 동등하게 보고 성능을 평균 내는 지표

## 문서 안내

- [전체 단계 계획](cs/docs/phase-01.md)
- [모델 선택과 라이선스](cs/docs/model-selection.md)
- [라벨링 가이드](cs/docs/labeling-guide.md)
- [데이터셋 검수 보고서](cs/docs/dataset-report.md)
- [데이터 파이프라인 결과](cs/docs/data-pipeline-report.md)
- [Baseline 결과](cs/docs/baseline-report.md)
- [Transformer 학습 결과](cs/docs/training-report.md)

## 주의사항

- 데이터셋은 전부 인공 데이터이며 실제 개인정보를 포함하지 않습니다.
- 고정 Test 데이터는 다음 평가 단계 전까지 학습·설정 조정에 사용하지 않습니다.
- Hugging Face Hub 업로드, 외부 배포, 실제 데이터 반입은 별도 확인 후 진행합니다.
- `klue/roberta-base`의 라이선스와 파생 모델 배포 의무는 외부 공개 전에 재검토해야 합니다.

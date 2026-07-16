# Hugging Face 한국어 CS 모델 파인튜닝 프로젝트

## 0. 역할

너는 Hugging Face 기반 머신러닝 프로젝트를 설계하고 구현하는 ML 엔지니어이자, 사용자가 파인튜닝 전체 과정을 이해하도록 돕는 기술 멘토다.

이번 프로젝트의 목적은 완성형 고객센터 서비스를 빠르게 만드는 것이 아니다.

다음 전체 사이클을 사용자가 직접 이해하고 경험할 수 있는 재현 가능한 프로젝트를 만드는 것이 목적이다.

1. Hugging Face 모델 탐색 및 선택
2. 사전학습 모델 다운로드
3. 학습 데이터셋 설계 및 제작
4. 텍스트 전처리와 토큰화
5. 모델 파인튜닝
6. 성능 평가와 오분류 분석
7. 모델 저장 및 재사용
8. Hugging Face Hub 업로드
9. FastAPI 추론 API 연결
10. 간단한 Model Lab 화면 연결
11. 이후 소형 LLM LoRA 파인튜닝으로 확장

모든 작업은 단계별로 진행한다.

한 번에 전체 프로젝트를 구현하지 말고, 각 단계가 끝날 때마다 결과와 근거를 보고한 뒤 사용자의 확인을 받고 다음 단계로 이동한다.

중요한 판단을 할 때 내부 추론 전체를 장황하게 노출하지 말고 다음만 명확하게 설명한다.

- 확인한 사실
- 현재 가정
- 선택한 방향
- 선택 이유
- 대안과 트레이드오프
- 다음 단계 진입 조건

---

# 1. 프로젝트 목표

한국어 고객 문의를 자동 분류하는 모델을 직접 파인튜닝한다.

예시 입력:

"고객이 카드 결제는 완료됐는데 주문 내역에 상품이 표시되지 않는다고 문의했습니다."

예시 출력:

{
  "label": "payment",
  "label_name": "결제",
  "confidence": 0.94
}

초기 분류 카테고리는 다음 7개를 후보로 사용한다.

| ID | 코드 | 카테고리 |
|---:|---|---|
| 0 | payment | 결제 |
| 1 | cancellation_refund | 취소·환불 |
| 2 | delivery | 배송 |
| 3 | exchange_return | 교환·반품 |
| 4 | account | 계정 |
| 5 | technical_issue | 기술 오류 |
| 6 | product_general | 상품·일반 문의 |

단, 이 분류 체계를 바로 확정하지 않는다.

데이터셋 제작 전에 다음을 검토해야 한다.

- 카테고리가 서로 배타적인가?
- 문의 문장만 보고 정답을 판단할 수 있는가?
- 결제와 환불처럼 경계가 겹치는 유형은 무엇인가?
- 하나의 문의에 여러 문제가 포함되면 어떻게 처리할 것인가?
- 단일 라벨 분류가 적합한가, 다중 라벨이 필요한가?
- 기타 카테고리가 지나치게 넓지 않은가?
- 실제 운영에서 이 분류 결과가 어떤 행동으로 이어지는가?

첫 프로젝트에서는 학습과 평가를 단순화하기 위해 단일 라벨 분류를 우선 검토한다.

---

# 2. 프로젝트 성공 기준

이번 프로젝트의 성공은 단순히 모델 학습 코드가 실행되는 것으로 판단하지 않는다.

다음 결과물이 모두 재현 가능해야 한다.

## 필수 결과물

- 모델 선정 근거
- 모델 라이선스 확인 결과
- 라벨 정의서
- 데이터셋 생성 및 검수 기준
- Train/Validation/Test 분리 데이터
- 전처리 및 토큰화 코드
- 학습 설정 파일
- 학습 실행 코드
- 학습 로그 및 체크포인트
- 평가 지표
- Confusion Matrix
- 클래스별 Precision, Recall, F1
- 오분류 사례 보고서
- 모델 추론 스크립트
- FastAPI 추론 API
- 실행 및 재현 방법이 정리된 README
- Model Card 초안

## 주요 평가 지표

1차 핵심 지표는 Macro F1으로 한다.

함께 확인할 지표:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1
- 클래스별 Precision
- 클래스별 Recall
- 클래스별 F1
- Confusion Matrix
- 추론 시간
- 모델 크기

Accuracy만 보고 성능을 판단하지 않는다.

특정 카테고리에 데이터가 몰리면 Accuracy가 높아도 소수 카테고리를 제대로 분류하지 못할 수 있기 때문이다.

## 권장 목표

- Macro F1 0.80 이상
- 기존 Baseline보다 Macro F1 0.05 이상 개선
- 중요 카테고리 Recall 0.75 이상
- 동일 설정으로 학습 재현 가능
- 새로운 문장에 대한 로컬 추론 가능

이 수치는 절대적인 완료 조건이 아니다.

데이터가 적거나 인공 데이터 중심이라면 높은 점수가 실제 일반화 성능을 의미하지 않을 수 있다. 지표가 높더라도 데이터 누수나 문장 패턴 반복이 없는지 먼저 검증한다.

---

# 3. 핵심 원칙

## 3.1 파인튜닝이 필요한지 검증한다

다음 질문에 먼저 답한다.

- 키워드 규칙만으로 해결할 수 있는 문제인가?
- 일반 LLM 프롬프트만으로 충분한가?
- 파인튜닝으로 개선하고자 하는 구체적인 실패 유형은 무엇인가?
- 파인튜닝 결과를 객관적으로 측정할 수 있는가?
- 모델이 학습해야 할 일관된 패턴이 데이터에 존재하는가?

이번 프로젝트는 학습 목적이 있으므로 파인튜닝을 진행하되, TF-IDF와 Logistic Regression 같은 간단한 Baseline도 함께 만든다.

`klue/roberta-base`는 기본 상태에서 CS 카테고리를 출력하는 모델이 아니다. 무작위로 초기화된 분류 헤드를 붙인 학습 전 모델과 단순 비교해 “성능이 개선됐다”고 과장하지 않는다.

비교 대상으로 다음을 사용한다.

1. Majority Class Baseline
2. TF-IDF + Logistic Regression
3. Fine-tuned Transformer
4. 선택적으로 일반 LLM Prompt 분류

## 3.2 데이터 품질을 모델보다 우선한다

- 라벨 기준이 모호하면 학습을 시작하지 않는다.
- Test 데이터는 학습과 프롬프트 개선에 사용하지 않는다.
- 동일하거나 지나치게 유사한 문장이 Train과 Test에 동시에 들어가지 않게 한다.
- 템플릿 문장에서 단어만 바꾼 데이터를 무작정 늘리지 않는다.
- 생성형 AI로 만든 데이터는 인공 데이터임을 표시한다.
- 개인정보가 포함된 실제 상담 데이터는 원본 그대로 커밋하지 않는다.
- 이메일, 전화번호, 주문번호, 이름 등은 마스킹한다.

## 3.3 외부 변경은 확인 후 수행한다

다음 작업은 사용자 확인 없이 수행하지 않는다.

- Hugging Face Hub에 모델 또는 데이터셋 Push
- Public Repository 생성
- 외부 서비스 배포
- 유료 GPU 또는 유료 API 사용
- 실제 고객 데이터 업로드
- Git Remote Push

Hugging Face Repository를 생성할 경우 기본적으로 Private을 제안한다.

## 3.4 범위를 통제한다

초기 단계에서는 다음 기능을 구현하지 않는다.

- 완성형 CS 관리자 대시보드
- 고객센터 채널 연동
- 이메일 자동 발송
- RAG
- 상담사 자동 배정
- 음성인식
- 실시간 운영 자동화
- 환불이나 보상 자동 결정
- 멀티 에이전트 구조

먼저 한 개 모델, 한 개 입력, 한 개 출력, 한 개 평가 문제를 완성한다.

---

# 4. 권장 기술 구성

## 파인튜닝 1단계

- Python 3.11 권장
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Hugging Face Evaluate
- Accelerate
- scikit-learn
- pandas
- matplotlib
- seaborn
- FastAPI
- Uvicorn
- Pydantic

## 모델 후보

1차 후보:

- `klue/roberta-base`

모델을 확정하기 전에 반드시 다음을 확인한다.

- Model Card
- License
- 언어
- 모델 크기
- 최대 입력 길이
- 지원 프레임워크
- 최근 사용 가능 여부
- Safetensors 지원 여부
- 상업적 이용 또는 배포 제약
- 신뢰할 수 있는 제작 주체인지

모델 선택 결과는 `docs/model-selection.md`에 기록한다.

`klue/roberta-base` 사용에 문제가 있으면 한국어 텍스트 분류에 적합한 다른 모델을 비교하고 대안을 제시한다. 인기나 다운로드 수만으로 모델을 선택하지 않는다.

## 실행 환경

우선순위:

1. 현재 로컬 환경 점검
2. Apple Silicon MPS 사용 가능 여부 확인
3. 로컬 학습이 비효율적이면 Google Colab 또는 Kaggle용 Notebook 생성
4. 추론은 로컬 환경에서 가능하게 유지

Notebook만 유일한 실행 코드로 만들지 않는다.

학습의 기준 코드는 Python Script로 관리하고, Notebook은 해당 코드를 호출하거나 동일 절차를 설명하는 보조 수단으로 사용한다.

---

# 5. 권장 프로젝트 구조

현재 저장소 구조와 기존 파일을 먼저 확인하고, 기존 프로젝트가 있다면 함부로 덮어쓰지 않는다.

신규 프로젝트라면 다음 구조를 후보로 사용한다.

hf-cs-finetune-lab/
├── README.md
├── .gitignore
├── .env.example
├── pyproject.toml
├── requirements.txt
├── configs/
│   ├── labels.yaml
│   └── train.yaml
├── data/
│   ├── README.md
│   ├── raw/
│   ├── interim/
│   └── processed/
├── docs/
│   ├── project-plan.md
│   ├── model-selection.md
│   ├── labeling-guide.md
│   ├── dataset-report.md
│   ├── evaluation-report.md
│   └── error-analysis.md
├── notebooks/
│   ├── 01_dataset_analysis.ipynb
│   └── 02_colab_training.ipynb
├── src/
│   └── cs_classifier/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── preprocess.py
│       ├── metrics.py
│       ├── train_baseline.py
│       ├── train_transformer.py
│       ├── evaluate.py
│       ├── predict.py
│       └── api.py
├── scripts/
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── tests/
│   ├── test_preprocess.py
│   ├── test_label_mapping.py
│   └── test_api.py
├── artifacts/
│   ├── metrics/
│   ├── figures/
│   └── reports/
└── models/
    └── .gitkeep

다음 파일은 Git에 포함하지 않는다.

- Hugging Face Token
- 실제 개인정보
- 원본 고객 상담 데이터
- 대용량 모델 체크포인트
- 캐시 파일
- 로컬 가상환경
- 학습 중 생성된 임시 파일

---

# 6. 단계별 실행 계획

각 Phase가 끝나면 반드시 다음 형식으로 보고한다.

## Phase 완료 보고

- 수행한 작업
- 생성하거나 수정한 파일
- 실행한 검증
- 검증 결과
- 발견된 문제
- 현재 가정
- 다음 단계에서 결정할 사항
- 다음 Phase 진입 가능 여부

사용자 확인 전에는 다음 Phase로 넘어가지 않는다.

---

## Phase 0. 저장소와 실행 환경 진단

### 목표

코드를 작성하기 전에 현재 프로젝트와 실행 환경을 파악한다.

### 수행 작업

- 현재 디렉터리 구조 확인
- 기존 Git 저장소 여부 확인
- 기존 수정사항 확인
- AGENTS.md 또는 프로젝트 지침 확인
- Python 버전 확인
- 패키지 관리자 확인
- 운영체제와 CPU 아키텍처 확인
- Apple Silicon 여부 확인
- CUDA 또는 MPS 가능 여부 확인
- 사용 가능한 디스크 용량 확인
- Hugging Face 관련 기존 설정 확인
- `.gitignore` 상태 확인

민감한 Token 값은 출력하지 않는다.

### 사용자에게 확인할 사항

환경에서 확인할 수 없는 경우 다음만 질문한다.

1. 학습에 사용할 환경
   - Mac 로컬
   - Google Colab
   - Kaggle
   - NVIDIA GPU PC

2. 학습 데이터 상황
   - 실제 CS 데이터 보유
   - 익명화된 샘플만 보유
   - 데이터 없음
   - 인공 데이터로 학습 경험만 우선 진행

3. Hugging Face 계정
   - 계정과 Token 준비됨
   - 계정만 있음
   - 아직 없음

### 완료 조건

- 현재 환경 보고서 작성
- 프로젝트 루트 확정
- 학습 환경 후보 결정
- 데이터 보유 상태 확인
- 다음 Phase 진행 전 사용자 확인

### 중요

첫 실행에서는 Phase 0만 수행한다.

아직 의존성을 설치하거나 모델을 다운로드하거나 프로젝트 전체를 구현하지 않는다.

---

## Phase 1. 문제와 라벨 체계 설계

### 목표

모델이 무엇을 예측해야 하는지 명확하게 정의한다.

### 수행 작업

- CS 분류 문제 정의
- 7개 후보 카테고리 검토
- 라벨 코드와 표시명 정의
- 포함 기준 작성
- 제외 기준 작성
- 경계 사례 작성
- 복합 문의 처리 기준 작성
- 기타 카테고리 사용 기준 작성
- 각 카테고리당 예시 10개 작성
- 애매한 예시 20개 별도 작성

### 라벨 정의서 예시

payment:
- 포함: 결제 실패, 중복 결제, 결제 승인 후 주문 미생성
- 제외: 결제 후 주문 취소 요청은 cancellation_refund
- 경계: 쿠폰 적용 오류가 결제 문제인지 상품 정책 문제인지 정의 필요

### 소크라틱 검토 질문

- 이 라벨은 문의 문장만 보고 판단 가능한가?
- 다른 라벨과 겹치지 않는가?
- 담당자가 바뀌어도 같은 라벨을 선택할 수 있는가?
- 이 라벨로 분류했을 때 실제 후속 행동이 달라지는가?
- 모델이 단어를 외우는 것이 아니라 의도를 학습할 수 있는가?
- 복합 문의에서 가장 중요한 의도를 어떻게 정하는가?

### 완료 조건

- `configs/labels.yaml`
- `docs/labeling-guide.md`
- 카테고리별 예시
- 경계 사례 목록
- 사용자 라벨 체계 승인

---

## Phase 2. 모델 조사 및 Baseline 설계

### 목표

모델을 다운로드하기 전에 후보 모델과 비교 기준을 확정한다.

### 수행 작업

- `klue/roberta-base` Model Card 조사
- License 확인
- 모델 크기와 입력 길이 확인
- 대안 모델 1~2개 조사
- 모델 선택 비교표 작성
- Majority Baseline 설계
- TF-IDF + Logistic Regression Baseline 설계
- 평가 지표 정의
- Random Seed 정의
- 재현성 기준 정의

### 확인 질문

- 이 문제에 생성형 LLM이 정말 필요한가?
- 더 작은 모델로 같은 목적을 달성할 수 있는가?
- 모델 성능을 무엇과 비교할 것인가?
- 모델 크기 증가가 정확도 향상에 비해 정당한가?
- 한국어 문장과 실제 CS 표현에 적합한가?

### 완료 조건

- `docs/model-selection.md`
- Baseline 계획
- 평가 기준
- 모델 선택 승인

---

## Phase 3. 소규모 데이터셋 제작

### 목표

대량 데이터 제작 전에 전체 파이프라인을 검증할 최소 데이터를 만든다.

### 초기 데이터 규모

- 7개 카테고리
- 카테고리당 20개
- 총 140개 내외

### 데이터 필드

- `id`
- `text`
- `label`
- `source`
- `is_synthetic`
- `review_status`
- `notes`

### 원칙

- 문장 패턴을 다양하게 작성
- 특정 키워드만으로 라벨이 결정되지 않게 구성
- 짧은 문의와 긴 문의를 섞기
- 오타와 구어체 일부 포함
- 경계 사례 포함
- 개인정보는 가상 값이나 마스킹 값 사용
- 생성 데이터는 `is_synthetic=true`로 표시
- Test 후보 데이터는 별도 검수

### 검증

- 중복 문장 확인
- 유사 문장 확인
- 라벨 분포 확인
- 빈 값 확인
- 너무 짧거나 긴 문장 확인
- 카테고리별 대표 키워드 편향 확인

### 완료 조건

- 샘플 데이터셋
- 데이터 통계
- 중복·유사도 검사 결과
- 사용자 샘플 검수 및 승인

---

## Phase 4. 데이터 파이프라인 구현

### 목표

원본 데이터를 재현 가능한 학습 데이터로 변환한다.

### 수행 작업

- CSV 또는 JSONL 로더 구현
- 텍스트 정규화 구현
- 라벨 인코딩 구현
- Stratified Split 구현
- Train/Validation/Test 분리
- 데이터 통계 생성
- Hugging Face Dataset 변환
- 토크나이저 연결
- 최대 입력 길이 분석
- 단위 테스트 작성

### 기본 분할

- Train 70%
- Validation 15%
- Test 15%

소규모 샘플에서는 클래스별 데이터가 너무 적을 수 있으므로 분할 안정성을 검토한다. 필요하면 Phase 4 전에 데이터를 확대한다.

### 데이터 누수 검사

- 완전 동일 문장
- 공백·문장부호만 다른 문장
- 단어 일부만 바꾼 템플릿 문장
- 같은 원본에서 파생된 문장
- Test 문장이 Train 문장과 지나치게 유사한 경우

### 완료 조건

- 동일 Seed에서 동일 분할 재현
- 클래스 분포 보고서
- 데이터 누수 검사 통과
- 전처리 테스트 통과

---

## Phase 5. Baseline 학습

### 목표

Transformer 파인튜닝 전에 비교 기준을 만든다.

### 구현 대상

1. Majority Class Baseline
2. TF-IDF + Logistic Regression

### 평가 결과

- Accuracy
- Macro F1
- 클래스별 F1
- Confusion Matrix
- 오분류 샘플

### 확인 질문

- 단순 모델이 이미 충분히 잘하고 있지 않은가?
- 데이터가 키워드 중심으로 너무 쉽게 구성되지 않았는가?
- 특정 단어 하나가 정답을 결정하고 있지 않은가?
- Test 데이터가 실제보다 지나치게 단순하지 않은가?

Baseline 성능이 비정상적으로 높으면 Transformer 학습 전에 데이터셋 편향을 먼저 조사한다.

### 완료 조건

- Baseline 학습 재현 가능
- 결과 파일 저장
- 평가 보고서 초안
- Transformer 학습 진입 승인

---

## Phase 6. Transformer 파인튜닝

### 목표

선택한 한국어 사전학습 모델을 CS 분류 데이터로 파인튜닝한다.

### 초기 학습 설정 후보

- Seed: 42
- Epoch: 3
- Learning Rate: 2e-5
- Train Batch Size: 16
- Eval Batch Size: 16
- Weight Decay: 0.01
- Max Length: 128
- Evaluation Strategy: Epoch
- Save Strategy: Epoch
- Best Model 기준: Macro F1
- Early Stopping 검토

환경과 데이터 규모를 확인한 후 값을 확정한다.

### 구현

- `AutoTokenizer`
- `AutoModelForSequenceClassification`
- `TrainingArguments`
- `Trainer`
- `compute_metrics`
- 체크포인트 저장
- 최고 성능 모델 저장
- 학습 로그 저장

### 반드시 기록할 정보

- 모델 ID
- 모델 Revision
- 라이브러리 버전
- 데이터셋 버전
- Seed
- Hyperparameter
- 학습 환경
- 학습 시간
- 최고 체크포인트
- Trainable Parameter 수

### 완료 조건

- 학습 정상 완료
- Validation 평가 완료
- 모델 저장
- 동일 모델 재로드 가능
- 샘플 문장 추론 가능

---

## Phase 7. 최종 평가 및 오분류 분석

### 목표

단순 점수가 아니라 모델이 무엇을 배우고 어디서 실패하는지 분석한다.

### 수행 작업

- 고정된 Test 데이터로 최종 평가
- Baseline과 Transformer 비교
- Confusion Matrix 생성
- 클래스별 지표 생성
- 오분류 사례 수집
- 낮은 신뢰도 사례 수집
- 높은 신뢰도의 오답 수집
- 카테고리 경계 문제 분석
- 데이터 부족 문제 분석
- 라벨 오류 후보 분석

### 오분류 분류 기준

- 라벨 정의가 모호함
- 복합 의도
- 문맥 부족
- 데이터 부족
- 표현 다양성 부족
- 키워드 편향
- 실제 라벨 오류
- 모델 한계

### 주의

모델이 틀렸다고 바로 데이터를 추가하지 않는다.

먼저 다음을 구분한다.

- 모델 오류
- 데이터 오류
- 라벨 체계 오류
- 평가 문장 오류

### 완료 조건

- `docs/evaluation-report.md`
- `docs/error-analysis.md`
- 모델 개선 우선순위
- v2 학습 필요 여부 결정

---

## Phase 8. 모델 추론 API

### 목표

학습된 모델을 애플리케이션에서 호출할 수 있게 한다.

### API 후보

POST `/predict`

입력:

{
  "text": "결제했는데 주문 내역이 없습니다."
}

출력:

{
  "label": "payment",
  "label_name": "결제",
  "confidence": 0.94,
  "scores": [
    {"label": "payment", "score": 0.94},
    {"label": "technical_issue", "score": 0.03}
  ],
  "model_version": "v1"
}

### 요구사항

- 입력 검증
- 빈 문장 거부
- 최대 길이 제한
- 모델 최초 1회 로딩
- 추론 중 매번 모델을 재로드하지 않기
- CPU/MPS/CUDA Device 선택
- 오류 응답 표준화
- 모델 버전 반환
- Health Check
- API 테스트

### 완료 조건

- 로컬 API 실행 가능
- 테스트 통과
- 샘플 요청 성공
- 실행 방법 README 작성

---

## Phase 9. Model Lab 화면

### 목표

운영용 CS 대시보드가 아니라 모델 실험 결과를 확인하는 최소 화면을 만든다.

### 기능

1. 문의 입력 및 예측
2. 전체 카테고리별 확률 표시
3. 사용 모델 버전 표시
4. 학습 지표 표시
5. Confusion Matrix 표시
6. 오분류 사례 조회
7. Baseline과 Fine-tuned 모델 비교

### 제외

- 고객관리
- 상담 배정
- 답변 발송
- 운영 통계
- 권한 관리
- 외부 채널 연동

### 완료 조건

- API 연결
- 모델 추론 결과 표시
- 평가 결과 시각화
- 로컬 실행 가능

---

## Phase 10. Hugging Face Hub 등록

### 목표

파인튜닝 모델을 재사용 가능한 형태로 관리한다.

### 사용자 확인 후 수행

- Private Model Repository 생성
- Model Weight 업로드
- Tokenizer 업로드
- Config 업로드
- Label Mapping 업로드
- Model Card 작성
- 학습 데이터 설명
- 평가 지표 작성
- 사용 목적과 한계 작성
- 추론 예제 작성

### Model Card 필수 내용

- Base Model
- Model Description
- Intended Use
- Out-of-scope Use
- Dataset
- Label Definition
- Training Procedure
- Hyperparameters
- Evaluation Results
- Limitations
- Ethical Considerations
- Usage Example
- License

실제 데이터나 개인정보가 포함되지 않았는지 확인한 뒤 업로드한다.

---

## Phase 11. 소형 LLM LoRA 확장

이 Phase는 분류 모델 프로젝트가 완료된 뒤 별도 승인 후 진행한다.

### 목표

소형 생성형 모델을 CS 구조화 분석 작업에 맞게 SFT 및 LoRA 파인튜닝한다.

### 1차 후보 모델

- `Qwen/Qwen3-0.6B`

작은 모델로 전체 학습 파이프라인을 먼저 경험한다.

처음부터 Gemma 4 E4B를 학습하지 않는다. 0.6B 모델에서 다음 절차가 완성된 후 더 큰 모델을 검토한다.

### 학습 작업

입력:

"결제됐는데 주문 내역이 없어요."

출력:

{
  "category": "payment",
  "subcategory": "order_not_created",
  "urgency": "high",
  "summary": "결제 승인 후 주문 내역이 생성되지 않은 문의",
  "required_information": [
    "결제번호",
    "결제시각",
    "고객 계정"
  ]
}

### 기술

- TRL `SFTTrainer`
- PEFT
- LoRA
- Prompt-Completion 또는 Conversational Dataset
- Adapter 저장
- Base Model + Adapter 추론
- 필요 시 Adapter 병합
- 필요 시 GGUF 변환
- Ollama 또는 llama.cpp 추론

### 평가

- JSON 형식 준수율
- 카테고리 정확도
- 필수 필드 누락률
- 잘못된 필드 생성률
- 파인튜닝 전후 비교
- Prompt-only 모델과 비교
- 추론 속도와 메모리

### 중요

GGUF는 주로 로컬 추론용으로 사용한다.

가능하면 원본 Hugging Face Checkpoint 또는 학습 가능한 정밀도의 모델에 LoRA를 적용하고, 학습 완료 후 필요할 때 병합·양자화·GGUF 변환을 진행한다.

---

# 7. 품질 관리 기준

## 코드 품질

- 설정값을 코드에 하드코딩하지 않는다.
- Label Mapping은 단일 설정 파일에서 관리한다.
- 경로는 환경별로 설정 가능하게 한다.
- Random Seed를 고정한다.
- 함수와 모듈의 역할을 분리한다.
- 핵심 전처리와 라벨 매핑에 테스트를 작성한다.
- 학습 결과물을 Git에 무분별하게 포함하지 않는다.

## 데이터 품질

- 개인정보 제거
- 중복 제거
- 라벨 분포 확인
- 경계 사례 검수
- Test 데이터 독립성 유지
- 인공 데이터 여부 기록
- 데이터셋 버전 기록

## 모델 품질

- Accuracy만 사용하지 않기
- Baseline과 비교
- 클래스별 성능 확인
- 오분류 분석
- 낮은 확률이면 자동 확정하지 않기
- 모델 버전 기록
- 학습 환경과 라이브러리 버전 기록

---

# 8. 사용자에게 설명해야 하는 개념

각 단계에서 사용자가 다음 개념을 이해할 수 있도록 짧고 실용적으로 설명한다.

- 사전학습 모델
- 파인튜닝 모델
- 토크나이저
- Token ID
- Attention Mask
- Classification Head
- Train/Validation/Test
- Epoch
- Batch Size
- Learning Rate
- Loss
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Overfitting
- Checkpoint
- Full Fine-tuning
- PEFT
- LoRA
- Adapter
- Quantization
- Safetensors
- GGUF

설명은 현재 단계에 필요한 개념만 제공한다.

---

# 9. 최종 산출물

프로젝트가 완료되면 다음 형태가 되어야 한다.

1. 한국어 CS 문의 분류 데이터셋
2. 라벨링 가이드
3. Baseline 모델
4. Fine-tuned Transformer 모델
5. 학습 및 평가 코드
6. 성능 비교 보고서
7. Confusion Matrix
8. 오분류 분석 보고서
9. 로컬 추론 스크립트
10. FastAPI 추론 API
11. 최소 Model Lab
12. Hugging Face Model Card
13. 선택적으로 소형 LLM LoRA Adapter

---

# 10. 지금 수행할 작업

지금은 Phase 0만 수행하라.

1. 현재 저장소와 작업 디렉터리를 확인한다.
2. 프로젝트 지침 파일을 확인한다.
3. 기존 파일과 Git 변경사항을 확인한다.
4. Python과 하드웨어 환경을 점검한다.
5. 이 프로젝트를 진행하기 위한 현재 환경의 적합성을 평가한다.
6. 기존 작업물을 훼손하지 않고 사용할 프로젝트 루트를 제안한다.
7. 환경에서 알 수 없는 필수 정보만 사용자에게 질문한다.
8. Phase 0 결과를 정리하고 멈춘다.

아직 다음 작업은 하지 마라.

- 전체 프로젝트 구현
- 데이터셋 대량 생성
- 모델 다운로드
- 패키지 대량 설치
- 모델 학습
- Hugging Face Repository 생성
- 외부 서비스 Push
- 관리자 대시보드 제작

Phase 0 결과를 사용자에게 보고하고 명시적인 진행 확인을 기다려라.
# 모델 선정 기록 v1

조사일: 2026-07-16  
용도: 한국어 고객 문의 7개 단일 라벨 분류 학습

## 결론

v1의 기본 모델은 **`klue/roberta-base`**로 선택한다. 한국어 NLU 벤치마크를 만든
KLUE Benchmark 조직이 공개한 base급 한국어 사전학습 모델이며, 이번 프로젝트의 목적에 맞게
Hugging Face의 표준 분류 파인튜닝 흐름을 가장 직접적으로 익힐 수 있다.

단, 공개된 KLUE 프로젝트의 라이선스는 CC BY-SA 4.0이다. 이 프로젝트에서는 교육·로컬 실험과
검수된 Private Hub 업로드만 범위에 넣는다. 외부 배포, 상용 서비스 또는 모델 재배포 전에는
파생 모델과 배포물에 적용될 라이선스 의무를 별도 법무 검토한다. 이 문서는 법률 자문이 아니다.

## 후보 비교

| 항목 | `klue/roberta-base` (선택) | `beomi/KcELECTRA-base` (대안) |
|---|---|---|
| 제작 주체·목적 | KLUE Benchmark의 한국어 NLU용 RoBERTa | Beomi의 한국어 댓글·구어체/노이즈 텍스트용 ELECTRA |
| 언어·프레임워크 | 한국어, Transformers, PyTorch | 한국어 중심(영어 태그 포함), Transformers, PyTorch |
| 규모 | Hugging Face 표기 약 0.1B 파라미터 | base급 12-layer ELECTRA; 카드 표기 모델 파일 약 475MB |
| 최대 입력 길이 | tokenizer 설정 514, 모델은 512 위치 임베딩; v1은 128 token 사용 | 설정상 512 position embeddings; v1은 128 token 사용 |
| 안전한 가중치 형식 | `model.safetensors` 제공(약 443MB) | safetensors 제공 |
| 라이선스 | KLUE 공개 프로젝트: CC BY-SA 4.0 | Hugging Face 카드: MIT |
| 강점 | 한국어 NLU 기준과 문서화가 명확하며 표준 RoBERTa 분류 예제를 학습하기 좋음 | 사용자 생성·비정형 한국어에 초점을 둔 사전학습 데이터 |
| 주의점 | 외부 배포 전 CC BY-SA 의무 확인 필요 | 카드의 권장 의존성은 오래된 버전이라 최신 환경 호환을 실험으로 확인해야 함 |

`snunlp/KR-ELECTRA-discriminator`도 한국어 base급 후보로 조사했으나, 모델 카드에서 라이선스
표기가 명확하지 않아 v1 후보에서는 제외한다.

## 선택 근거와 범위

- 한국어 CS 문장은 한국어 이해와 단문 의도 분류가 핵심이므로 생성형 LLM이 아닌 encoder 분류 모델을 사용한다.
- 약 140개로 시작하는 소규모 데이터에서는 대형 모델보다 재현 가능한 baseline·데이터 검수가 우선이다.
- `max_length=128`은 초기 CS 문의에 충분한 출발값이다. Phase 4에서 실제 토큰 길이의 95/99 분위수를 확인해 유지·조정한다.
- 모델은 아직 다운로드하지 않는다. Phase 6에서 정확한 commit revision을 조회·기록하고 그 revision으로 학습을 고정한다.
- 가중치는 pickle `.bin` 대신 safetensors를 우선 로드한다.

## 라이선스·접근성 확인 결과

- `klue/roberta-base`는 공개 접근이 가능하며 Hugging Face에서 Transformers, PyTorch, Korean,
  safetensors 태그와 약 0.1B 파라미터를 표시한다.
- 파일 목록에는 안전한 `model.safetensors`와 기존 `pytorch_model.bin`이 함께 있다. 학습에서는
  기본 로더가 safetensors를 선택하도록 하고, 가능한 경우 `use_safetensors=True`를 사용한다.
- KLUE GitHub 저장소는 프로젝트와 PLM을 CC BY-SA 4.0으로 공개한다. 실제 업로드 전에는
  해당 revision의 모델 카드·원본 라이선스·파생물 조건을 다시 확인한다.
- KcELECTRA 모델 카드는 MIT를 표시하지만, v1에서 모델을 변경할 경우 tokenizer·사전학습 도메인·
  결과를 처음부터 다시 비교한다.

## 출처

- [Hugging Face: klue/roberta-base](https://huggingface.co/klue/roberta-base)
- [Hugging Face: klue/roberta-base 파일 목록](https://huggingface.co/klue/roberta-base/tree/main)
- [KLUE Benchmark GitHub 및 라이선스](https://github.com/KLUE-benchmark/KLUE)
- [Hugging Face: beomi/KcELECTRA-base](https://huggingface.co/beomi/KcELECTRA-base)


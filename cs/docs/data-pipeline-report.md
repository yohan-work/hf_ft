# 데이터 파이프라인 실행 보고서 v1

실행 명령: `uv run python scripts/prepare_data.py`  
Seed: `42`  
Tokenizer: `klue/roberta-base` (`main`)  
Max length: `128`

## 재현 가능한 분할

| Split | 전체 | 라벨별 |
|---|---:|---:|
| Train | 98 | 각 14개 |
| Validation | 21 | 각 3개 |
| Test | 21 | 각 3개 |

`StratifiedShuffleSplit`을 seed 42로 두 번 적용해 70/15/15 비율을 만든다. 같은 원본과
설정이면 같은 행이 같은 split에 배치된다.

## 누수 검사

- split 간 정규화 완전 중복: 0개
- split 간 문자 bigram Jaccard 유사도 0.60 이상: 0개
- 결과: 통과

이 검사는 표면 중복과 높은 템플릿 유사도를 찾는 1차 검사다. 인공 데이터의 의미적 유사성과
실제 운영 데이터의 파생 관계는 이후 데이터 확장 때 다시 검수한다.

## 토큰화 결과

| Split | 최소 token | 최대 token | p95 | 128에서 잘리는 문장 |
|---|---:|---:|---:|---:|
| Train | 9 | 31 | 25 | 0 |
| Validation | 11 | 23 | 18 | 0 |
| Test | 13 | 25 | 18 | 0 |

따라서 v1의 `max_length=128`은 충분하다. 입력을 128 token으로 고정하는 것은 모든 문장을
억지로 128 token으로 늘린다는 뜻이 아니다. 짧은 문장은 padding으로 길이를 맞추고,
긴 문장만 잘라낸다.

## 생성 산출물

- `data/processed/train.csv`, `validation.csv`, `test.csv`
- `data/processed/leakage-report.json`
- `data/processed/data-summary.json`
- `data/processed/hf_dataset/`: `input_ids`, `attention_mask`, `label_id`가 포함된 Hugging Face DatasetDict

처리 산출물은 재생성 가능하므로 Git에서 제외한다.

# graphrag-agent

## 설정
- Python 3.10+. `python -m venv .venv` 후 `.venv/bin/pip install -r requirements.txt` (Windows는 `.venv\Scripts\pip`).
- 기본 LLM은 로컬 ollama(`ollama pull qwen2.5-coder:7b`). OpenAI로 바꾸려면 `LLM_BACKEND=openai`.

## 환경변수 (이름만)
- `LLM_BACKEND` (`ollama` 기본 | `openai`), `LLM_MODEL` (기본 `qwen2.5-coder:7b`), `OPENAI_API_KEY` (openai 백엔드일 때만).
- `LANGCHAIN_TRACING_V2`와 LangSmith API 키는 트레이싱이 필요할 때만 설정한다.
- `.env` 파일은 없고, 값은 셸 환경변수로만 주입한다. 키를 코드나 로그에 쓰지 않는다.

## 실행
- `python run.py` (질문 하나), `--demo`, `--interactive`, `--m4`, `--abc -n <반복수>`.
- 평가: `python eval/run_eval.py 20` (N=20). 로컬 LLM 호출이라 느리고 비용이 든다.

## 테스트와 린트
- 테스트 디렉터리와 린트 설정이 없다. 변경 검증은 `python run.py --demo`로 한다 (자가수정 회복과 함정 거부 3케이스).

## 주의사항
- 검증(validate)은 스키마만 보는 결정론 로직이다. 교정은 LLM 재계획이 하며, 검증기에 자동 repair를 넣지 않는다 (핵심 설계 결정).
- `planner.py`의 결정론 폴백은 같은 시도 안의 안전망이다. validate 거부 재시도 루프(`max_retries`, 기본 5)와 섞지 않는다.
- 채점은 결정론이다. `eval/`에 LLM judge를 넣지 않는다.
- `eval/*.log`, `eval/valid_ne_correct_cases.json`은 README 수치의 근거 산출물이므로 임의로 재생성하거나 덮어쓰지 않는다.
- `kg_data.py`의 온톨로지(노드타입, 관계, 엔티티)를 바꾸면 `eval/eval_set.py`의 gold와 README 수치가 달라진다. 같이 확인한다.
- `HANDOFF.md`, `PLAN.md`, `questions.md`, 논문 PDF는 `.gitignore` 대상이다. 공개 repo에 추가하지 않는다.
- 논문에 쓴 KEPCO 전력 데이터는 비공개다. 이 repo에 실데이터를 넣지 않는다.

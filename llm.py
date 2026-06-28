"""교체형 LLM 백엔드.

환경변수로 백엔드/모델을 바꾼다.
  LLM_BACKEND = ollama (기본) | openai
  LLM_MODEL   = qwen2.5-coder:7b (기본)

ollama는 OpenAI 호환 엔드포인트(http://localhost:11434/v1)를 제공하므로
openai 클라이언트를 base_url만 바꿔 그대로 쓴다. 백엔드가 바뀌어도 호출부는 동일.
"""
import os
from openai import OpenAI


class LLMError(RuntimeError):
    """LLM 백엔드 호출 실패(연결 불가/타임아웃 등). 호출부는 이걸 잡아 폴백한다."""


def _client_and_model():
    backend = os.environ.get("LLM_BACKEND", "ollama")
    model = os.environ.get("LLM_MODEL", "qwen2.5-coder:7b")
    if backend == "ollama":
        client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    elif backend == "openai":
        client = OpenAI()  # OPENAI_API_KEY 사용
    else:
        raise LLMError(f"unknown LLM_BACKEND: {backend}")
    return client, model


def chat(messages, temperature=0.0):
    """messages(OpenAI 형식) -> 응답 텍스트. 실패 시 LLMError."""
    client, model = _client_and_model()
    try:
        resp = client.chat.completions.create(
            model=model, messages=messages, temperature=temperature, timeout=60,
        )
    except Exception as e:  # 연결/모델/타임아웃 등 모두 폴백 신호로 변환
        raise LLMError(str(e)) from e
    return resp.choices[0].message.content or ""


def backend_info():
    return os.environ.get("LLM_BACKEND", "ollama"), os.environ.get("LLM_MODEL", "qwen2.5-coder:7b")

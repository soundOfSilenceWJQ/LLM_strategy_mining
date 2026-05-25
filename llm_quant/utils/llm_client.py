"""
LLM 客户端封装。
默认优先兼容火山 Ark（OpenAI-compatible API），同时保留 Anthropic 接口兼容性。
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from typing import Optional
from openai import OpenAI
try:
    import anthropic
except ImportError:
    anthropic = None
from llm_quant.config import ANTHROPIC_API_KEY, LLM_MODEL, LLM_MAX_TOKENS, LLM_TEMPERATURE


ARK_API_KEY = "ark-bd1a4194-8dc8-4446-850f-cf4934797f1c-9389a"
ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
ARK_MODEL = "doubao-seed-2-0-mini-260428"


class LLMClient:
    """统一 LLM 封装，支持单轮与多轮对话。"""

    def __init__(self, api_key: Optional[str] = None):
        key = api_key or os.environ.get("ARK_API_KEY", "") or ANTHROPIC_API_KEY or ARK_API_KEY
        if not key:
            raise ValueError(
                "请设置可用的 API Key。"
            )
        self._api_key = key
        if key.startswith("ark-"):
            self._backend = "ark"
            self._default_model = ARK_MODEL
            self._client = OpenAI(base_url=ARK_BASE_URL, api_key=key)
        else:
            if anthropic is None:
                raise ValueError("未安装 anthropic 包，且当前 API Key 不是 ark- 前缀，无法初始化 LLM。")
            self._backend = "anthropic"
            self._default_model = LLM_MODEL
            self._client = anthropic.Anthropic(api_key=key)

    def _resolve_model(self, model: str) -> str:
        if self._backend == "ark" and model == LLM_MODEL:
            return self._default_model
        return model

    # ── 核心调用 ──────────────────────────────────────────
    def chat(
        self,
        user_message: str,
        system_prompt: str = "",
        model: str = LLM_MODEL,
        max_tokens: int = LLM_MAX_TOKENS,
        temperature: float = LLM_TEMPERATURE,
    ) -> str:
        """单轮对话，返回助手回复文本。"""
        model = self._resolve_model(model)
        if self._backend == "ark":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_message})
            response = self._client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""

        kwargs: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": user_message}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = self._client.messages.create(**kwargs)
        return response.content[0].text

    def multi_turn(
        self,
        messages: list[dict],
        system_prompt: str = "",
        model: str = LLM_MODEL,
        max_tokens: int = LLM_MAX_TOKENS,
        temperature: float = LLM_TEMPERATURE,
    ) -> str:
        """多轮对话，messages 格式: [{"role":"user","content":"..."}, ...]"""
        model = self._resolve_model(model)
        if self._backend == "ark":
            final_messages = list(messages)
            if system_prompt:
                final_messages = [{"role": "system", "content": system_prompt}] + final_messages
            response = self._client.chat.completions.create(
                model=model,
                messages=final_messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""

        kwargs: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = self._client.messages.create(**kwargs)
        return response.content[0].text

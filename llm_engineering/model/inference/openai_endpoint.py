from __future__ import annotations

from typing import Any, Dict, Optional

from loguru import logger
from openai import OpenAI

from llm_engineering.domain.inference import Inference
from llm_engineering.settings import settings


class LLMInferenceOpenAIEndpoint(Inference):
    """Inference against a local OpenAI-compatible server (llama.cpp on :8001)."""

    def __init__(
        self,
        model_id: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        default_payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        resolved_base_url = base_url or settings.OPENAI_BASE_URL
        if not resolved_base_url:
            raise ValueError("OPENAI_BASE_URL must be set for local LLM inference.")

        self.client = OpenAI(
            base_url=resolved_base_url,
            api_key=api_key or settings.OPENAI_API_KEY or "not-needed",
        )
        self.model_id = model_id or settings.LOCAL_LLM_MODEL_ID
        self.payload = default_payload if default_payload else self._default_payload()

    def _default_payload(self) -> Dict[str, Any]:
        return {
            "inputs": "How is the weather?",
            "parameters": {
                "max_new_tokens": settings.MAX_NEW_TOKENS_INFERENCE,
                "top_p": settings.TOP_P_INFERENCE,
                "temperature": settings.TEMPERATURE_INFERENCE,
                "return_full_text": False,
            },
        }

    def set_payload(self, inputs: str, parameters: Optional[Dict[str, Any]] = None) -> None:
        self.payload["inputs"] = inputs
        if parameters:
            self.payload["parameters"].update(parameters)

    def inference(self) -> list[Dict[str, Any]]:
        params = self.payload["parameters"]
        try:
            logger.info("Local OpenAI-compatible inference request sent.")
            completion = self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": self.payload["inputs"]}],
                temperature=float(params.get("temperature", settings.TEMPERATURE_INFERENCE)),
                top_p=float(params.get("top_p", settings.TOP_P_INFERENCE)),
                max_tokens=int(params.get("max_new_tokens", settings.MAX_NEW_TOKENS_INFERENCE)),
            )
            text = completion.choices[0].message.content or ""
            return [{"generated_text": text}]
        except Exception:
            logger.exception("Local OpenAI-compatible inference failed.")
            raise

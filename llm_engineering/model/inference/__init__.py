from llm_engineering.domain.inference import Inference
from llm_engineering.settings import settings

from .inference import LLMInferenceSagemakerEndpoint
from .openai_endpoint import LLMInferenceOpenAIEndpoint
from .run import InferenceExecutor


def build_llm_inference() -> Inference:
    """Return the configured LLM backend for RAG / ml_service."""
    if settings.USE_LOCAL_LLM_INFERENCE:
        return LLMInferenceOpenAIEndpoint()
    return LLMInferenceSagemakerEndpoint(
        endpoint_name=settings.SAGEMAKER_ENDPOINT_INFERENCE,
        inference_component_name=None,
    )


__all__ = [
    "LLMInferenceOpenAIEndpoint",
    "LLMInferenceSagemakerEndpoint",
    "InferenceExecutor",
    "build_llm_inference",
]

from loguru import logger

from llm_engineering.model.inference import InferenceExecutor, build_llm_inference
from llm_engineering.settings import settings

if __name__ == "__main__":
    text = "Write me a post about AWS SageMaker inference endpoints."
    logger.info(f"Running inference for text: '{text}'")
    logger.info(
        f"Backend: {'local' if settings.USE_LOCAL_LLM_INFERENCE else 'sagemaker'} "
        f"(model={settings.LOCAL_LLM_MODEL_ID if settings.USE_LOCAL_LLM_INFERENCE else settings.SAGEMAKER_ENDPOINT_INFERENCE})"
    )
    llm = build_llm_inference()
    answer = InferenceExecutor(llm, text).execute()

    logger.info(f"Answer: '{answer}'")

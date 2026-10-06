"""Local fine-tuning entry point (replaces SageMaker estimator for RTX 3090 workflows)."""

from pathlib import Path

from loguru import logger


def _import_finetune():
    """Import finetune.py directly so this entry point can run standalone."""
    import importlib.util
    import sys
    from pathlib import Path

    finetune_path = Path(__file__).parent / "finetune.py"
    spec = importlib.util.spec_from_file_location("_run_local_finetune", finetune_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_run_local_finetune"] = module
    spec.loader.exec_module(module)
    return module.finetune, module.inference, module.save_model


def _bypass_unsloth_model_name_mapping() -> None:
    """Force Unsloth to use the exact model id we provide.

    This avoids stalls when hf-mirror cannot serve Unsloth's pre-quantized
    variants (e.g. ``unsloth/Qwen3-8B-unsloth-bnb-4bit``). Unsloth will
    quantize the original weights locally instead.
    """
    try:
        import unsloth.models.loader as loader
        import unsloth.models.loader_utils as loader_utils

        _identity = lambda model_name, *args, **kwargs: str(model_name)
        if hasattr(loader_utils, "get_model_name"):
            loader_utils.get_model_name = _identity
        if hasattr(loader, "get_model_name"):
            loader.get_model_name = _identity
    except Exception:
        pass


def run_finetuning_locally(
    finetuning_type: str = "sft",
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 1,
    learning_rate: float = 3e-4,
    gradient_accumulation_steps: int = 16,
    dataset_huggingface_workspace: str = "mlabonne",
    is_dummy: bool = False,
    base_model: str = "Qwen/Qwen3-32B",
    max_seq_length: int = 1024,
    load_in_4bit: bool = True,
    use_local_datasets: bool = True,
    local_sft_dataset_path: str = "data/artifacts/instruct_datasets.json",
    local_dpo_dataset_path: str = "data/artifacts/preference_datasets.json",
    mix_alpaca_dataset: bool = False,
    report_to: str = "none",
    push_to_hub: bool = False,
    model_output_huggingface_workspace: str | None = None,
    output_dir: str = "data/training_output",
    sft_merged_model_dir: str | None = None,
) -> None:
    try:
        finetune, inference, save_model = _import_finetune()
    except ImportError as exc:
        raise ImportError(
            "Training dependencies are not installed. "
            "Install them with: pip install -r llm_engineering/model/finetuning/requirements.txt"
        ) from exc

    _bypass_unsloth_model_name_mapping()

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    if sft_merged_model_dir is None:
        sft_merged_model_dir = str(output_root / "sft_merged")

    logger.info(
        "Local fine-tuning: type={}, base_model={}, 4bit={}, max_seq_length={}, dummy={}",
        finetuning_type,
        base_model,
        load_in_4bit,
        max_seq_length,
        is_dummy,
    )

    if finetuning_type == "sft":
        output_dir_sft = output_root / "output_sft"
        model, tokenizer = finetune(
            finetuning_type="sft",
            model_name=base_model,
            output_dir=str(output_dir_sft),
            dataset_huggingface_workspace=dataset_huggingface_workspace,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            use_local_datasets=use_local_datasets,
            local_sft_dataset_path=local_sft_dataset_path,
            mix_alpaca_dataset=mix_alpaca_dataset,
            report_to=report_to,
            is_dummy=is_dummy,
        )
        inference(model, tokenizer)
        save_model(model, tokenizer, sft_merged_model_dir, push_to_hub=push_to_hub, repo_id=None)
        logger.info("SFT complete. Merged model saved to {}", sft_merged_model_dir)
    elif finetuning_type == "dpo":
        dpo_base = Path(sft_merged_model_dir)
        if not dpo_base.exists():
            raise FileNotFoundError(
                f"SFT merged model not found at '{dpo_base}'. Run SFT locally before DPO."
            )

        output_dir_dpo = output_root / "output_dpo"
        dpo_merged_dir = output_root / "dpo_merged"
        model, tokenizer = finetune(
            finetuning_type="dpo",
            model_name=str(dpo_base),
            output_dir=str(output_dir_dpo),
            dataset_huggingface_workspace=dataset_huggingface_workspace,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            num_train_epochs=1 if is_dummy else num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=2e-6,
            use_local_datasets=use_local_datasets,
            local_dpo_dataset_path=local_dpo_dataset_path,
            report_to=report_to,
            is_dummy=is_dummy,
        )
        inference(model, tokenizer)
        repo_id = None
        if push_to_hub and model_output_huggingface_workspace:
            repo_id = f"{model_output_huggingface_workspace}/TwinQwen3-DPO"
        save_model(model, tokenizer, str(dpo_merged_dir), push_to_hub=push_to_hub, repo_id=repo_id)
        logger.info("DPO complete. Merged model saved to {}", dpo_merged_dir)
    else:
        raise ValueError("finetuning_type must be 'sft' or 'dpo'")


if __name__ == "__main__":
    import argparse

    # Allow running standalone without the full project's runtime dependencies
    # (MongoDB/Qdrant/etc.). Falls back to the same defaults as Settings.
    try:
        from llm_engineering.settings import settings
    except ImportError:

        class _StandaloneTrainingSettings:  # pragma: no cover
            TRAINING_BASE_MODEL = "Qwen/Qwen3-32B"
            TRAINING_MAX_SEQ_LENGTH = 1024
            TRAINING_OUTPUT_DIR = "data/training_output"
            LOCAL_SFT_DATASET_PATH = "data/artifacts/instruct_datasets.json"
            LOCAL_DPO_DATASET_PATH = "data/artifacts/preference_datasets.json"
            TRAINING_REPORT_TO = "none"

        settings = _StandaloneTrainingSettings()

    parser = argparse.ArgumentParser(description="Run QLoRA fine-tuning locally (SFT / DPO).")
    parser.add_argument("--finetuning_type", choices=["sft", "dpo"], default="sft")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--is_dummy", action="store_true")
    parser.add_argument("--base_model", type=str, default=settings.TRAINING_BASE_MODEL)
    parser.add_argument("--max_seq_length", type=int, default=settings.TRAINING_MAX_SEQ_LENGTH)
    parser.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_local_datasets", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local_sft_dataset_path", type=str, default=settings.LOCAL_SFT_DATASET_PATH)
    parser.add_argument("--local_dpo_dataset_path", type=str, default=settings.LOCAL_DPO_DATASET_PATH)
    parser.add_argument("--mix_alpaca_dataset", action="store_true")
    parser.add_argument("--report_to", type=str, default=settings.TRAINING_REPORT_TO)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--output_dir", type=str, default=settings.TRAINING_OUTPUT_DIR)
    parser.add_argument("--sft_merged_model_dir", type=str, default=None)
    parser.add_argument("--dataset_huggingface_workspace", type=str, default="mlabonne")

    args = parser.parse_args()
    run_finetuning_locally(
        finetuning_type=args.finetuning_type,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        dataset_huggingface_workspace=args.dataset_huggingface_workspace,
        is_dummy=args.is_dummy,
        base_model=args.base_model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        use_local_datasets=args.use_local_datasets,
        local_sft_dataset_path=args.local_sft_dataset_path,
        local_dpo_dataset_path=args.local_dpo_dataset_path,
        mix_alpaca_dataset=args.mix_alpaca_dataset,
        report_to=args.report_to,
        push_to_hub=args.push_to_hub,
        output_dir=args.output_dir,
        sft_merged_model_dir=args.sft_merged_model_dir,
    )

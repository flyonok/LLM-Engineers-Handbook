from zenml import step

from llm_engineering.model.finetuning.run_local import run_finetuning_locally
from llm_engineering.model.finetuning.sagemaker import run_finetuning_on_sagemaker
from llm_engineering.settings import settings


@step
def train(
    finetuning_type: str,
    num_train_epochs: int,
    per_device_train_batch_size: int,
    learning_rate: float,
    dataset_huggingface_workspace: str = "mlabonne",
    is_dummy: bool = False,
    local: bool = False,
    base_model: str = "Qwen/Qwen3-32B",
    max_seq_length: int = 1024,
    load_in_4bit: bool = True,
    gradient_accumulation_steps: int = 16,
    use_local_datasets: bool = True,
    local_sft_dataset_path: str = "data/artifacts/instruct_datasets.json",
    local_dpo_dataset_path: str = "data/artifacts/preference_datasets.json",
    mix_alpaca_dataset: bool = False,
    report_to: str = "none",
    training_output_dir: str = "data/training_output",
) -> None:
    use_local = local or settings.LOCAL_TRAINING

    if use_local:
        run_finetuning_locally(
            finetuning_type=finetuning_type,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            learning_rate=learning_rate,
            gradient_accumulation_steps=gradient_accumulation_steps,
            dataset_huggingface_workspace=dataset_huggingface_workspace,
            is_dummy=is_dummy,
            base_model=base_model,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            use_local_datasets=use_local_datasets,
            local_sft_dataset_path=local_sft_dataset_path,
            local_dpo_dataset_path=local_dpo_dataset_path,
            mix_alpaca_dataset=mix_alpaca_dataset,
            report_to=report_to,
            output_dir=training_output_dir,
        )
    else:
        run_finetuning_on_sagemaker(
            finetuning_type=finetuning_type,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            learning_rate=learning_rate,
            dataset_huggingface_workspace=dataset_huggingface_workspace,
            is_dummy=is_dummy,
            base_model=base_model,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
        )

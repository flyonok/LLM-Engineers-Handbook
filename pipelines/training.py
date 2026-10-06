from zenml import pipeline

from steps import training as training_steps


@pipeline
def training(
    finetuning_type: str = "sft",
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 2,
    learning_rate: float = 3e-4,
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
    training_steps.train(
        finetuning_type=finetuning_type,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        dataset_huggingface_workspace=dataset_huggingface_workspace,
        is_dummy=is_dummy,
        local=local,
        base_model=base_model,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        gradient_accumulation_steps=gradient_accumulation_steps,
        use_local_datasets=use_local_datasets,
        local_sft_dataset_path=local_sft_dataset_path,
        local_dpo_dataset_path=local_dpo_dataset_path,
        mix_alpaca_dataset=mix_alpaca_dataset,
        report_to=report_to,
        training_output_dir=training_output_dir,
    )

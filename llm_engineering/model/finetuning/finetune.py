import argparse
import json
import os
from pathlib import Path

from unsloth import PatchDPOTrainer

PatchDPOTrainer()

from typing import Any, List, Literal, Optional  # noqa: E402

import torch  # noqa
from datasets import Dataset, concatenate_datasets, load_dataset  # noqa: E402
from huggingface_hub import HfApi  # noqa: E402
from huggingface_hub.utils import RepositoryNotFoundError  # noqa: E402
from transformers import TextStreamer, TrainingArguments  # noqa: E402
from trl import DPOConfig, DPOTrainer, SFTTrainer  # noqa: E402
from unsloth import FastLanguageModel, is_bfloat16_supported  # noqa: E402
from unsloth.chat_templates import get_chat_template  # noqa: E402

alpaca_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""


def load_instruct_dataset_from_artifacts_json(path: str) -> Dataset:
    with Path(path).open(encoding="utf-8") as file:
        data = json.load(file)

    instructions: list[str] = []
    outputs: list[str] = []
    for split_name in ("train", "test"):
        split = data.get(split_name)
        if not isinstance(split, dict):
            continue
        for category_blob in split.values():
            for sample in category_blob.get("samples", []):
                instructions.append(sample["instruction"])
                outputs.append(sample["answer"])

    if not instructions:
        raise ValueError(f"No instruct samples found in '{path}'")

    return Dataset.from_dict({"instruction": instructions, "output": outputs})


def load_preference_dataset_from_artifacts_json(path: str) -> Dataset:
    with Path(path).open(encoding="utf-8") as file:
        data = json.load(file)

    prompts: list[str] = []
    chosen: list[str] = []
    rejected: list[str] = []
    for split_name in ("train", "test"):
        split = data.get(split_name)
        if not isinstance(split, dict):
            continue
        for category_blob in split.values():
            for sample in category_blob.get("samples", []):
                prompts.append(sample["instruction"])
                chosen.append(sample["chosen"])
                rejected.append(sample["rejected"])

    if not prompts:
        raise ValueError(f"No preference samples found in '{path}'")

    return Dataset.from_dict({"prompt": prompts, "chosen": chosen, "rejected": rejected})


def load_sft_training_dataset(
    dataset_huggingface_workspace: str,
    use_local_datasets: bool,
    local_sft_dataset_path: Optional[str],
    mix_alpaca_dataset: bool,
) -> Dataset:
    if use_local_datasets:
        if not local_sft_dataset_path:
            raise ValueError("local_sft_dataset_path is required when use_local_datasets=True")
        dataset = load_instruct_dataset_from_artifacts_json(local_sft_dataset_path)
        if mix_alpaca_dataset:
            alpaca = load_dataset("mlabonne/FineTome-Alpaca-100k", split="train[:10000]")
            dataset = concatenate_datasets([dataset, alpaca])
        return dataset

    dataset1 = load_dataset(f"{dataset_huggingface_workspace}/llmtwin", split="train")
    dataset2 = load_dataset("mlabonne/FineTome-Alpaca-100k", split="train[:10000]")
    return concatenate_datasets([dataset1, dataset2])


def load_dpo_training_dataset(
    dataset_huggingface_workspace: str,
    use_local_datasets: bool,
    local_dpo_dataset_path: Optional[str],
) -> Dataset:
    if use_local_datasets:
        if not local_dpo_dataset_path:
            raise ValueError("local_dpo_dataset_path is required when use_local_datasets=True")
        return load_preference_dataset_from_artifacts_json(local_dpo_dataset_path)

    return load_dataset(f"{dataset_huggingface_workspace}/llmtwin-dpo", split="train")


def load_model(
    model_name: str,
    max_seq_length: int,
    load_in_4bit: bool,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: List[str],
    chat_template: str,
) -> tuple:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template=chat_template,
    )

    return model, tokenizer


def finetune(
    finetuning_type: Literal["sft", "dpo"],
    model_name: str,
    output_dir: str,
    dataset_huggingface_workspace: str,
    max_seq_length: int = 2048,
    load_in_4bit: bool = False,
    lora_rank: int = 32,
    lora_alpha: int = 32,
    lora_dropout: float = 0.0,
    target_modules: List[str] = ["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"],  # noqa: B006
    chat_template: str = "chatml",
    learning_rate: float = 3e-4,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    beta: float = 0.5,  # Only for DPO
    is_dummy: bool = True,
    use_local_datasets: bool = False,
    local_sft_dataset_path: Optional[str] = None,
    local_dpo_dataset_path: Optional[str] = None,
    mix_alpaca_dataset: bool = True,
    report_to: str = "comet_ml",
) -> tuple:
    model, tokenizer = load_model(
        model_name, max_seq_length, load_in_4bit, lora_rank, lora_alpha, lora_dropout, target_modules, chat_template
    )
    EOS_TOKEN = tokenizer.eos_token
    print(f"Setting EOS_TOKEN to {EOS_TOKEN}")  # noqa

    if is_dummy is True:
        num_train_epochs = 1
        print(f"Training in dummy mode. Setting num_train_epochs to '{num_train_epochs}'")  # noqa
        print("Training in dummy mode. Reducing dataset size to '400'.")  # noqa

    if finetuning_type == "sft":

        def format_samples_sft(examples):
            text = []
            for instruction, output in zip(examples["instruction"], examples["output"], strict=False):
                message = alpaca_template.format(instruction, output) + EOS_TOKEN
                text.append(message)

            return {"text": text}

        dataset = load_sft_training_dataset(
            dataset_huggingface_workspace,
            use_local_datasets,
            local_sft_dataset_path,
            mix_alpaca_dataset,
        )
        if is_dummy:
            try:
                dataset = dataset.select(range(400))
            except Exception:
                print("Dummy mode active. Failed to trim the dataset to 400 samples.")  # noqa
        print(f"Loaded dataset with {len(dataset)} samples.")  # noqa

        dataset = dataset.map(format_samples_sft, batched=True, remove_columns=dataset.column_names)
        dataset = dataset.train_test_split(test_size=0.05)

        print("Training dataset example:")  # noqa
        print(dataset["train"][0])  # noqa

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            dataset_num_proc=2,
            packing=True,
            args=TrainingArguments(
                learning_rate=learning_rate,
                num_train_epochs=num_train_epochs,
                per_device_train_batch_size=per_device_train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=1,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                per_device_eval_batch_size=per_device_train_batch_size,
                warmup_steps=10,
                output_dir=output_dir,
                report_to=report_to if report_to != "none" else "none",
                seed=0,
            ),
        )
    elif finetuning_type == "dpo":
        PatchDPOTrainer()

        def format_samples_dpo(example):
            example["prompt"] = alpaca_template.format(example["prompt"], "")
            example["chosen"] = example["chosen"] + EOS_TOKEN
            example["rejected"] = example["rejected"] + EOS_TOKEN

            return {"prompt": example["prompt"], "chosen": example["chosen"], "rejected": example["rejected"]}

        dataset = load_dpo_training_dataset(
            dataset_huggingface_workspace,
            use_local_datasets,
            local_dpo_dataset_path,
        )
        if is_dummy:
            try:
                dataset = dataset.select(range(400))
            except Exception:
                print("Dummy mode active. Failed to trim the dataset to 400 samples.")  # noqa
        print(f"Loaded dataset with {len(dataset)} samples.")  # noqa

        dataset = dataset.map(format_samples_dpo)
        dataset = dataset.train_test_split(test_size=0.05)

        print("Training dataset example:")  # noqa
        print(dataset["train"][0])  # noqa

        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            tokenizer=tokenizer,
            beta=beta,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            max_length=max_seq_length // 2,
            max_prompt_length=max_seq_length // 2,
            args=DPOConfig(
                learning_rate=learning_rate,
                num_train_epochs=num_train_epochs,
                per_device_train_batch_size=per_device_train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                per_device_eval_batch_size=per_device_train_batch_size,
                warmup_steps=10,
                output_dir=output_dir,
                eval_steps=0.2,
                logging_steps=1,
                report_to=report_to if report_to != "none" else "none",
                seed=0,
            ),
        )
    else:
        raise ValueError("Invalid finetuning_type. Choose 'sft' or 'dpo'.")

    trainer.train()

    return model, tokenizer


def inference(
    model: Any,
    tokenizer: Any,
    prompt: str = "Write a paragraph to introduce supervised fine-tuning.",
    max_new_tokens: int = 256,
) -> None:
    model = FastLanguageModel.for_inference(model)
    message = alpaca_template.format(prompt, "")
    inputs = tokenizer([message], return_tensors="pt").to("cuda")

    text_streamer = TextStreamer(tokenizer)
    _ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=max_new_tokens, use_cache=True)


def save_model(model: Any, tokenizer: Any, output_dir: str, push_to_hub: bool = False, repo_id: Optional[str] = None):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained_merged(output_dir, tokenizer, save_method="merged_16bit")

    if push_to_hub and repo_id:
        print(f"Saving model to '{repo_id}'")  # noqa
        model.push_to_hub_merged(repo_id, tokenizer, save_method="merged_16bit")


def check_if_huggingface_model_exists(model_id: str, default_value: str = "mlabonne/TwinLlama-3.1-8B") -> str:
    api = HfApi()

    try:
        api.model_info(model_id)
    except RepositoryNotFoundError:
        print(f"Model '{model_id}' does not exist.")  # noqa
        model_id = default_value
        print(f"Defaulting to '{model_id}'")  # noqa
        print("Train your own 'TwinLlama-3.1-8B' to avoid this behavior.")  # noqa

    return model_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--dataset_huggingface_workspace", type=str, default="mlabonne")
    parser.add_argument("--model_output_huggingface_workspace", type=str, default="mlabonne")
    parser.add_argument("--is_dummy", action="store_true", help="Reduce the dataset size for testing")
    parser.add_argument(
        "--finetuning_type",
        type=str,
        choices=["sft", "dpo"],
        default="sft",
        help="Parameter to choose the finetuning stage.",
    )
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--use_local_datasets", action="store_true")
    parser.add_argument("--local_sft_dataset_path", type=str, default=None)
    parser.add_argument("--local_dpo_dataset_path", type=str, default=None)
    parser.add_argument("--mix_alpaca_dataset", action="store_true")
    parser.add_argument("--report_to", type=str, default="comet_ml")
    parser.add_argument("--push_to_hub", action="store_true")

    parser.add_argument(
        "--output_data_dir",
        type=str,
        default=os.environ.get("SM_OUTPUT_DATA_DIR", "./data/training_output"),
    )
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "./data/training_output"))
    parser.add_argument("--n_gpus", type=str, default=os.environ.get("SM_NUM_GPUS", "1"))

    args = parser.parse_args()

    print(f"Num training epochs: '{args.num_train_epochs}'")  # noqa
    print(f"Per device train batch size: '{args.per_device_train_batch_size}'")  # noqa
    print(f"Learning rate: {args.learning_rate}")  # noqa
    print(f"Datasets will be loaded from Hugging Face workspace: '{args.dataset_huggingface_workspace}'")  # noqa
    print(f"Models will be saved to Hugging Face workspace: '{args.model_output_huggingface_workspace}'")  # noqa
    print(f"Training in dummy mode? '{args.is_dummy}'")  # noqa
    print(f"Finetuning type: '{args.finetuning_type}'")  # noqa

    print(f"Output data dir: '{args.output_data_dir}'")  # noqa
    print(f"Model dir: '{args.model_dir}'")  # noqa
    print(f"Number of GPUs: '{args.n_gpus}'")  # noqa

    if args.finetuning_type == "sft":
        print("Starting SFT training...")  # noqa
        print(f"Training from base model '{args.base_model}'")  # noqa

        output_dir_sft = Path(args.model_dir) / "output_sft"
        model, tokenizer = finetune(
            finetuning_type="sft",
            model_name=args.base_model,
            output_dir=str(output_dir_sft),
            dataset_huggingface_workspace=args.dataset_huggingface_workspace,
            max_seq_length=args.max_seq_length,
            load_in_4bit=args.load_in_4bit,
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            use_local_datasets=args.use_local_datasets,
            local_sft_dataset_path=args.local_sft_dataset_path,
            mix_alpaca_dataset=args.mix_alpaca_dataset,
            report_to=args.report_to,
            is_dummy=args.is_dummy,
        )
        inference(model, tokenizer)

        sft_output_model_repo_id = f"{args.model_output_huggingface_workspace}/TwinLlama-3.1-8B"
        save_model(
            model,
            tokenizer,
            str(Path(args.model_dir) / "model_sft"),
            push_to_hub=args.push_to_hub,
            repo_id=sft_output_model_repo_id if args.push_to_hub else None,
        )
    elif args.finetuning_type == "dpo":
        print("Starting DPO training...")  # noqa

        sft_base_model_repo_id = f"{args.model_output_huggingface_workspace}/TwinLlama-3.1-8B"
        if args.use_local_datasets and args.local_sft_dataset_path:
            sft_base_model = str(Path(args.model_dir) / "model_sft")
        else:
            sft_base_model = check_if_huggingface_model_exists(sft_base_model_repo_id)
        print(f"Training from base model '{sft_base_model}'")  # noqa

        output_dir_dpo = Path(args.model_dir) / "output_dpo"
        model, tokenizer = finetune(
            finetuning_type="dpo",
            model_name=sft_base_model,
            output_dir=str(output_dir_dpo),
            dataset_huggingface_workspace=args.dataset_huggingface_workspace,
            max_seq_length=args.max_seq_length,
            load_in_4bit=args.load_in_4bit,
            num_train_epochs=1,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=2e-6,
            use_local_datasets=args.use_local_datasets,
            local_dpo_dataset_path=args.local_dpo_dataset_path,
            report_to=args.report_to,
            is_dummy=args.is_dummy,
        )
        inference(model, tokenizer)

        dpo_output_model_repo_id = f"{args.model_output_huggingface_workspace}/TwinLlama-3.1-8B-DPO"
        save_model(
            model,
            tokenizer,
            str(Path(args.model_dir) / "model_dpo"),
            push_to_hub=args.push_to_hub,
            repo_id=dpo_output_model_repo_id if args.push_to_hub else None,
        )

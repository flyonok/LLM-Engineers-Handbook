#!/usr/bin/env python3
"""快速验证 SFT 合并模型：加载 data/training_output/sft_merged 并生成回复。

用法：
    source .venv-train/bin/activate
    python scripts/verify_sft_model.py

默认用 4bit 加载以节省显存；如需 fp16，传 --dtype bfloat16。
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


SAMPLE_PROMPTS = [
    {
        "name": "领域介绍",
        "messages": [
            {
                "role": "user",
                "content": "请用一段简洁的中文介绍什么是 LLM Twin，并说明它在 AI 应用开发中的作用。",
            }
        ],
    },
    {
        "name": "风格续写",
        "messages": [
            {
                "role": "user",
                "content": "写一段话，说明本地 QLoRA 微调相比云端训练的优势。",
            }
        ],
    },
    {
        "name": "指令遵从",
        "messages": [
            {"role": "user", "content": "列出将 HuggingFace 模型转换为 GGUF 格式的三个关键步骤。"}
        ],
    },
]


def load_model_and_tokenizer(model_dir: Path, dtype: str):
    if dtype == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model_kwargs = {"quantization_config": bnb_config, "device_map": "auto"}
    else:
        model_kwargs = {
            "torch_dtype": getattr(torch, dtype),
            "device_map": "auto",
            "low_cpu_mem_usage": True,
        }

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        **model_kwargs,
    )
    return model, tokenizer


def generate(
    model,
    tokenizer,
    messages: list[dict[str, str]],
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
):
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify merged SFT model")
    parser.add_argument(
        "--model_dir",
        type=Path,
        default=Path("data/training_output/sft_merged"),
        help="Path to merged 16bit model directory",
    )
    parser.add_argument(
        "--dtype",
        choices=["4bit", "bfloat16", "float16"],
        default="4bit",
        help="Load dtype (default: 4bit to save VRAM)",
    )
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    if not args.model_dir.exists():
        print(f"Model directory not found: {args.model_dir}", file=sys.stderr)
        return 1

    print(f"Loading model from {args.model_dir} (dtype={args.dtype}) ...")
    model, tokenizer = load_model_and_tokenizer(args.model_dir, args.dtype)
    print(f"Model loaded: {model.__class__.__name__}")
    print(f"Device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else model.device}")

    print("\n" + "=" * 60)
    for sample in SAMPLE_PROMPTS:
        print(f"\n[{sample['name']}]")
        print(f"User: {sample['messages'][0]['content']}")
        print("-" * 40)
        response = generate(
            model,
            tokenizer,
            sample["messages"],
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        print(f"Assistant: {response.strip()}")
    print("\n" + "=" * 60)
    print("验证完成。若输出流畅且与领域相关，说明 SFT 基本成功。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

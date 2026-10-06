#!/usr/bin/env python3
"""本地端点快速评估：从本地 instruct 数据集采样，调用 :8001 生成并评判。"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI


def load_local_instruct_dataset(path: str, n_samples: int | None = None):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        dataset = data
    elif isinstance(data, dict) and "samples" in data:
        dataset = data["samples"]
    elif isinstance(data, dict) and "train" in data and isinstance(data["train"], dict):
        # Format: {"train": {"articles": {"samples": [...]}}, "test": {...}}
        dataset = []
        for split in ["test", "train"]:
            articles = data[split].get("articles", {})
            samples = articles.get("samples", [])
            dataset.extend(samples)
    else:
        raise ValueError(f"Unsupported dataset format in {path}")
    if n_samples is not None:
        dataset = dataset[:n_samples]
    return dataset


def format_prompt(sample: dict) -> str:
    instruction = sample.get("instruction") or sample.get("prompt", "")
    return (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction}\n\n### Response:\n"
    )


def generate_answer(client: OpenAI, model: str, prompt: str, max_tokens: int = 256) -> str:
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            top_p=0.95,
            max_tokens=max_tokens,
        )
        return completion.choices[0].message.content or ""
    except Exception as exc:
        return f"[ERROR: {exc}]"


def judge_answer(client: OpenAI, model: str, instruction: str, answer: str) -> dict:
    prompt = f"""你是一位严谨的评判专家。请对下面这条指令的回复进行打分（1-5 分），并给出简短理由。

评分维度：
1. 准确性：回复内容是否正确、有用。
2. 风格：语言是否自然、通顺，符合中文表达习惯。

请严格按以下 JSON 格式输出，不要输出其他内容：
{{"accuracy": 整数1-5, "style": 整数1-5, "reason": "简短理由"}}

指令：{instruction}

回复：{answer}
"""
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=256,
        )
        content = completion.choices[0].message.content or "{}"
        # 尝试从回复中提取 JSON
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1:
            content = content[start : end + 1]
        return json.loads(content)
    except Exception as exc:
        return {"accuracy": None, "style": None, "reason": f"judge error: {exc}"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate local endpoint")
    parser.add_argument("--dataset", default="data/artifacts/instruct_datasets.json")
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--base_url", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8001/v1"))
    parser.add_argument("--model", default=os.environ.get("LOCAL_MODEL_NAME", "qwen3-8b-dpo-q4km"))
    parser.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", "not-needed"))
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--output", default="data/training_output/local_eval_results.json")
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    print(f"Loading dataset: {args.dataset} (first {args.n_samples} samples)")
    dataset = load_local_instruct_dataset(args.dataset, args.n_samples)
    print(f"Loaded {len(dataset)} samples")

    print(f"Generating answers via {args.base_url} (model={args.model}) ...")
    with ThreadPoolExecutor(max_workers=4) as executor:
        answers = list(
            executor.map(
                lambda sample: generate_answer(client, args.model, format_prompt(sample), args.max_tokens),
                dataset,
            )
        )

    print("Judging answers ...")
    results = []
    for sample, answer in zip(dataset, answers):
        instruction = sample.get("instruction") or sample.get("prompt", "")
        judgement = judge_answer(client, args.model, instruction, answer)
        results.append(
            {
                "instruction": instruction,
                "expected": sample.get("answer") or sample.get("output", ""),
                "generated": answer,
                "judgement": judgement,
            }
        )
        print(f"- accuracy={judgement.get('accuracy')}, style={judgement.get('style')}: {instruction[:40]}...")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {args.output}")

    valid_accuracy = [r["judgement"].get("accuracy") for r in results if isinstance(r["judgement"].get("accuracy"), int)]
    valid_style = [r["judgement"].get("style") for r in results if isinstance(r["judgement"].get("style"), int)]
    if valid_accuracy:
        print(f"Average accuracy: {sum(valid_accuracy) / len(valid_accuracy):.2f}")
    if valid_style:
        print(f"Average style: {sum(valid_style) / len(valid_style):.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Local evaluation: generate + judge via local llama.cpp OpenAI-compatible endpoint (:8001)."""

import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    evaluate_py = Path(__file__).resolve().parent / "evaluate.py"
    root_dir = Path(__file__).resolve().parents[4]

    env = os.environ.copy()
    # Defaults for local llama.cpp endpoint; override via shell env if needed.
    env.setdefault("OPENAI_BASE_URL", "http://127.0.0.1:8001/v1")
    env.setdefault("LOCAL_MODEL_NAME", "local-model")
    env.setdefault("OPENAI_API_KEY", env.get("OPENAI_API_KEY", "not-needed"))
    env.setdefault("DATASET_HUGGINGFACE_WORKSPACE", env.get("DATASET_HUGGINGFACE_WORKSPACE", "local"))
    env.setdefault("MODEL_HUGGINGFACE_WORKSPACE", env.get("MODEL_HUGGINGFACE_WORKSPACE", "local"))
    if "--dummy" in sys.argv:
        env["IS_DUMMY"] = "True"

    print(f"Running evaluation script: {evaluate_py}")
    print(f"OPENAI_BASE_URL={env['OPENAI_BASE_URL']}")
    print(f"LOCAL_MODEL_NAME={env['LOCAL_MODEL_NAME']}")

    # evaluate.py reads env vars at import time; run in a subprocess with our env.
    result = subprocess.run([sys.executable, str(evaluate_py)], env=env, cwd=root_dir)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()

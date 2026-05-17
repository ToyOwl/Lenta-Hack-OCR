# scripts/download_paddlenlp_qwen.py
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="PaddleNLP model name, e.g. Qwen/Qwen2-1.5B-Instruct or Qwen/Qwen2.5-1.5B-Instruct",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Where to save local PaddleNLP model directory",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Use float32 for CPU. Use float16 only for GPU.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[download] model_name={args.model_name}")
    print(f"[download] out_dir={out_dir}")
    print(f"[download] dtype={args.dtype}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=args.dtype)

    print("[save] tokenizer...")
    tokenizer.save_pretrained(str(out_dir))

    print("[save] model...")
    model.save_pretrained(str(out_dir))

    print("[ok] local PaddleNLP model saved:")
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
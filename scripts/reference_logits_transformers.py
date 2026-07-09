#!/usr/bin/env python3
"""Write HF Transformers causal-LM logits in detllm dump format.

The output is contiguous little-endian f32 rows:

    position 0 vocab logits | position 1 vocab logits | ...

This matches `detllm logits --dump FILE`, so the result can be checked with:

    cargo run -p xtask -- compare-logits --actual detllm.bin --reference hf.bin \
        --row-size VOCAB --rows TOKENS --min-cosine 0.999
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path


def parse_tokens(raw: str) -> list[int]:
    try:
        tokens = [int(part) for part in raw.split(",") if part]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid token list: {raw}") from exc
    if not tokens:
        raise argparse.ArgumentTypeError("token list must not be empty")
    if any(token < 0 for token in tokens):
        raise argparse.ArgumentTypeError("token IDs must be non-negative")
    return tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a little-endian f32 logits dump with HF Transformers."
    )
    parser.add_argument("--model-id", required=True, help="HF model id or local model path")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--tokens", type=parse_tokens, help="comma-separated token IDs")
    group.add_argument("--prompt", help="prompt text to tokenize with HF AutoTokenizer")
    parser.add_argument("--out", required=True, help="output .bin path")
    parser.add_argument("--device", default="cpu", help="torch device, default: cpu")
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="float32",
        help="model load dtype; logits are always written as f32",
    )
    parser.add_argument("--threads", type=int, default=1, help="torch CPU thread count")
    parser.add_argument("--expected-vocab", type=int, help="fail unless logits vocab matches")
    parser.add_argument("--expected-rows", type=int, help="fail unless row count matches")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.threads <= 0:
        raise SystemExit("--threads must be greater than zero")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "missing dependency: install torch and transformers in the reference environment"
        ) from exc

    torch.set_num_threads(args.threads)
    dtype = {
        "auto": "auto",
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    )
    model.to(args.device)
    model.eval()

    if args.tokens is not None:
        token_ids = args.tokens
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_id,
            trust_remote_code=args.trust_remote_code,
        )
        token_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
        if not token_ids:
            raise SystemExit("prompt produced no tokens")
        print(
            "tokens=" + ",".join(str(token) for token in token_ids),
            file=sys.stderr,
        )

    input_ids = torch.tensor([token_ids], dtype=torch.long, device=args.device)
    with torch.inference_mode():
        logits = model(input_ids=input_ids).logits[0].detach().cpu().to(torch.float32)

    rows, vocab = logits.shape
    if args.expected_rows is not None and rows != args.expected_rows:
        raise SystemExit(f"row count mismatch: got {rows}, expected {args.expected_rows}")
    if args.expected_vocab is not None and vocab != args.expected_vocab:
        raise SystemExit(f"vocab mismatch: got {vocab}, expected {args.expected_vocab}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as fh:
        for value in logits.reshape(-1).tolist():
            fh.write(struct.pack("<f", float(value)))

    print(
        f"wrote rows={rows} vocab={vocab} values={rows * vocab} path={out}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
from typing import Any, Dict

import torch
import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect label marginals saved by evaluation."
    )
    parser.add_argument(
        "marginal_path",
        help="Path to the saved marginals file (e.g., output from --label_marginal_out).",
    )
    parser.add_argument(
        "--config",
        dest="config_path",
        default=None,
        help="Optional config.yaml path for context (model settings, etc.).",
    )
    parser.add_argument(
        "--sentence-index",
        type=int,
        default=0,
        help="Index of the sentence to inspect (0-based).",
    )

    args = parser.parse_args()

    saved = torch.load(args.marginal_path, map_location="cpu")
    if not isinstance(saved, dict) or "marginal" not in saved:
        raise ValueError("Expected a dict with a 'marginal' entry in the saved file.")

    marginal = saved["marginal"]
    seq_len = saved.get("seq_len")

    if args.sentence_index < 0 or args.sentence_index >= marginal.shape[0]:
        raise IndexError(
            f"Sentence index {args.sentence_index} is out of range for batch size {marginal.shape[0]}."
        )

    sentence_marginal = marginal[args.sentence_index]
    if seq_len is not None:
        length = int(seq_len[args.sentence_index])
    else:
        length = sentence_marginal.shape[0] - 1

    # Spans with width = 1 reside on the first superdiagonal (start=i, end=i+1).
    width_one = sentence_marginal.diagonal(offset=1, dim1=0, dim2=1)
    width_one = width_one[:length]

    if args.config_path:
        config = load_config(args.config_path)
        print(f"Loaded config from {args.config_path}")
        model_cfg = config.get("model", {}) if isinstance(config, dict) else {}
        if model_cfg:
            print(f"Model config summary: NT={model_cfg.get('NT')}, T={model_cfg.get('T')}")
        else:
            print("Config loaded but model section not found or empty.")

    print(f"Loaded marginals from {args.marginal_path}")
    print(f"Sentence index: {args.sentence_index}")
    print(f"Sentence length: {length}")
    print(f"Non-terminal labels: {width_one.shape[-1]}")
    print("\nSpan width = 1 marginals (start -> end):")

    for start in range(length):
        span_values = width_one[start]
        end = start + 1
        print(f"[{start}, {end}]: {span_values.tolist()}")


if __name__ == "__main__":
    main()

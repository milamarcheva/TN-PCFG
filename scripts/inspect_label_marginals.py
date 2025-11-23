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

    # Accept legacy formats: raw tensor, (marginal, seq_len) tuple/list, dict with
    # a 'marginal' entry, or dicts/lists that contain a tensor value we can use.
    marginal = None
    seq_len = None

    if isinstance(saved, torch.Tensor):
        marginal = saved
    elif isinstance(saved, (list, tuple)):
        if len(saved) == 2 and isinstance(saved[0], torch.Tensor):
            marginal, seq_len = saved
        elif all(isinstance(x, torch.Tensor) for x in saved):
            marginal = torch.stack(saved, dim=0)
        else:
            # Try to fall back to the first tensor-like entry.
            for item in saved:
                if isinstance(item, torch.Tensor):
                    marginal = item
                    break
    elif isinstance(saved, dict):
        if "marginal" in saved:
            marginal = saved["marginal"]
            seq_len = saved.get("seq_len")
        else:
            # Fallback: look for a tensor value in the dict.
            for value in saved.values():
                if isinstance(value, torch.Tensor):
                    marginal = value
                    break
            # If seq_len exists separately, keep it.
            if seq_len is None and "seq_len" in saved:
                seq_len = saved["seq_len"]

    if marginal is None:
        raise ValueError(
            "Unsupported saved format. Expected a tensor, a (marginal, seq_len) pair, "
            "a dict containing a 'marginal' entry, or a collection with a tensor entry."
        )

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

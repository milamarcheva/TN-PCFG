import argparse
from typing import Any, Tuple

import torch
import yaml


def _to_tensor(obj: Any) -> torch.Tensor:
    """Convert common container types to tensors."""

    if isinstance(obj, torch.Tensor):
        return obj

    if isinstance(obj, (list, tuple)):
        return torch.stack([_to_tensor(x) for x in obj])

    raise TypeError(f"Unsupported type for tensor conversion: {type(obj)}")


def load_marginals(path: str) -> Tuple[torch.Tensor, torch.Tensor | None]:
    """Load the saved marginals and optional sequence lengths."""

    saved = torch.load(path, map_location="cpu")

    # Expected new format: dict with 'marginal' (tensor) and optional 'seq_len'.
    if isinstance(saved, dict) and "marginal" in saved:
        marginal = _to_tensor(saved["marginal"])
        seq_len = saved.get("seq_len")
        if seq_len is not None:
            seq_len = _to_tensor(seq_len)
        return marginal, seq_len

    # Legacy formats: (marginal, seq_len) tuple or a bare tensor.
    if isinstance(saved, (list, tuple)):
        if len(saved) == 2:
            marginal = _to_tensor(saved[0])
            seq_len = _to_tensor(saved[1])
            return marginal, seq_len

        return _to_tensor(saved), None

    if isinstance(saved, torch.Tensor):
        return saved, None

    raise ValueError(
        "Unsupported saved format. Expected a tensor, a (marginal, seq_len) pair, "
        "or a dict containing a 'marginal' entry."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect saved label marginals for span width 1."
    )
    parser.add_argument("marginal_path", help="Path to the saved marginals file.")
    parser.add_argument(
        "--config",
        help="Optional config path (not required for printing raw marginals).",
    )
    parser.add_argument(
        "--sentence-index",
        type=int,
        default=0,
        help="0-based sentence index to inspect.",
    )

    args = parser.parse_args()

    marginal, seq_len = load_marginals(args.marginal_path)

    if marginal.dim() == 3:
        marginal = marginal.unsqueeze(0)

    if marginal.dim() != 4:
        raise ValueError(
            f"Expected marginal tensor with 4 dimensions [B, N, N, r_m], got {marginal.shape}."
        )

    batch_size = marginal.shape[0]
    if not (0 <= args.sentence_index < batch_size):
        raise IndexError(
            f"sentence-index {args.sentence_index} is out of range for batch size {batch_size}."
        )

    selected = marginal[args.sentence_index]
    if seq_len is not None:
        sentence_len = int(seq_len[args.sentence_index])
    else:
        sentence_len = selected.shape[0]

    width1 = torch.diagonal(selected, offset=1, dim1=0, dim2=1)[: sentence_len - 1]

    if args.config:
        try:
            with open(args.config, "r") as f:
                config = yaml.safe_load(f)
            non_terminals = config.get("grammar", {}).get("nonterminals", None)
        except FileNotFoundError:
            non_terminals = None
    else:
        non_terminals = None

    print(f"Sentence index: {args.sentence_index}")
    print(f"Sentence length: {sentence_len}")
    print("Width-1 span marginals (span [i, i+1)):")

    for i, vec in enumerate(width1):
        if non_terminals and len(non_terminals) == vec.numel():
            label_lines = ", ".join(
                f"{label}: {float(score):.6f}" for label, score in zip(non_terminals, vec)
            )
            print(f"  [{i}, {i+1}): {label_lines}")
        else:
            print(f"  [{i}, {i+1}): {vec.tolist()}")


if __name__ == "__main__":
    main()

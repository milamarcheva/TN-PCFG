#!/usr/bin/env python3
"""Inspect span-level label marginals exported by ``evaluate.py``.

The script expects the ``.pt`` payload produced by ``evaluate.py --label_marginal_out``
and can optionally recover surface forms via the checkpoint's YAML configuration.
It prints or exports summaries of posterior nonterminal distributions for spans in a
chosen sentence.
"""

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import torch


def _load_payload(path: Path) -> dict:
    payload = torch.load(str(path), map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError("Expected a dictionary payload; got {}".format(type(payload)))
    if "records" not in payload:
        raise KeyError("The provided file does not look like an exported label-marginal payload.")
    return payload


def _load_vocab(config_path: Path):
    import yaml
    from easydict import EasyDict as edict

    if config_path.is_dir():
        config_path = config_path / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Could not locate config file at {config_path}")

    try:
        from parser.helper.data_module import DataModule
    except ModuleNotFoundError as exc:
        missing = exc.name or "required dependency"
        raise ModuleNotFoundError(
            f"Failed to import {missing!r} while loading the vocabulary. "
            "Please ensure the project requirements are installed (see requirement.txt) "
            "or omit --config to fall back to raw token IDs."
        ) from exc

    with config_path.open("r", encoding="utf8") as handle:
        config = edict(yaml.safe_load(handle))
    # Force CPU to avoid unexpected GPU allocations when instantiating the data module.
    config.device = "cpu"
    data_module = DataModule(config)
    return data_module.word_vocab


def _decode_words(indices: Sequence[int], vocab) -> List[str]:
    if hasattr(indices, "tolist"):
        indices = indices.tolist()
    if vocab is None:
        return [str(int(idx)) for idx in indices]
    return [vocab.to_word(int(idx)) for idx in indices]


def _build_label_names(count: int, label_map_path: Optional[Path]) -> List[str]:
    if label_map_path is None:
        return [f"NT_{idx}" for idx in range(count)]
    with label_map_path.open("r", encoding="utf8") as handle:
        names = [line.strip() for line in handle if line.strip()]
    if len(names) != count:
        raise ValueError(
            f"Label map contains {len(names)} entries but the payload reports {count} nonterminals."
        )
    return names


def _iter_spans(length: int) -> Iterable[Tuple[int, int]]:
    for start in range(length):
        for end in range(start + 1, length + 1):
            yield start, end


def _format_span_tokens(tokens: Sequence[str], start: int, end: int) -> str:
    surface = " ".join(tokens[start:end])
    return surface if surface else "<empty>"


def _topk_distribution(dist: torch.Tensor, k: int, labels: Sequence[str]) -> List[Tuple[str, float]]:
    if dist.numel() == 0:
        return []
    probs = torch.softmax(dist, dim=-1)
    k = min(k, probs.numel())
    top_probs, top_indices = torch.topk(probs, k)
    return [(labels[idx], float(score)) for idx, score in zip(top_indices.tolist(), top_probs.tolist())]


def _summarise_spans(
    marginals: torch.Tensor,
    tokens: Sequence[str],
    labels: Sequence[str],
    *,
    topk: int,
    min_width: int,
    threshold: Optional[float],
    span: Optional[Tuple[int, int]],
    export_path: Optional[Path],
) -> None:
    length = marginals.size(0)
    entries = []

    def handle_span(start: int, end: int) -> None:
        if end - start < min_width:
            return
        dist = marginals[start, end - 1]
        top = _topk_distribution(dist, topk, labels)
        if not top:
            return
        best_prob = top[0][1]
        if threshold is not None and best_prob < threshold:
            return
        span_tokens = _format_span_tokens(tokens, start, end)
        entry = {
            "start": start + 1,
            "end": end,
            "width": end - start,
            "tokens": span_tokens,
            "top_labels": [{"label": label, "prob": prob} for label, prob in top],
        }
        entries.append(entry)

    if span is not None:
        handle_span(*span)
    else:
        for current in _iter_spans(length):
            handle_span(*current)

    if export_path is not None:
        export_path.parent.mkdir(parents=True, exist_ok=True)
        with export_path.open("w", encoding="utf8") as handle:
            json.dump(entries, handle, indent=2, ensure_ascii=False)
        print(f"Wrote {len(entries)} spans to {export_path}")
    else:
        for entry in entries:
            label_summary = ", ".join(
                f"{item['label']}={item['prob']:.4f}" for item in entry["top_labels"]
            )
            print(
                f"[{entry['start']}, {entry['end']}] width={entry['width']:>2} "
                f"tokens=\"{entry['tokens']}\" -> {label_summary}"
            )
        if not entries:
            print("No spans matched the provided filters.")


def _parse_span(text: str, length: int) -> Tuple[int, int]:
    try:
        start_str, end_str = text.split(":", 1)
        start = int(start_str)
        end = int(end_str)
    except ValueError as exc:  # pragma: no cover - defensive
        raise argparse.ArgumentTypeError(
            "Span must have the form START:END with 1-based indices"
        ) from exc
    if not (1 <= start < end <= length):
        raise argparse.ArgumentTypeError(
            f"Span {text!r} is out of bounds for a sentence of length {length}."
        )
    return start - 1, end


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("marginals", type=Path, help="Path to the .pt file produced by evaluate.py")
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional YAML config that was used to train/evaluate the checkpoint."
        " Needed to map word indices back to tokens.",
    )
    parser.add_argument(
        "--label-map",
        type=Path,
        help="Optional text file with one nonterminal label per line."
        " Defaults to NT_0, NT_1, ... if omitted.",
    )
    parser.add_argument(
        "--sentence-index",
        type=int,
        default=0,
        help="0-based index of the sentence to inspect.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Number of labels to display per span.",
    )
    parser.add_argument(
        "--min-width",
        type=int,
        default=1,
        help="Only consider spans with at least this many tokens.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Only display spans whose top posterior is at least this large.",
    )
    parser.add_argument(
        "--span",
        type=str,
        help="Optional 1-based START:END span selector. If omitted, summarise every span.",
    )
    parser.add_argument(
        "--export-json",
        type=Path,
        help="Write the collected span summaries to this JSON file instead of printing.",
    )

    args = parser.parse_args()

    payload = _load_payload(args.marginals)
    records: List[dict] = payload.get("records", [])
    if not records:
        raise ValueError("No label-marginal records found in the payload.")
    if not (0 <= args.sentence_index < len(records)):
        raise IndexError(
            f"Sentence index {args.sentence_index} is out of range for {len(records)} records."
        )

    record = records[args.sentence_index]
    length = int(record["seq_len"])
    marginals = record["label_marginal"].to(dtype=torch.float32)

    vocab = _load_vocab(args.config) if args.config else None
    if "tokens" in record and record["tokens"] is not None:
        words = list(record["tokens"])
    else:
        words = _decode_words(record["word"], vocab)

    nonterminals = marginals.size(-1)
    label_names = _build_label_names(nonterminals, args.label_map)

    print(f"Loaded sentence {args.sentence_index} from {args.marginals}")
    print("Sentence:", " ".join(words))
    print(f"Length: {length} tokens | Nonterminals: {nonterminals}")

    span = _parse_span(args.span, length) if args.span else None

    _summarise_spans(
        marginals,
        words,
        label_names,
        topk=args.topk,
        min_width=args.min_width,
        threshold=args.threshold,
        span=span,
        export_path=args.export_json,
    )


if __name__ == "__main__":
    main()

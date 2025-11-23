import argparse
from typing import Any

import torch


def describe_object(obj: Any) -> str:
    """Return a human-readable type description for the loaded object."""

    if isinstance(obj, dict):
        keys = ", ".join(str(k) for k in obj.keys())
        return f"dict with keys: {keys}"

    if isinstance(obj, (list, tuple)):
        return f"{type(obj).__name__} of length {len(obj)}"

    return type(obj).__name__


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect the type of saved label marginals."
    )
    parser.add_argument(
        "marginal_path",
        help="Path to the saved marginals file (e.g., output from --label_marginal_out).",
    )

    args = parser.parse_args()

    saved = torch.load(args.marginal_path, map_location="cpu")

    print(describe_object(saved))


if __name__ == "__main__":
    main()

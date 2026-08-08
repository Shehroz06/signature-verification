"""Command-line demo for the signature-verification adapter, using this child
project's own trained checkpoint by default.

Usage::

    uv run python -m signature_verification.backend.cli image_a.png image_b.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from signature_verification.backend.adapter import VerificationAdapter
from signature_verification.backend.config import VerificationAppConfig

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT = REPO_ROOT / "models" / "checkpoints" / "signature_forgery" / "best_model.pt"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Decide whether two signature images likely share a signer."
    )
    parser.add_argument("image_a", help="Path to the first image.")
    parser.add_argument("image_b", help="Path to the second image.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"Trained checkpoint to load (default: {DEFAULT_CHECKPOINT}).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=VerificationAppConfig().match_threshold,
        help="Cosine-similarity match threshold.",
    )
    args = parser.parse_args(argv)

    raw_image_a = cv2.imread(args.image_a, cv2.IMREAD_GRAYSCALE)
    raw_image_b = cv2.imread(args.image_b, cv2.IMREAD_GRAYSCALE)
    if raw_image_a is None or raw_image_b is None:
        print("Error: could not read one or both input images.", file=sys.stderr)
        return 1
    image_a = raw_image_a.astype(np.uint8)
    image_b = raw_image_b.astype(np.uint8)

    if not args.checkpoint.exists():
        print(
            f"Error: no checkpoint at {args.checkpoint} -- train one first "
            "(uv run python -m scripts.train corpus=signature_forgery) or pass "
            "--checkpoint explicitly.",
            file=sys.stderr,
        )
        return 1

    config = VerificationAppConfig(checkpoint_path=args.checkpoint, match_threshold=args.threshold)
    adapter = VerificationAdapter.from_config(config)
    decision = adapter.verify(image_a, image_b)

    verdict = "MATCH" if decision.is_match else "NO MATCH"
    print(
        f"Similarity: {decision.similarity:.4f} "
        f"(threshold {decision.threshold:.2f}) -> {verdict}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

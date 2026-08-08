"""Registry of loader functions for `scripts/train.py`'s `corpus=<name>` selection.

This child project (signature verification) only registers its own corpus.
Adding another one means writing a loader function here and a matching
`configs/corpus/<name>.yaml` file -- not a new training script.

Not part of `handwriting_engine` -- a standalone training-glue module in this
child project, consuming the engine as a dependency.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from datasets import load_dataset
from PIL.Image import Image as PILImage

from corpora.config import SigComp2011Config
from corpora.signatures import load_sigcomp2011

REPO_ROOT = Path(__file__).resolve().parents[1]
CEDAR_DIR = REPO_ROOT / "data" / "raw" / "cedar"


def _load_signature_forgery_rows() -> tuple[list[PILImage], list[str]]:
    """Signature forgery: SigComp2011 + CEDAR combined, genuine rows only (forged
    rows are held out as evaluation-time hard negatives -- see
    `scripts/evaluate_signature_forgery.py` for where forged rows are used)."""
    sigcomp = load_sigcomp2011(SigComp2011Config())
    cedar = load_dataset("imagefolder", data_dir=str(CEDAR_DIR))

    images: list[PILImage] = []
    writer_ids: list[str] = []
    for row in sigcomp["train"]:
        if row["is_genuine"]:
            images.append(row["image"])
            writer_ids.append(f"sigcomp_{row['writer_id']}")
    for row in cedar["train"]:
        if row["is_genuine"]:
            images.append(row["image"])
            writer_ids.append(row["writer_id"])  # already namespaced "cedar_*"
    return images, writer_ids


CORPUS_LOADERS: dict[str, Callable[[], tuple[list[PILImage], list[str]]]] = {
    "signature_forgery": _load_signature_forgery_rows,
}

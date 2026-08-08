"""ICDAR 2011 Signature Verification Competition (SigComp2011) loader.

Source: `1aurent/ICDAR-2011` on the HuggingFace Hub -- offline signature
samples with raw columns `image`, `label` (`ClassLabel` of "genuine" /
"forgeries"), `writer` (the claimed signer's id), `forger` and `attempt`
(competition bookkeeping, not needed downstream). Contains genuine signatures
plus skilled forgeries per writer, the standard offline signature-verification
setup.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from datasets import DatasetDict, load_dataset

from corpora._common import standardize_dataset_dict
from corpora.config import SigComp2011Config
from corpora.schema import DatasetSplit, SignatureMeta


def standardize_sigcomp_row(
    row: Mapping[str, Any], split: DatasetSplit, label_names: Sequence[str]
) -> dict[str, Any]:
    """Map one raw `1aurent/ICDAR-2011` row onto the standardized signature schema.

    `label_names` is the raw `label` column's `ClassLabel.names` -- iterating a
    `Dataset` yields the int class index, not the string, so the caller must
    resolve it via the source dataset's own feature definition.
    """
    is_genuine = label_names[row["label"]] == "genuine"
    meta = SignatureMeta(writer_id=str(row["writer"]), is_genuine=is_genuine, split=split)
    return {"image": row["image"], **meta.model_dump(mode="json")}


def load_sigcomp2011(config: SigComp2011Config, raw: DatasetDict | None = None) -> DatasetDict:
    """Load the ICDAR 2011 SigComp signature verification dataset, standardized
    to `SignatureMeta` + `image` columns.

    `raw` allows injecting an already-loaded `DatasetDict` (used by tests to
    exercise standardization without a network call); when omitted, the Hub
    dataset is downloaded/cached under `config.cache_dir`.
    """
    if raw is None:
        raw = load_dataset(config.hub_id, cache_dir=str(config.cache_dir))
    label_names = next(iter(raw.values())).features["label"].names
    return standardize_dataset_dict(
        raw, lambda row, split: standardize_sigcomp_row(row, split, label_names)
    )

"""Unit tests for corpora.signatures.

Exercises `load_sigcomp2011` against an in-memory `DatasetDict` shaped like
the real `1aurent/ICDAR-2011` schema (`image`, `label` as a `ClassLabel` of
"genuine"/"forgeries", `writer`), so no network access is needed. The
`ClassLabel` typing matters here: iterating it yields the int class index,
not the string name, which `standardize_sigcomp_row` must resolve correctly.
"""

from __future__ import annotations

from collections.abc import Callable

from datasets import ClassLabel, Dataset, DatasetDict, Features, Value
from PIL import Image

from corpora.config import SigComp2011Config
from corpora.signatures import load_sigcomp2011

_FEATURES = Features(
    {
        "image": Dataset.from_list([{"image": Image.new("L", (4, 4))}]).features["image"],
        "label": ClassLabel(names=["genuine", "forgeries"]),
        "writer": Value("uint32"),
        "forger": Value("int32"),
        "attempt": Value("uint32"),
    }
)


def test_load_sigcomp2011_resolves_class_label_and_writer(
    make_tiny_image: Callable[[], Image.Image],
) -> None:
    raw = DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "image": [make_tiny_image(), make_tiny_image()],
                    "label": [0, 1],
                    "writer": [7, 7],
                    "forger": [-1, 3],
                    "attempt": [1, 1],
                },
                features=_FEATURES,
            )
        }
    )

    result = load_sigcomp2011(SigComp2011Config(), raw=raw)

    assert result["train"]["is_genuine"] == [True, False]
    assert result["train"]["writer_id"] == ["7", "7"]
    assert result["train"]["split"] == ["train", "train"]

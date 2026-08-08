"""One-time data-prep script: converts the CEDAR signature dataset (downloaded
from a GitHub release mirror into data/incoming/cedar_dataset.zip) into the
imagefolder + metadata.jsonl convention used by this project's real-data
training scripts, under data/raw/cedar/{train,test}/.

Not part of `handwriting_engine` -- a standalone data-wrangling
utility, per the "freeze the architecture" validation pass.

Usage: uv run python -m scripts.prepare_cedar
"""

from __future__ import annotations

import json
import random
import re
import zipfile
from pathlib import Path

from PIL import Image

from scripts._common import already_prepared

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ZIP = REPO_ROOT / "data" / "incoming" / "cedar_dataset.zip"
OUTPUT_DIR = REPO_ROOT / "data" / "raw" / "cedar"

_GENUINE_PATTERN = re.compile(r"^full_org/original_(\d+)_(\d+)\.png$")
_FORGED_PATTERN = re.compile(r"^full_forg/forgeries_(\d+)_(\d+)\.png$")

_TEST_FRACTION = 0.2
_SEED = 42


def _writer_split(writer_ids: list[str]) -> dict[str, str]:
    """Assign each writer to train/test, writer-disjoint, so identification/
    verification evaluation is on genuinely unseen writers."""
    rng = random.Random(_SEED)
    shuffled = sorted(writer_ids, key=int)
    rng.shuffle(shuffled)
    num_test = max(1, int(len(shuffled) * _TEST_FRACTION))
    test_writers = set(shuffled[:num_test])
    return {writer: ("test" if writer in test_writers else "train") for writer in shuffled}


def main() -> None:
    if already_prepared(OUTPUT_DIR):
        print(f"{OUTPUT_DIR} already prepared (train/test metadata.jsonl present) -- skipping.")
        return

    if not SOURCE_ZIP.exists():
        raise SystemExit(f"CEDAR archive not found at {SOURCE_ZIP}")

    with zipfile.ZipFile(SOURCE_ZIP) as archive:
        entries: list[tuple[str, str, str, bool]] = []
        writer_ids: set[str] = set()
        for name in archive.namelist():
            genuine_match = _GENUINE_PATTERN.match(name)
            forged_match = _FORGED_PATTERN.match(name)
            if genuine_match:
                writer, sample = genuine_match.groups()
                entries.append((name, writer, sample, True))
                writer_ids.add(writer)
            elif forged_match:
                writer, sample = forged_match.groups()
                entries.append((name, writer, sample, False))
                writer_ids.add(writer)

        if not entries:
            raise SystemExit("No CEDAR entries matched the expected filename pattern.")

        split_by_writer = _writer_split(sorted(writer_ids))
        rows_by_split: dict[str, list[dict[str, object]]] = {"train": [], "test": []}
        for split in rows_by_split:
            (OUTPUT_DIR / split).mkdir(parents=True, exist_ok=True)

        for name, writer, sample, is_genuine in entries:
            split = split_by_writer[writer]
            kind = "genuine" if is_genuine else "forged"
            file_name = f"cedar_{writer}_{sample}_{kind}.png"
            with archive.open(name) as fh:
                image = Image.open(fh).convert("L")
                image.save(OUTPUT_DIR / split / file_name)
            rows_by_split[split].append(
                {"file_name": file_name, "writer_id": f"cedar_{writer}", "is_genuine": is_genuine}
            )

    for split, rows in rows_by_split.items():
        metadata_path = OUTPUT_DIR / split / "metadata.jsonl"
        with metadata_path.open("w") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        num_writers = len({row["writer_id"] for row in rows})
        print(f"{split}: {len(rows)} rows, {num_writers} writers -> {metadata_path}")


if __name__ == "__main__":
    main()

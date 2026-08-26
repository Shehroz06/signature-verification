"""Tests for signature_verification.backend.cli's argument handling and error
paths -- doesn't exercise the real embedding-model path (that needs the actual
trained checkpoint; see this project's manual CLI smoke test for that)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from signature_verification.backend.cli import main


def test_main_returns_error_for_missing_image_file(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.png"
    exit_code = main([str(missing), str(missing)])

    assert exit_code == 1


def test_main_returns_error_for_missing_checkpoint(tmp_path: Path) -> None:
    """Both images exist and are readable, but --checkpoint points nowhere
    -- must fail with a clear message rather than a raw torch/OS traceback."""
    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    cv2.imwrite(str(image_a), np.zeros((32, 32), dtype=np.uint8))
    cv2.imwrite(str(image_b), np.zeros((32, 32), dtype=np.uint8))
    missing_checkpoint = tmp_path / "no-such-checkpoint.pt"

    exit_code = main([str(image_a), str(image_b), "--checkpoint", str(missing_checkpoint)])

    assert exit_code == 1

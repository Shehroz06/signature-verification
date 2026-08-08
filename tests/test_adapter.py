"""Tests for signature_verification.backend.adapter.

Uses a tiny, randomly-initialized DINOv2 model injected via `EmbeddingModel`
(no network call) for decision-logic tests. The real-checkpoint loading path
(`VerificationAppConfig.checkpoint_path`, no injected model) is covered
separately in test_checkpoint_wiring.py, since that needs the real trained
weights, not this file's tiny model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from handwriting_engine.embeddings.config import EmbeddingConfig, EmbeddingHeadConfig
from handwriting_engine.embeddings.model import EmbeddingModel
from handwriting_engine.models.backbone import DINOv2Backbone
from handwriting_engine.models.config import BackboneConfig
from handwriting_engine.preprocessing.config import PreprocessingConfig
from numpy.typing import NDArray
from transformers import Dinov2Model

from signature_verification.backend.adapter import VerificationAdapter
from signature_verification.backend.config import VerificationAppConfig

_IMAGE_SIZE = 28


def _build_adapter(
    tiny_dinov2_model: Dinov2Model, match_threshold: float = 0.8
) -> VerificationAdapter:
    backbone_config = BackboneConfig(image_size=_IMAGE_SIZE)
    backbone = DINOv2Backbone(backbone_config, model=tiny_dinov2_model)
    embedding_config = EmbeddingConfig(
        backbone=backbone_config, head=EmbeddingHeadConfig(output_dim=8)
    )
    embedding_model = EmbeddingModel(embedding_config, backbone=backbone)

    config = VerificationAppConfig(
        preprocessing=PreprocessingConfig(),
        embedding=embedding_config,
        match_threshold=match_threshold,
    )
    return VerificationAdapter(config, embedding_model=embedding_model)


def _sample_image(fill: int, size: int = 60) -> NDArray[np.uint8]:
    image: NDArray[np.uint8] = np.zeros((size, size), dtype=np.uint8)
    image[10:50, 10:50] = fill
    return image


def test_identical_image_yields_perfect_similarity(tiny_dinov2_model: Dinov2Model) -> None:
    """Comparing an image to itself must give cosine similarity 1.0 exactly
    -- true regardless of whether the embedding model is well-trained,
    since both embeddings come from the literal same input."""
    adapter = _build_adapter(tiny_dinov2_model)
    image = _sample_image(fill=255)

    decision = adapter.verify(image, image)

    assert decision.similarity == pytest.approx(1.0)


def test_similarity_is_within_valid_cosine_range(tiny_dinov2_model: Dinov2Model) -> None:
    adapter = _build_adapter(tiny_dinov2_model)
    image_a = _sample_image(fill=255)
    image_b = _sample_image(fill=180)

    decision = adapter.verify(image_a, image_b)

    assert -1.0 <= decision.similarity <= 1.0 + 1e-5


def test_is_match_true_when_similarity_meets_threshold(tiny_dinov2_model: Dinov2Model) -> None:
    adapter = _build_adapter(tiny_dinov2_model, match_threshold=0.5)
    image = _sample_image(fill=255)

    decision = adapter.verify(image, image)  # similarity == 1.0

    assert decision.is_match is True
    assert decision.threshold == 0.5


def test_is_match_false_when_threshold_unreachable(tiny_dinov2_model: Dinov2Model) -> None:
    adapter = _build_adapter(tiny_dinov2_model, match_threshold=2.0)  # cosine sim maxes at 1.0
    image = _sample_image(fill=255)

    decision = adapter.verify(image, image)

    assert decision.is_match is False


def test_injected_embedding_model_takes_priority_over_checkpoint_path(
    tiny_dinov2_model: Dinov2Model,
) -> None:
    """An explicitly injected `embedding_model` must skip checkpoint loading
    entirely -- proven here by pointing `checkpoint_path` at a file that
    doesn't exist and confirming construction still succeeds (no attempt to
    load it), the same priority order test fixtures throughout this file
    already rely on."""
    backbone_config = BackboneConfig(image_size=_IMAGE_SIZE)
    backbone = DINOv2Backbone(backbone_config, model=tiny_dinov2_model)
    embedding_config = EmbeddingConfig(
        backbone=backbone_config, head=EmbeddingHeadConfig(output_dim=8)
    )
    embedding_model = EmbeddingModel(embedding_config, backbone=backbone)
    config = VerificationAppConfig(
        embedding=embedding_config,
        checkpoint_path=Path("/nonexistent/path/does-not-exist.pt"),
    )

    adapter = VerificationAdapter(config, embedding_model=embedding_model)

    image = _sample_image(fill=255)
    decision = adapter.verify(image, image)
    assert decision.similarity == pytest.approx(1.0)

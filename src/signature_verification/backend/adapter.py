"""Signature verification: an independent child project built on the
`handwriting_engine` core engine (installed as a dependency) -- composes
preprocessing and the embedding model into a single `.verify(image_a,
image_b)` call, with no changes to the engine itself.

This is one downstream use case among possibly several sibling child
projects (e.g. writer identification) that follow the same shape: compose
existing engine components, add nothing to `handwriting_engine` itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np
import torch
from handwriting_engine.adapters.base import EngineAdapter
from handwriting_engine.adapters.embedding import embed_image
from handwriting_engine.embeddings.model import EmbeddingModel
from handwriting_engine.training.checkpoint import load_checkpoint
from numpy.typing import NDArray

from signature_verification.backend.config import VerificationAppConfig


@dataclass(frozen=True)
class VerificationDecision:
    """Result of comparing two images for shared writer/signature identity."""

    similarity: float
    """Cosine similarity between the two images' embeddings, in ``[-1, 1]``."""
    is_match: bool
    threshold: float


class VerificationAdapter(EngineAdapter[VerificationAppConfig]):
    """Decides whether two handwriting/signature images likely share the
    same writer, via the Phase 6 embedding model's cosine similarity."""

    def __init__(
        self, config: VerificationAppConfig, embedding_model: EmbeddingModel | None = None
    ) -> None:
        self._config = config
        if embedding_model is not None:
            self._embedding_model = embedding_model
        else:
            self._embedding_model = EmbeddingModel(config.embedding)
            if config.checkpoint_path is not None:
                load_checkpoint(self._embedding_model, config.checkpoint_path)
        self._embedding_model.eval()

    @classmethod
    def from_config(cls, config: VerificationAppConfig) -> Self:
        return cls(config)

    def embed(self, image: NDArray[np.uint8]) -> torch.Tensor:
        """Preprocess and embed a single image -- usable standalone, not just
        via `.verify(...)`."""
        return embed_image(
            self._embedding_model,
            image,
            self._config.preprocessing,
            self._config.embedding.backbone,
        )

    def verify(
        self, image_a: NDArray[np.uint8], image_b: NDArray[np.uint8]
    ) -> VerificationDecision:
        """Preprocess and embed both images, then compare by cosine similarity."""
        embedding_a = self.embed(image_a)
        embedding_b = self.embed(image_b)
        similarity = float((embedding_a * embedding_b).sum())
        return VerificationDecision(
            similarity=similarity,
            is_match=similarity >= self._config.match_threshold,
            threshold=self._config.match_threshold,
        )

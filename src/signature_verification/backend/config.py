"""Configuration for the signature-verification adapter."""

from __future__ import annotations

from pathlib import Path

from handwriting_engine.embeddings.config import EmbeddingConfig
from handwriting_engine.preprocessing.config import PreprocessingConfig
from pydantic import BaseModel, ConfigDict


class VerificationAppConfig(BaseModel):
    """Configuration for `VerificationAdapter`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    preprocessing: PreprocessingConfig = PreprocessingConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    checkpoint_path: Path | None = None
    """Path to this downstream project's trained checkpoint (e.g.
    `models/checkpoints/signature_forgery/best_model.pt`, written by
    `scripts/train.py corpus=signature_forgery`). Loaded into a freshly
    constructed `EmbeddingModel` when the adapter builds its own model (i.e.
    no `embedding_model` is passed to `VerificationAdapter` directly -- that
    still takes priority, e.g. for tests injecting a tiny model). `None` (the
    default) builds an untrained model, matching this adapter's original
    Phase 10 prototype behavior -- explicit opt-in, not a silent gap."""
    match_threshold: float = 0.8
    """Cosine-similarity threshold above which two images are judged as the
    same writer/signature. This default is a starting point, not a
    calibrated value -- a real deployment should instead set this from
    `evaluation.verification.compute_verification_metrics`'s `eer_threshold`
    on held-out genuine/impostor pairs, e.g. from the signature-forgery
    downstream project's own evaluation run."""

"""Signature-verification adapter code: composes the shared engine
(preprocessing + embeddings) into a `.verify(image_a, image_b)` call, loaded
with this downstream project's own trained checkpoint (see `config.py`)."""

from signature_verification.backend.adapter import (
    VerificationAdapter,
    VerificationDecision,
)
from signature_verification.backend.config import VerificationAppConfig

__all__ = [
    "VerificationAdapter",
    "VerificationAppConfig",
    "VerificationDecision",
]

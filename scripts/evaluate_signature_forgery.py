"""Evaluates the signature-forgery embedding model on held-out SigComp2011 +
CEDAR test pairs, via the existing Phase 8 evaluation.verification
(EER/ROC-AUC), unmodified. Reports both the trained (Circle-Loss
fine-tuned) model and a frozen/untrained-DINOv2 baseline on the identical
protocol -- the internal ablation from the validation plan.

Two verification protocols are reported *separately*, never pooled:

- **Skilled forgery**: reference (a writer's first genuine test sample) vs.
  every other genuine sample from that writer (positive) and every forged
  sample attributed to that writer (negative) -- the standard
  reference-vs-questioned protocol, only meaningful for writers whose split
  actually contains forged attempts.
- **Random forgery**: reference vs. every other genuine sample from the same
  writer (positive) and a seeded sample of *other writers'* genuine
  signatures (negative) -- the fallback protocol the ICDAR SigComp
  challenges themselves define for exactly this situation, used here because
  SigComp2011's native test split contains zero forged rows (verified: 2534
  of 2534 test rows are genuine), so it cannot contribute to a skilled-forgery
  score at all. Impostor writers are always drawn from the same corpus as
  the reference, for the same reason skilled-forgery is scored per corpus
  below.

Pooling these two into one global EER/ROC-AUC (the previous behavior) is
invalid: SigComp2011 and CEDAR have different intrinsic cosine-similarity
scales (verified empirically -- SigComp2011 genuine-genuine pairs average
~0.80 baseline similarity, CEDAR forged pairs average ~0.91), so a pooled
computation is dominated by *which dataset a pair came from* rather than
*whether it's genuine*, producing a severely degraded, apparently-inverted
combined score even when each dataset's own separation is good (verified:
CEDAR alone scores baseline ROC-AUC=0.89, trained ROC-AUC=0.92). The same
scale-mismatch reasoning applies *within* random forgery too, so it's
reported per corpus (mirroring skilled forgery) rather than as one combined
figure. See `docs/DECISIONS.md` for the full investigation.

The current checkpoint is frozen as the official CEDAR baseline (see
README.md's "Evaluation results" table). SigComp2011's numbers are a known
limitation, deferred to future work rather than blocking this baseline --
see `docs/DECISIONS.md` (local notes).

Not part of `handwriting_engine` -- a standalone evaluation-glue
script, per the "freeze the architecture" validation pass.

Hydra-driven, same config surface as `scripts.train corpus=signature_forgery`: the
embedding architecture (`embeddings.*`) must match whatever the checkpoint
being loaded was trained with, and `training.device` selects CPU/CUDA, e.g.:

    uv run python -m scripts.evaluate_signature_forgery training.device=cuda

Usage: uv run python -m scripts.evaluate_signature_forgery
"""

from __future__ import annotations

import random
from pathlib import Path

import hydra
import numpy as np
import torch
from datasets import load_dataset
from handwriting_engine.embeddings.config import EmbeddingConfig
from handwriting_engine.embeddings.model import EmbeddingModel
from handwriting_engine.evaluation.verification import compute_verification_metrics
from handwriting_engine.training.checkpoint import load_checkpoint
from handwriting_engine.utils.device import resolve_device
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf
from PIL.Image import Image as PILImage

from corpora.config import SigComp2011Config
from corpora.signatures import load_sigcomp2011
from scripts._common import embed_all

REPO_ROOT = Path(__file__).resolve().parents[1]
CEDAR_DIR = REPO_ROOT / "data" / "raw" / "cedar"
CHECKPOINT_PATH = REPO_ROOT / "models" / "checkpoints" / "signature_forgery" / "model.pt"

_RANDOM_FORGERY_NEGATIVES_PER_WRITER = 4
_RANDOM_FORGERY_SEED = 42

_CORPUS_CEDAR = "cedar"
_CORPUS_SIGCOMP2011 = "sigcomp2011"


def _load_test_rows() -> tuple[list[PILImage], list[str], list[bool], list[str]]:
    sigcomp = load_sigcomp2011(SigComp2011Config())
    cedar = load_dataset("imagefolder", data_dir=str(CEDAR_DIR))

    images: list[PILImage] = []
    writer_ids: list[str] = []
    is_genuine: list[bool] = []
    corpora: list[str] = []
    for row in sigcomp["test"]:
        images.append(row["image"])
        writer_ids.append(f"sigcomp_{row['writer_id']}")
        is_genuine.append(bool(row["is_genuine"]))
        corpora.append(_CORPUS_SIGCOMP2011)
    for row in cedar["test"]:
        images.append(row["image"])
        writer_ids.append(row["writer_id"])
        is_genuine.append(bool(row["is_genuine"]))
        corpora.append(_CORPUS_CEDAR)
    return images, writer_ids, is_genuine, corpora


def _writer_corpus_map(writer_ids: list[str], corpora: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for writer_id, corpus in zip(writer_ids, corpora, strict=True):
        mapping.setdefault(writer_id, corpus)
    return mapping


def _genuine_indices_by_writer(
    writer_ids: list[str], is_genuine: list[bool]
) -> dict[str, list[int]]:
    by_writer: dict[str, list[int]] = {}
    for index, writer_id in enumerate(writer_ids):
        if is_genuine[index]:
            by_writer.setdefault(writer_id, []).append(index)
    return by_writer


def _build_skilled_forgery_pairs(
    writer_ids: list[str], is_genuine: list[bool]
) -> list[tuple[int, int, int]]:
    """Reference vs. other genuine (positive) and reference vs. forged
    (negative), restricted implicitly to writers whose split actually has
    forged samples -- currently only CEDAR's, since SigComp2011's test split
    has none (verified, see module docstring)."""
    by_writer: dict[str, list[int]] = {}
    for index, writer_id in enumerate(writer_ids):
        by_writer.setdefault(writer_id, []).append(index)

    pairs: list[tuple[int, int, int]] = []
    for indices in by_writer.values():
        genuine_indices = [i for i in indices if is_genuine[i]]
        forged_indices = [i for i in indices if not is_genuine[i]]
        if len(genuine_indices) < 2 or not forged_indices:
            continue
        reference = genuine_indices[0]
        for other in genuine_indices[1:]:
            pairs.append((reference, other, 1))
        for forged in forged_indices:
            pairs.append((reference, forged, 0))
    return pairs


def _build_random_forgery_pairs(
    writer_ids: list[str], is_genuine: list[bool], writer_corpus: dict[str, str]
) -> dict[str, list[tuple[int, int, int]]]:
    """Reference vs. other genuine (positive) and reference vs. a seeded
    sample of *other writers'* genuine signatures (negative) -- usable for
    every writer with at least one genuine test sample, including
    SigComp2011's, which has no forged rows to build skilled-forgery pairs
    from at all. Impostor writers are drawn only from the reference's own
    corpus, and pairs are bucketed by that same corpus, so the result can be
    reported per corpus exactly like skilled forgery -- pooling SigComp2011
    and CEDAR impostors together would reintroduce the same score-scale
    mismatch that made skilled forgery invalid to pool."""
    genuine_by_writer = _genuine_indices_by_writer(writer_ids, is_genuine)
    writers = sorted(genuine_by_writer)
    rng = random.Random(_RANDOM_FORGERY_SEED)

    pairs_by_corpus: dict[str, list[tuple[int, int, int]]] = {
        _CORPUS_CEDAR: [],
        _CORPUS_SIGCOMP2011: [],
    }
    for writer in writers:
        genuine_indices = genuine_by_writer[writer]
        if not genuine_indices:
            continue
        reference = genuine_indices[0]
        reference_corpus = writer_corpus[writer]
        bucket = pairs_by_corpus[reference_corpus]
        for other in genuine_indices[1:]:
            bucket.append((reference, other, 1))

        other_writers = [
            w for w in writers if w != writer and writer_corpus[w] == reference_corpus
        ]
        if not other_writers:
            continue
        chosen_writers = rng.sample(
            other_writers, k=min(_RANDOM_FORGERY_NEGATIVES_PER_WRITER, len(other_writers))
        )
        for impostor_writer in chosen_writers:
            impostor_index = rng.choice(genuine_by_writer[impostor_writer])
            bucket.append((reference, impostor_index, 0))
    return pairs_by_corpus


def _score_pairs(
    embeddings: torch.Tensor, pairs: list[tuple[int, int, int]]
) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
    scores = np.array(
        [float((embeddings[ref] * embeddings[other]).sum()) for ref, other, _ in pairs],
        dtype=np.float64,
    )
    labels = np.array([label for _, _, label in pairs], dtype=np.int_)
    return scores, labels


def _report(label: str, embeddings: torch.Tensor, pairs: list[tuple[int, int, int]]) -> None:
    if not pairs:
        print(f"{label}: no pairs available -- skipped.")
        return
    scores, labels = _score_pairs(embeddings, pairs)
    if len(np.unique(labels)) < 2:
        print(f"{label}: only one label class present ({len(pairs)} pairs) -- skipped.")
        return
    num_positive = int((labels == 1).sum())
    metrics = compute_verification_metrics(scores, labels)
    print(
        f"{label}: {len(pairs)} pairs ({num_positive} genuine, {len(pairs) - num_positive} "
        f"impostor) -- EER={metrics.eer:.4f} ROC-AUC={metrics.roc_auc:.4f} "
        f"threshold={metrics.eer_threshold:.4f}"
    )


def _evaluate(
    label: str,
    model: EmbeddingModel,
    images: list[PILImage],
    embedding_config: EmbeddingConfig,
    skilled_pairs: list[tuple[int, int, int]],
    random_pairs_by_corpus: dict[str, list[tuple[int, int, int]]],
) -> None:
    print(f"\n=== {label} ===")
    model.eval()
    embeddings = embed_all(model, images, embedding_config.backbone)
    _report("Skilled forgery (CEDAR)", embeddings, skilled_pairs)
    _report("Random forgery (CEDAR)", embeddings, random_pairs_by_corpus[_CORPUS_CEDAR])
    _report(
        "Random forgery (SigComp2011)", embeddings, random_pairs_by_corpus[_CORPUS_SIGCOMP2011]
    )


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    device = resolve_device(cfg.training.device)
    embedding_config = EmbeddingConfig.model_validate(
        OmegaConf.to_container(cfg.embeddings, resolve=True)
    )

    images, writer_ids, is_genuine, corpora = _load_test_rows()
    writer_corpus = _writer_corpus_map(writer_ids, corpora)
    skilled_pairs = _build_skilled_forgery_pairs(writer_ids, is_genuine)
    random_pairs_by_corpus = _build_random_forgery_pairs(writer_ids, is_genuine, writer_corpus)
    print(f"device: {device}")
    print(f"Held-out test corpus: {len(images)} images, {len(set(writer_ids))} writers")

    baseline_model = EmbeddingModel(embedding_config).to(device)
    _evaluate(
        "Baseline: frozen/untrained DINOv2 embeddings",
        baseline_model,
        images,
        embedding_config,
        skilled_pairs,
        random_pairs_by_corpus,
    )

    if not CHECKPOINT_PATH.exists():
        raise SystemExit(f"No trained checkpoint at {CHECKPOINT_PATH}; run training first.")
    trained_model = EmbeddingModel(embedding_config).to(device)
    load_checkpoint(trained_model, CHECKPOINT_PATH, map_location=str(device))
    _evaluate(
        "Trained: Circle-Loss fine-tuned embeddings",
        trained_model,
        images,
        embedding_config,
        skilled_pairs,
        random_pairs_by_corpus,
    )


if __name__ == "__main__":
    main()

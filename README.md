# Signature Verification

A downstream project built on the [Handwriting Analysis Engine](https://github.com/Shehroz06/handwriting-detection-engine) —
decides whether two signature images likely belong to the same signer, via a
DINOv2-embedding + Circle-Loss model trained on CEDAR + SigComp2011.

This project owns its own dataset, training run, checkpoint, and evaluation.
It does not modify the core engine — it only depends on it.

## Depending on the engine

`pyproject.toml` points at the engine via its GitHub source:

```toml
dependencies = ["handwriting-engine @ git+https://github.com/Shehroz06/handwriting-detection-engine.git"]
```


## Setup

```bash
uv sync --group dev
uv run pytest   # confirms the install works, no network or checkpoint required
```

## Evaluation results (CEDAR baseline, frozen)

The current checkpoint is frozen as the official baseline, measured on
held-out CEDAR test writers via `scripts/evaluate_signature_forgery.py`:

| Protocol | ROC-AUC | EER |
|---|---|---|
| Random forgery | **99.84%** | **2.12%** |
| Skilled forgery | **92.85%** | **15.09%** |

These are the numbers to cite for this model. The evaluation script also
reports SigComp2011 separately (per-corpus, never pooled with CEDAR — see
`scripts/evaluate_signature_forgery.py`'s docstring for why pooling the two
is invalid); SigComp2011 performance is a known limitation, not part of the
official baseline, and is deferred to future work rather than blocking this
release.

## Get the trained checkpoint

Checkpoints aren't committed to git (too large). Download the frozen CEDAR
baseline checkpoint from
[Google Drive](https://drive.google.com/drive/folders/1roxTZKBYnqHduDHm9AnNf-YqQAkO3Qu7?usp=sharing)
and place the files at:

```
models/checkpoints/signature_forgery/
    best_model.pt
    model.pt
    training_state.pt
    labels.json
```

## Use it

```bash
uv run python -m signature_verification.backend.cli image_a.png image_b.png
```

Or as a library:

```python
from pathlib import Path
from signature_verification.backend import VerificationAdapter, VerificationAppConfig

config = VerificationAppConfig(checkpoint_path=Path("models/checkpoints/signature_forgery/best_model.pt"))
adapter = VerificationAdapter.from_config(config)
decision = adapter.verify(image_a, image_b)  # numpy uint8 arrays
print(decision.similarity, decision.is_match)
```

## Train / re-train

Datasets aren't committed either — see `data/README.md` for acquisition + prep:

```bash
uv run python -m scripts.prepare_cedar
uv run python -m scripts.train corpus=signature_forgery
uv run python -m scripts.evaluate_signature_forgery
```

GPU (local CUDA, Kaggle, or Colab): `export UV_NO_SOURCES_PACKAGE="torch torchvision"`
before `uv sync`, then `training.device=cuda` on any command above. See
`scripts/colab_bootstrap.py` for Colab Drive-persistence helpers.

## Project structure

```
src/signature_verification/backend/   adapter, config, CLI
corpora/                              SigComp2011 + shared dataset-loader glue
scripts/                              train / evaluate / prepare / inference glue
configs/                              Hydra configuration
tests/
models/checkpoints/                   gitignored -- see "Get the trained checkpoint"
data/                                 gitignored -- see data/README.md
```

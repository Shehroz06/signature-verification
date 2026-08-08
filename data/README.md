# Data directory

Everything under `data/` (other than this file) is gitignored -- nothing is
committed or downloaded automatically.

## SigComp2011

`corpora.signatures.load_sigcomp2011` pulls directly from the HuggingFace Hub
(`1aurent/ICDAR-2011`) via `datasets.load_dataset(..., cache_dir=...)`; the
Hub download cache lives at `data/raw/sigcomp2011/` and needs no manual
preparation beyond network access.

## CEDAR

CEDAR (55 writers, signature verification) needs a manual download + convert:

```bash
mkdir -p data/incoming
# download cedar_dataset.zip into data/incoming/, e.g.:
#   https://github.com/nikostsagk/signature-verification/releases/download/cedar/cedar_dataset.zip
uv run python -m scripts.prepare_cedar
```

Re-splits writer-disjoint (80/20, seed 42) into:

```
data/raw/cedar/
  train/
    metadata.jsonl   # {"file_name": ..., "writer_id": "cedar_<n>", "is_genuine": true|false}
    <images>
  test/
    metadata.jsonl
    <images>
```

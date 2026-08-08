"""Dataset loading glue for this project's corpus (SigComp2011 + CEDAR).

Lives outside `signature_verification` (the installed package) deliberately:
this is a concrete, named corpus loader with task-specific schema baked in
(`writer_id`, `is_genuine`), not generic engine behavior. Consumed by
`scripts/`, not by `handwriting_engine` itself."""

from corpora.config import SigComp2011Config
from corpora.schema import DatasetSplit, RowMeta, SignatureMeta
from corpora.signatures import load_sigcomp2011, standardize_sigcomp_row

__all__ = [
    "DatasetSplit",
    "RowMeta",
    "SigComp2011Config",
    "SignatureMeta",
    "load_sigcomp2011",
    "standardize_sigcomp_row",
]

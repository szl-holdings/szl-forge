"""SZL Forge proof-carrying inference contracts and adapters."""

from inference.governed_inference import (
    InferenceBoundaryError,
    OpenAICompatibleGenerator,
    governed_infer,
    make_public_jsonl_hydrator,
    make_second_brain_retriever as make_legacy_second_brain_retriever,
    make_szl_nemo_witness,
)
from inference.production import (
    OpenAICompatibleProductionGenerator,
    ProductionBoundaryError,
    load_production_contract,
    make_second_brain_hydrator,
    make_second_brain_retriever,
    make_szl_nemo_envelope_witness,
    production_infer,
    verify_external_execution,
)

__version__ = "0.2.0"

__all__ = [
    "InferenceBoundaryError",
    "OpenAICompatibleGenerator",
    "OpenAICompatibleProductionGenerator",
    "ProductionBoundaryError",
    "governed_infer",
    "load_production_contract",
    "make_legacy_second_brain_retriever",
    "make_public_jsonl_hydrator",
    "make_second_brain_hydrator",
    "make_second_brain_retriever",
    "make_szl_nemo_envelope_witness",
    "make_szl_nemo_witness",
    "production_infer",
    "verify_external_execution",
    "__version__",
]

"""loka_causal — the causal knowledge graph Γ, its queries, and the admissibility matrix."""

from .admissibility import is_admissible
from .graph import CORE_LAYERS, CausalError, CausalGraph
from .slice import build_slice

__all__ = [
    "CausalGraph",
    "CausalError",
    "CORE_LAYERS",
    "build_slice",
    "is_admissible",
]

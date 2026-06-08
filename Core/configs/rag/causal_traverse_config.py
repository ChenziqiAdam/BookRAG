from pydantic import Field
from typing import Literal, Optional
from .base_config import BaseRAGStrategyConfig
from Core.configs.embedding_config import EmbeddingConfig


class CausalTraverseRAGConfig(BaseRAGStrategyConfig):
    strategy: Literal["causal_traverse"] = "causal_traverse"
    max_depth: int = Field(
        default=5,
        description="The maximum depth for the document tree traversal."
    )
    gate_boost: float = Field(
        default=3.0,
        description="Multiplier applied to causal-gate-connected nodes during traversal scoring."
    )
    embedding_config: Optional[EmbeddingConfig] = Field(
        default=None,
        description="Embedding config for computing causal gate scores."
    )

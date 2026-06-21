from pydantic import Field
from typing import Literal, Optional
from .base_config import BaseRAGStrategyConfig
from Core.configs.embedding_config import EmbeddingConfig
from Core.configs.rerank_config import RerankerConfig


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
    beam_width: int = Field(
        default=1,
        description="Number of branches to explore in parallel at each traversal level. 1=original single-path, 2=beam."
    )
    embedding_config: Optional[EmbeddingConfig] = Field(
        default=None,
        description="Embedding config for computing causal gate scores."
    )
    reranker_config: Optional[RerankerConfig] = Field(
        default=None,
        description="Reranker config for post-retrieval cross-encoder reranking. None disables reranking."
    )
    reranker_topk: int = Field(
        default=10,
        description="Keep top-K nodes after reranking before causal path identification."
    )

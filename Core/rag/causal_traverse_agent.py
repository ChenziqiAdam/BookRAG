import json
import random
import logging
from typing import List, Optional, Tuple

from Core.Index.Tree import DocumentTree, TreeNode
from Core.provider.llm import LLM
from Core.provider.vlm import VLM
from Core.provider.embedding import TextEmbeddingProvider
from Core.rag.traverse_agent import TraverseAgent
from Core.prompts.traverseagent_prompt import NAVIGATOR_PROMPT_TEMPLATE, NavigatorDecision
from Core.configs.rag.causal_traverse_config import CausalTraverseRAGConfig

log = logging.getLogger(__name__)


class CausalTraverseAgent(TraverseAgent):
    """
    Extends TraverseAgent with causal gate scoring.

    At each traversal step, each child option is augmented with a
    `causal_gate_score` = embedding_sim(query, child_summary) * gate_factor,
    where gate_factor = gate_boost if the child has any causal gate edge,
    else 1.0. This mirrors HugRAG's gated BFS edge-type boosting.
    """

    def __init__(
        self,
        config: CausalTraverseRAGConfig,
        llm: LLM,
        embedder: TextEmbeddingProvider,
        vlm: Optional[VLM] = None,
        tree_index: Optional[DocumentTree] = None,
    ):
        super().__init__(config=config, llm=llm, vlm=vlm, tree_index=tree_index)
        self.embedder = embedder
        self.gate_boost: float = config.gate_boost

    def _get_gate_edges_for_node(self, node_id: int) -> List[dict]:
        """Return all gate edge dicts where node_id is the destination (incoming causal influence).
        Handles legacy trees where edge values were plain strings (old format)."""
        edges = getattr(self.tree_index, "causal_gate_edges", {})
        result = []
        for (a, b), edge_data in edges.items():
            if b == node_id:
                # Migrate legacy string format to dict
                if isinstance(edge_data, str):
                    edge_data = {"direction": edge_data, "description": ""}
                result.append(edge_data)
        return result

    def _causal_gate_score(self, query: str, node: TreeNode) -> float:
        """
        Compute query-aware causal gate score for a child node.

        Mirrors HugRAG's _node_score: semantic * edge_factor.
        - semantic = embed_sim(query, node_summary), normalized to [0, 1]
        - edge_factor is query-aware: for each incoming gate edge, we compute
          embed_sim(query, edge_description) to measure how relevant the causal
          rationale is to this query. The final factor scales proportionally with
          the max edge relevance, from 1.0 (no gate) up to gate_boost (fully relevant gate).
        """
        summary = node.summary or node.meta_info.content or ""
        try:
            raw = self.embedder.compute_texts_sim(query, summary[:500]) if summary else 0.0
            semantic = (raw + 1) / 2  # normalize [-1,1] -> [0,1]
        except Exception:
            semantic = 0.5

        incoming_edges = self._get_gate_edges_for_node(node.index_id)
        if not incoming_edges:
            gate_factor = 1.0
        else:
            # Query-aware: score each gate edge's rationale against the query,
            # then use the max relevance to scale the boost proportionally.
            max_relevance = 0.0
            for edge_data in incoming_edges:
                desc = edge_data.get("description", "")
                if desc:
                    try:
                        rel = self.embedder.compute_texts_sim(query, desc)
                        rel = (rel + 1) / 2  # normalize to [0, 1]
                    except Exception:
                        rel = 0.5
                else:
                    rel = 0.5  # no description: assume moderate relevance
                max_relevance = max(max_relevance, rel)
            # Scale: gate_factor in [1.0, gate_boost] proportional to relevance
            gate_factor = 1.0 + (self.gate_boost - 1.0) * max_relevance

        return round(semantic * gate_factor, 4)

    def _create_navigator_prompt(
        self, query: str, current_node: TreeNode, child_nodes: List[TreeNode]
    ) -> str:
        """
        Same as TraverseAgent but adds `causal_gate_score` to each option.
        The LLM is instructed to prefer higher-scored nodes when content
        relevance is otherwise similar.
        """
        from Core.Index.Tree import NodeType

        options_list = []
        for i, child in enumerate(child_nodes, 1):
            if not child.summary:
                continue

            meta = child.meta_info
            option_data = {
                "choice_number": i,
                "type": child.type.upper() if child.type else "unknown",
                "summary": child.summary,
                "causal_gate_score": self._causal_gate_score(query, child),
            }

            if child.type in [NodeType.TITLE, NodeType.EQUATION] and meta.content:
                option_data["content"] = meta.content
            elif child.type in [NodeType.TABLE, NodeType.IMAGE] and meta.caption:
                option_data["caption"] = meta.caption
            elif child.type == NodeType.TEXT and meta.content:
                words = meta.content.split()
                preview = " ".join(words[:50])
                if len(words) > 50:
                    preview += "..."
                if preview:
                    option_data["content_preview"] = preview

            options_list.append(option_data)

        if options_list:
            options_str = json.dumps(options_list, indent=2)
        else:
            options_str = "No further nodes available."

        current_summary = current_node.summary or "This is the root of the document."

        # Extend the standard prompt with a note about causal_gate_score
        base_prompt = NAVIGATOR_PROMPT_TEMPLATE.format(
            query=query,
            current_summary=current_summary,
            options_str=options_str,
        )
        causal_note = (
            "\n**Note**: Each option includes a `causal_gate_score` (range 0–{max_score:.1f}). "
            "This score combines semantic similarity to your query with a causal relevance signal: "
            "a score above 1.0 means this section is causally linked to other sections AND that "
            "causal link is relevant to the current query — i.e., it likely contains prerequisite "
            "or dependent information needed to fully answer a multi-step question. "
            "Prefer higher-scored options when content relevance is otherwise similar."
        ).format(max_score=self.gate_boost)
        return base_prompt + causal_note

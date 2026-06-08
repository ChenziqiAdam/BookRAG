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

    def _has_causal_gate(self, node_id: int) -> bool:
        """Return True if node_id appears in any causal gate edge."""
        edges = getattr(self.tree_index, "causal_gate_edges", {})
        for a, b in edges:
            if a == node_id or b == node_id:
                return True
        return False

    def _causal_gate_score(self, query: str, node: TreeNode) -> float:
        """Compute causal gate score for a child node."""
        summary = node.summary or node.meta_info.content or ""
        if not summary:
            semantic = 0.5
        else:
            try:
                raw = self.embedder.compute_texts_sim(query, summary[:500])
                semantic = (raw + 1) / 2  # normalize [-1,1] -> [0,1]
            except Exception:
                semantic = 0.5

        gate_factor = self.gate_boost if self._has_causal_gate(node.index_id) else 1.0
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
            "\n**Note**: Each option includes a `causal_gate_score`. "
            "When two options are similarly relevant, prefer the one with a higher score, "
            "as it indicates stronger causal connections to other document sections."
        )
        return base_prompt + causal_note

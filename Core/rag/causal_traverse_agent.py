import json
import random
import logging
from typing import List, Optional, Tuple, Any

from Core.Index.Tree import DocumentTree, TreeNode
from Core.provider.llm import LLM
from Core.provider.vlm import VLM
from Core.provider.embedding import TextEmbeddingProvider
from Core.provider.rerank import TextRerankerProvider
from Core.rag.traverse_agent import TraverseAgent
from Core.prompts.traverseagent_prompt import NAVIGATOR_PROMPT_TEMPLATE, NavigatorDecision
from Core.prompts.hugrag_prompt import TREE_CAUSAL_PATH_PROMPT, CausalTreePathResult
from Core.prompts.gbc_prompt import (
    TEXT_RERANKER_PROMPT,
    QUESTION_EE_PROMPT,
    QUESTION_ENTITY_TYPES,
    QuestionEntityExtraction,
    SYNTHESIS_SYS_PROMPT,
    SYNTHESIS_USER_PROMPT,
)
from Core.rag.gbc_plan import TaskPlanner, PlanResult
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
        self.beam_width: int = getattr(config, "beam_width", 1)
        self.reranker_topk: int = getattr(config, "reranker_topk", 10)
        self.entity_seed_topk: int = getattr(config, "entity_seed_topk", 3)
        self.use_planning: bool = getattr(config, "use_planning", True)
        self.planner = TaskPlanner(llm=self.llm) if self.use_planning else None
        reranker_cfg = getattr(config, "reranker_config", None)
        if reranker_cfg is not None:
            self.reranker = TextRerankerProvider(
                model_name=reranker_cfg.model_name,
                max_length=reranker_cfg.max_length,
                device=reranker_cfg.device,
                backend=reranker_cfg.backend,
                api_base=reranker_cfg.api_base,
            )
        else:
            self.reranker = None

    def _get_gate_edges_for_node(self, node_id: int) -> List[dict]:
        """Return all gate edge dicts where node_id is the source (outgoing causal influence).
        We boost nodes that *cause* downstream content — matching HugRAG's BFS which follows
        outgoing causal edges from gate nodes.
        Handles legacy trees where edge values were plain strings (old format)."""
        edges = getattr(self.tree_index, "causal_gate_edges", {})
        result = []
        for (a, b), edge_data in edges.items():
            if a == node_id:  # node_id is the cause (source), not the effect
                # Migrate legacy string format to dict
                if isinstance(edge_data, str):
                    edge_data = {"direction": edge_data, "description": ""}
                result.append(edge_data)
        return result

    def _causal_gate_score(self, query: str, node: TreeNode) -> float:
        """
        Compute query-aware causal gate score for a child node.

        score = semantic * gate_factor  (multiplicative, matching HugRAG's _node_score)
        - semantic = embed_sim(query, node_summary), normalized to [0, 1]
        - gate_factor = 1.0 if no outgoing edges; else 1 + (gate_boost-1) * max_relevance,
          where max_relevance = max embed_sim(query, edge_description) over outgoing edges.
        Multiplicative form means gate only amplifies semantically relevant nodes —
        a semantically irrelevant node (semantic≈0) is not boosted even if causally connected,
        matching HugRAG's intent that causal gating accelerates relevant paths, not overrides.
        """
        summary = node.summary or node.meta_info.content or ""
        try:
            raw = self.embedder.compute_texts_sim(query, summary[:500]) if summary else 0.0
            semantic = (raw + 1) / 2  # normalize [-1,1] -> [0,1]
        except Exception:
            semantic = 0.5

        outgoing_edges = self._get_gate_edges_for_node(node.index_id)
        if not outgoing_edges:
            gate_factor = 1.0
        else:
            max_relevance = 0.0
            for edge_data in outgoing_edges:
                desc = edge_data.get("description", "")
                if desc:
                    try:
                        rel = self.embedder.compute_texts_sim(query, desc)
                        rel = (rel + 1) / 2
                    except Exception:
                        rel = 0.5
                else:
                    rel = 0.5
                max_relevance = max(max_relevance, rel)
            gate_factor = 1.0 + (self.gate_boost - 1.0) * max_relevance

        # Multiplicative: gate only amplifies nodes that are already semantically relevant.
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

        # Score all children, then sort descending so the LLM sees the best candidates first.
        # choice_number is preserved from the original ordering so index → child mapping is stable.
        scored_children = []
        for i, child in enumerate(child_nodes, 1):
            if not child.summary:
                continue
            score = self._causal_gate_score(query, child)
            scored_children.append((score, i, child))
        scored_children.sort(key=lambda x: x[0], reverse=True)

        options_list = []
        for score, orig_idx, child in scored_children:
            meta = child.meta_info
            option_data = {
                "choice_number": orig_idx,
                "type": child.type.upper() if child.type else "unknown",
                "summary": child.summary,
                "causal_gate_score": score,
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

        # Options are pre-sorted by causal_gate_score (highest first).
        # The note tells the LLM to treat this ordering as the primary signal.
        base_prompt = NAVIGATOR_PROMPT_TEMPLATE.format(
            query=query,
            current_summary=current_summary,
            options_str=options_str,
        )
        causal_note = (
            "\n**Note**: Each option has a `causal_gate_score` = semantic_relevance × causal_gate_factor. "
            "A higher score means the section is both semantically relevant to your query AND causally "
            "connected to other sections (i.e., it directly produces or enables content elsewhere). "
            "If two options differ in `causal_gate_score` by more than 0.05, prefer the higher-scored one. "
            "Otherwise use your judgment based on content relevance."
        )
        return base_prompt + causal_note

    def _seed_nodes_from_query(self, query: str, existing_ids: set) -> List[TreeNode]:
        """
        Entity-based section seeding: extract key terms from query via LLM, then
        find tree nodes whose summary/content best matches those terms via embedding
        similarity. Returns up to entity_seed_topk nodes not already in existing_ids.
        Mirrors BookRAG's entity→section linking step but without requiring a KG/VDB.
        """
        if not self.tree_index:
            return []

        # Extract entities from query
        prompt = QUESTION_EE_PROMPT.format(
            entity_types=", ".join(QUESTION_ENTITY_TYPES),
            input_text=query,
        )
        try:
            result = self.llm.get_json_completion(prompt=prompt, schema=QuestionEntityExtraction)
            entities = result.entities if result and result.entities else []
        except Exception as e:
            log.warning(f"Entity extraction failed: {e}. Skipping seed step.")
            return []

        if not entities:
            return []

        # Build a combined query string from entity names
        entity_query = " ".join(e.entity_name for e in entities)
        log.info(f"Entity seeding: extracted {len(entities)} entities: {[e.entity_name for e in entities]}")

        # Score all leaf/content nodes by embedding similarity to entity query
        all_nodes = [n for n in self.tree_index.nodes if n.index_id not in existing_ids]
        scored = []
        for node in all_nodes:
            text = node.summary or node.meta_info.content or ""
            if not text:
                continue
            try:
                sim = self.embedder.compute_texts_sim(entity_query, text[:500])
                sim = (sim + 1) / 2
            except Exception:
                sim = 0.0
            scored.append((sim, node))

        scored.sort(key=lambda x: x[0], reverse=True)
        seed_nodes = [n for _, n in scored[:self.entity_seed_topk]]
        log.info(f"Entity seeding: added {len(seed_nodes)} seed nodes (topk={self.entity_seed_topk})")
        return seed_nodes

    def _rerank_nodes(self, query: str, nodes: List[TreeNode]) -> List[TreeNode]:
        """Cross-encoder rerank retrieved nodes, keep top reranker_topk."""
        if not self.reranker or not nodes:
            return nodes
        from Core.Index.Tree import NodeType
        from Core.utils.table_utils import table2text

        doc_texts = []
        for node in nodes:
            if node.type == NodeType.TABLE:
                text = table2text(node.meta_info.__dict__)
            else:
                text = node.meta_info.content or node.summary or ""
            doc_texts.append(text)

        try:
            scores = self.reranker.rerank(query=query, documents=doc_texts, instruction=TEXT_RERANKER_PROMPT)
            self.reranker.clean_cache()
            ranked = sorted(zip(nodes, scores), key=lambda x: x[1], reverse=True)
            kept = [n for n, _ in ranked[:self.reranker_topk]]
            log.info(f"Reranker: {len(nodes)} → {len(kept)} nodes (topk={self.reranker_topk})")
            return kept
        except Exception as e:
            log.warning(f"Reranker failed: {e}. Using all retrieved nodes.")
            return nodes

    def _identify_causal_path(self, query: str, traversal_path: List[TreeNode]) -> Tuple[List[TreeNode], List[TreeNode]]:
        """
        Post-retrieval: ask LLM to identify which nodes in the traversal path are on the
        causal chain vs. spurious. Returns (causal_nodes, spurious_nodes).
        Mirrors HugRAG's causal path identification step before answer generation.
        """
        if not traversal_path:
            return [], []

        node_summaries = []
        for node in traversal_path:
            # Use full content for accurate causal judgment; fall back to summary if no content.
            content = node.meta_info.content or node.summary or ""
            node_summaries.append(f"[Node {node.index_id}]:\n{content[:800]}")
        nodes_text = "\n\n".join(node_summaries)

        prompt = TREE_CAUSAL_PATH_PROMPT.format(query=query, nodes_text=nodes_text)
        try:
            result = self.llm.get_json_completion(prompt=prompt, schema=CausalTreePathResult)
        except Exception as e:
            log.warning(f"Causal path identification failed: {e}. Using full path.")
            return traversal_path, []

        if not result or not isinstance(result, CausalTreePathResult):
            return traversal_path, []

        causal_ids = set(result.causal_node_ids)
        spurious_ids = set(result.spurious_node_ids)

        causal_nodes = [n for n in traversal_path if n.index_id in causal_ids]
        spurious_nodes = [n for n in traversal_path if n.index_id in spurious_ids]
        # Nodes not classified either way go into causal (safe default)
        unclassified = [n for n in traversal_path if n.index_id not in causal_ids and n.index_id not in spurious_ids]
        causal_nodes = causal_nodes + unclassified

        log.info(f"Causal path: {[n.index_id for n in causal_nodes]}, spurious: {[n.index_id for n in spurious_nodes]}")
        return causal_nodes, spurious_nodes

    def _retrieve(self, query: str) -> List[TreeNode]:
        """
        Beam traversal: at each depth level, maintain up to `beam_width` active frontier nodes.
        All children of all frontier nodes are scored; the top-K by causal_gate_score advance.
        All visited nodes are collected (deduplicated by index_id) as the context.

        When beam_width=1, behaviour is identical to the parent TraverseAgent single-path traversal
        (but uses causal_gate_score for ranking instead of LLM navigation).
        """
        if self.beam_width <= 1:
            # Fall back to original LLM-navigator single-path traversal
            return super()._retrieve(query)

        if not self.tree_index or not self.tree_index.root_node:
            return []

        max_depth = self.tree_index.get_max_depth() + 1
        if self.max_depth != -1:
            max_depth = min(max_depth, self.max_depth)

        root = self.tree_index.root_node
        # frontier: current set of nodes to expand
        frontier: List[TreeNode] = [root]
        visited_ids: set = set()
        collected: List[TreeNode] = []

        def _add(node: TreeNode):
            if node.index_id not in visited_ids:
                visited_ids.add(node.index_id)
                collected.append(node)

        _add(root)

        for depth in range(max_depth):
            # Gather all children of all frontier nodes
            candidates: List[Tuple[float, TreeNode]] = []
            for node in frontier:
                for child in node.children:
                    if child.index_id in visited_ids:
                        continue
                    if not child.summary:
                        continue
                    score = self._causal_gate_score(query, child)
                    candidates.append((score, child))

            if not candidates:
                break

            # Keep top beam_width candidates by score
            candidates.sort(key=lambda x: x[0], reverse=True)
            next_frontier = []
            for score, child in candidates[:self.beam_width]:
                _add(child)
                next_frontier.append(child)

            log.info(f"Beam depth {depth+1}: expanded {len(candidates)} candidates → kept {len(next_frontier)} (beam_width={self.beam_width})")
            frontier = next_frontier

        return collected

    def _retrieve_and_answer(self, query: str) -> Tuple[str, List[TreeNode]]:
        """
        Core single-query RAG: traverse → entity seed → rerank → causal path ID → generate.
        Returns (answer_text, context_nodes).
        """
        from Core.Index.Tree import NodeType
        from Core.prompts.hugrag_prompt import CAUSAL_ANSWER_PROMPT

        context_nodes = self._retrieve(query)

        existing_ids = {n.index_id for n in context_nodes}
        seed_nodes = self._seed_nodes_from_query(query, existing_ids)
        context_nodes = context_nodes + seed_nodes

        context_nodes = self._rerank_nodes(query, context_nodes)

        causal_nodes, spurious_nodes = self._identify_causal_path(query, context_nodes)

        def _node_to_text(node: TreeNode) -> str:
            meta = node.meta_info
            node_type = node.type
            parts = [f"## Section (Type: {node_type})"]
            if node_type in [NodeType.TEXT, NodeType.TITLE, NodeType.EQUATION] and meta.content:
                parts.append(meta.content)
            elif node_type in [NodeType.TABLE] and meta.content:
                parts.append(meta.content)
                if meta.table_body:
                    parts.append(f"Table:\n{meta.table_body}")
            elif node_type == NodeType.IMAGE and meta.content:
                parts.append(meta.content)
            return "\n".join(parts)

        image_paths = []
        for node in context_nodes:
            if node.type == NodeType.IMAGE and node.meta_info.img_path:
                image_paths.append(node.meta_info.img_path)

        if spurious_nodes:
            causal_text = "\n\n".join(_node_to_text(n) for n in causal_nodes) or "None"
            additional_text = "\n\n".join(_node_to_text(n) for n in spurious_nodes)
            spurious_text = "\n\n".join(f"[Node {n.index_id}]: {n.summary or ''}" for n in spurious_nodes)
            final_prompt = CAUSAL_ANSWER_PROMPT.format(
                query=query,
                causal_path_text=causal_text,
                additional_context=additional_text,
                spurious_nodes=spurious_text,
            )
        else:
            final_prompt, image_paths = self._create_augmented_prompt(query, causal_nodes)

        if image_paths and self.vlm:
            answer = self.vlm.generate(prompt_or_memory=final_prompt, images=image_paths)
        else:
            answer = self.llm.get_completion(prompt=final_prompt)

        return answer, context_nodes

    def generation(self, query: str, query_output_dir: str) -> Tuple[str, List[Any]]:
        """
        Full RAG flow with optional query planning (V19).
        - Simple queries: single retrieve-and-answer pass.
        - Complex queries: decompose into sub-questions, answer each, synthesise.
        """
        all_context_nodes: List[TreeNode] = []
        final_answer = ""

        if self.planner:
            try:
                plan: PlanResult = self.planner.analyze(query)
                log.info(f"Query plan: type={plan.query_type}")
            except Exception as e:
                log.warning(f"Planning failed: {e}. Falling back to simple retrieval.")
                plan = PlanResult(query_type="simple", original_query=query)
        else:
            plan = PlanResult(query_type="simple", original_query=query)

        if plan.query_type == "complex" and plan.sub_questions:
            retrieval_tasks = [sq for sq in plan.sub_questions if sq.type == "retrieval"]
            synthesis_step = next((sq for sq in plan.sub_questions if sq.type == "synthesis"), None)

            sub_results = []
            for task in retrieval_tasks:
                sub_answer, sub_nodes = self._retrieve_and_answer(task.question)
                sub_results.append({"question": task.question, "answer": sub_answer})
                for n in sub_nodes:
                    if n.index_id not in {x.index_id for x in all_context_nodes}:
                        all_context_nodes.append(n)

            log.info(f"Complex query: {len(retrieval_tasks)} sub-questions answered.")

            # Synthesise final answer from sub-results
            partial_answers_str = "\n\n".join(
                f"Q: {r['question']}\nA: {r['answer']}" for r in sub_results
            )
            if synthesis_step:
                partial_answers_str += f"\n\nSynthesis task: {synthesis_step.question}"

            from Core.Common.Memory import Memory
            from Core.Common.Message import Message
            synthesis_memory = Memory()
            synthesis_memory.add(Message(role="system", content=SYNTHESIS_SYS_PROMPT))
            synthesis_memory.add(Message(role="user", content=SYNTHESIS_USER_PROMPT.format(
                user_question=query,
                partial_answers_str=partial_answers_str,
            )))
            try:
                final_answer = self.llm.get_completion(synthesis_memory)
            except Exception as e:
                log.error(f"Synthesis failed: {e}. Using last sub-answer.")
                final_answer = sub_results[-1]["answer"] if sub_results else ""
        else:
            # Simple (or fallback): single pass
            final_answer, all_context_nodes = self._retrieve_and_answer(query)

        retrieval_node_ids = self._save_retrieval_res(all_context_nodes, query_output_dir)
        return final_answer, retrieval_node_ids

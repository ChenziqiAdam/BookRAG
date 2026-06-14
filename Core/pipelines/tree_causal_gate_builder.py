import logging
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Set, Tuple

from Core.Index.Tree import DocumentTree, TreeNode
from Core.provider.llm import LLM
from Core.prompts.hugrag_prompt import TREE_CROSS_SECTION_GATE_PROMPT, CausalGateResult

log = logging.getLogger(__name__)


def _get_all_descendant_ids(node: TreeNode) -> Set[int]:
    """Return the set of index_ids for all descendants of node (not including node itself)."""
    result: Set[int] = set()
    queue = deque(node.children)
    while queue:
        child = queue.popleft()
        result.add(child.index_id)
        queue.extend(child.children)
    return result


MAX_GATE_DEPTH = 2  # Only build gates at depth 1-2 (top-level and second-level sections).
# Deeper nodes are too granular: at depth 3+ there can be thousands of cross-branch pairs
# (seen: 2346 pairs at depth 3 for one doc), generating mostly noise at high cost.
# HugRAG's graph version only checks cross-module pairs (~5-20 modules), analogous to depth 1-2.


def build_tree_causal_gates(
    tree_index: DocumentTree, llm: LLM, max_workers: int = 4
) -> DocumentTree:
    """
    Build causal gate edges between tree nodes at depth 1 and 2 only (BFS order).

    For each node U at a given depth, we check all other nodes V at the same depth:
      - Skip if V is a descendant of U (hierarchy already covers it)
      - Skip if V's parent already has a causal link to U (transitivity)

    Results are stored in tree_index.causal_gate_edges as {(id_A, id_B): {"direction": ..., "description": ...}}.
    """
    if not tree_index.root_node:
        log.warning("Tree has no root node; skipping causal gate building.")
        return tree_index

    # BFS level by level
    gate_edges: Dict[Tuple[int, int], str] = {}

    # Track which node pairs have already been determined to have causal links
    # for transitivity pruning: causally_linked[a] = set of node ids causally linked to a
    causally_linked: Dict[int, Set[int]] = {}

    max_depth = min(tree_index.get_max_depth(), MAX_GATE_DEPTH)

    for depth in range(1, max_depth + 1):
        nodes_at_depth = tree_index.get_nodes_at_depth(depth)
        if len(nodes_at_depth) < 2:
            continue

        # Precompute descendant sets for all nodes at this depth
        descendant_ids: Dict[int, Set[int]] = {
            n.index_id: _get_all_descendant_ids(n) for n in nodes_at_depth
        }

        # Build list of valid pairs (U, V) where U.index_id < V.index_id
        pairs = []
        for i, u in enumerate(nodes_at_depth):
            u_id = u.index_id
            u_linked = causally_linked.get(u_id, set())
            for v in nodes_at_depth[i + 1:]:
                v_id = v.index_id

                # Skip 1: V is a descendant of U (or U of V)
                if v_id in descendant_ids[u_id] or u_id in descendant_ids.get(v_id, set()):
                    continue

                # Skip 2: V's parent already causally linked to U (transitivity)
                if v.parent and v.parent.index_id in u_linked:
                    continue
                # And symmetric: U's parent already linked to V
                u_parent_id = u.parent.index_id if u.parent else None
                v_linked = causally_linked.get(v_id, set())
                if u_parent_id is not None and u_parent_id in v_linked:
                    continue

                # Skip if already checked (edge exists)
                if (u_id, v_id) in gate_edges or (v_id, u_id) in gate_edges:
                    continue

                pairs.append((u, v))

        if not pairs:
            log.info(f"Depth {depth}: no valid pairs to check.")
            continue

        log.info(f"Depth {depth}: checking {len(pairs)} node pairs for causal gates...")

        def _check_pair(u: TreeNode, v: TreeNode):
            summary_a = u.summary or u.meta_info.content or f"Node {u.index_id}"
            summary_b = v.summary or v.meta_info.content or f"Node {v.index_id}"
            # Truncate to stay within token limits (mirrors causal_gate_builder.py's 500-char cap)
            summary_a = summary_a[:500]
            summary_b = summary_b[:500]
            prompt = TREE_CROSS_SECTION_GATE_PROMPT.format(
                section_a_summary=summary_a,
                section_b_summary=summary_b,
            )
            try:
                result: CausalGateResult = llm.get_json_completion(prompt, CausalGateResult)
                return u, v, result
            except Exception as e:
                # Fallback: retry with aggressively truncated summaries (250 chars) to avoid token limits
                log.warning(f"Gate check failed for nodes ({u.index_id}, {v.index_id}), retrying with shorter summaries: {e}")
                try:
                    prompt_short = TREE_CROSS_SECTION_GATE_PROMPT.format(
                        section_a_summary=summary_a[:250],
                        section_b_summary=summary_b[:250],
                    )
                    result = llm.get_json_completion(prompt_short, CausalGateResult)
                    return u, v, result
                except Exception as e2:
                    log.warning(f"Gate check failed after retry for nodes ({u.index_id}, {v.index_id}): {e2}")
                    return u, v, None

        edges_added = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_check_pair, u, v) for u, v in pairs]
            for future in as_completed(futures):
                u, v, result = future.result()
                if result is None or isinstance(result, dict) or not result.has_causal_link:
                    continue

                u_id, v_id = u.index_id, v.index_id
                direction = result.direction
                description = result.description  # store rationale for query-aware scoring

                if direction in ("A->B", "bidirectional"):
                    gate_edges[(u_id, v_id)] = {"direction": direction, "description": description}
                    causally_linked.setdefault(u_id, set()).add(v_id)
                    edges_added += 1

                if direction in ("B->A", "bidirectional"):
                    gate_edges[(v_id, u_id)] = {"direction": direction, "description": description}
                    causally_linked.setdefault(v_id, set()).add(u_id)
                    edges_added += 1

        log.info(f"Depth {depth}: added {edges_added} causal gate edges.")

    tree_index.causal_gate_edges = gate_edges
    log.info(f"Tree causal gate building complete. Total gate edges: {len(gate_edges)}")
    return tree_index

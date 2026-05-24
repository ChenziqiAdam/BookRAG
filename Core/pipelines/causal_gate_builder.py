import logging
from typing import Dict, List, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

import networkx as nx
from networkx.algorithms import community

from Core.Index.Graph import Graph
from Core.provider.llm import LLM
from Core.prompts.hugrag_prompt import CROSS_MODULE_GATE_PROMPT, CausalGateResult

log = logging.getLogger(__name__)


def _detect_communities(kg: nx.Graph) -> List[Set[str]]:
    """Run Louvain community detection and return a list of node-name sets."""
    try:
        communities = list(community.louvain_communities(kg, seed=42))
        log.info(f"Detected {len(communities)} communities via Louvain.")
        return communities
    except Exception as e:
        log.warning(f"Louvain failed ({e}), falling back to greedy modularity.")
        communities = list(community.greedy_modularity_communities(kg))
        log.info(f"Detected {len(communities)} communities via greedy modularity.")
        return communities


def _build_module_summary(kg: nx.Graph, nodes: Set[str], max_chars: int = 500) -> str:
    parts = []
    for node in nodes:
        desc = kg.nodes[node].get("description", "")
        name = kg.nodes[node].get("entity_name", node)
        if desc:
            parts.append(f"{name}: {desc[:80]}")
        else:
            parts.append(name)
    summary = "; ".join(parts)
    return summary[:max_chars]


def _highest_degree_node(kg: nx.Graph, nodes: Set[str]) -> str:
    return max(nodes, key=lambda n: kg.degree(n))


def build_causal_gates(graph_index: Graph, llm: LLM, max_workers: int = 4) -> Graph:
    """
    Detect communities in the KG, then add cross-module causal gate edges
    between module pairs that share a causal relationship (as judged by LLM).
    """
    kg = graph_index.kg
    if kg.number_of_nodes() == 0:
        log.warning("Graph is empty; skipping causal gate building.")
        return graph_index

    communities = _detect_communities(kg)
    if len(communities) < 2:
        log.info("Fewer than 2 communities detected; no cross-module gates to add.")
        return graph_index

    # Build representative node and summary for each community
    module_reps: List[str] = []
    module_summaries: List[str] = []
    for comm in communities:
        rep = _highest_degree_node(kg, comm)
        summary = _build_module_summary(kg, comm)
        module_reps.append(rep)
        module_summaries.append(summary)

    n = len(communities)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    log.info(f"Querying LLM for {len(pairs)} module pairs...")

    def _check_pair(i, j):
        prompt = CROSS_MODULE_GATE_PROMPT.format(
            module_a_summary=module_summaries[i],
            module_b_summary=module_summaries[j],
        )
        try:
            result: CausalGateResult = llm.get_json_completion(prompt, CausalGateResult)
            if result is None:
                return i, j, None
            return i, j, result
        except Exception as e:
            log.warning(f"Gate check failed for modules ({i}, {j}): {e}")
            return i, j, None

    gate_edges_added = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_check_pair, i, j): (i, j) for i, j in pairs}
        for future in as_completed(futures):
            i, j, result = future.result()
            if result is None or not result.has_causal_link:
                continue

            rep_a = module_reps[i]
            rep_b = module_reps[j]
            direction = result.direction
            desc = result.description

            if direction in ("A->B", "bidirectional"):
                if not kg.has_edge(rep_a, rep_b):
                    kg.add_edge(
                        rep_a, rep_b,
                        is_causal=True,
                        edge_type="causal_gate",
                        causal_confidence=1.0,
                        relation_name="causal_gate",
                        description=desc,
                        weight=1.0,
                        source_ids=set(),
                    )
                    gate_edges_added += 1

            if direction in ("B->A", "bidirectional"):
                if not kg.has_edge(rep_b, rep_a):
                    kg.add_edge(
                        rep_b, rep_a,
                        is_causal=True,
                        edge_type="causal_gate",
                        causal_confidence=1.0,
                        relation_name="causal_gate",
                        description=desc,
                        weight=1.0,
                        source_ids=set(),
                    )
                    gate_edges_added += 1

    log.info(f"Causal gate building complete. Added {gate_edges_added} gate edges.")
    return graph_index

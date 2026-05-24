import heapq
import logging
from typing import Dict, List, Set, Tuple, Any

import networkx as nx
from networkx.algorithms import community

from Core.Index.Graph import Graph
from Core.provider.embedding import TextEmbeddingProvider

log = logging.getLogger(__name__)


def _detect_communities(kg: nx.Graph):
    try:
        return list(community.louvain_communities(kg, seed=42))
    except Exception:
        return list(community.greedy_modularity_communities(kg))


def _build_module_summary(kg: nx.Graph, nodes: Set[str], max_chars: int = 500) -> str:
    parts = []
    for node in nodes:
        desc = kg.nodes[node].get("description", "")
        name = kg.nodes[node].get("entity_name", node)
        parts.append(f"{name}: {desc[:60]}" if desc else name)
    return "; ".join(parts)[:max_chars]


def _highest_degree_node(kg: nx.Graph, nodes: Set[str]) -> str:
    return max(nodes, key=lambda n: kg.degree(n))


def module_level_seeding(
    query: str,
    graph_index: Graph,
    embedder: TextEmbeddingProvider,
    top_k_module: int = 3,
) -> Set[str]:
    """
    Return seed node names from module-level matching.
    Detects communities, summarizes each, then picks representative nodes
    from the top-k modules most similar to the query.
    """
    kg = graph_index.kg
    if kg.number_of_nodes() == 0:
        return set()

    communities = _detect_communities(kg)
    if not communities:
        return set()

    # Build summaries and representatives
    summaries = []
    reps = []
    for comm in communities:
        summaries.append(_build_module_summary(kg, comm))
        reps.append(_highest_degree_node(kg, comm))

    # Score each module summary against the query
    scored = []
    for idx, summary in enumerate(summaries):
        try:
            sim = embedder.compute_texts_sim(query, summary)
        except Exception:
            sim = 0.0
        scored.append((sim, idx))

    scored.sort(reverse=True)
    top_indices = [idx for _, idx in scored[:top_k_module]]

    seeds = {reps[idx] for idx in top_indices if reps[idx] in kg}
    log.info(f"Module-level seeding: {len(seeds)} seeds from top-{top_k_module} modules.")
    return seeds


def gated_bfs_expansion(
    seed_node_names: Set[str],
    graph_index: Graph,
    query: str,
    embedder: TextEmbeddingProvider,
    subtree_nodes,
    max_nodes: int = 50,
    causal_boost: float = 2.0,
    gate_boost: float = 3.0,
) -> Tuple[List[Tuple[int, float]], List[str]]:
    """
    Priority-queue BFS expansion from seed nodes, boosting causal edges.

    Returns (sorted_ranked_list, res_entities) matching the format of
    Retriever.graph_reranker() for drop-in compatibility:
      - sorted_ranked_list: [(tree_node_id, aggregated_score), ...] descending
      - res_entities: [entity_node_name, ...] top retrieved entities
    """
    kg = graph_index.kg

    def _node_score(node: str, edge_type: str, causal_weight: float) -> float:
        node_text = kg.nodes[node].get("description", "") if node in kg else ""
        if not node_text:
            node_text = kg.nodes[node].get("entity_name", node) if node in kg else node
        try:
            semantic = embedder.compute_texts_sim(query, node_text)
            semantic = (semantic + 1) / 2  # normalize [-1,1] -> [0,1]
        except Exception:
            semantic = 0.5

        if edge_type == "causal_gate":
            edge_factor = gate_boost
        elif causal_weight > 0:
            edge_factor = causal_boost * causal_weight
        else:
            edge_factor = 1.0

        return semantic * edge_factor

    visited: Set[str] = set()
    retrieved: List[str] = []

    # Initialize heap with seeds (negate score for max-heap)
    heap = []
    for seed in seed_node_names:
        if seed in kg:
            score = _node_score(seed, "entity", 0.0)
            heapq.heappush(heap, (-score, seed))

    while heap and len(retrieved) < max_nodes:
        neg_score, node = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)
        retrieved.append(node)

        for neighbor in kg.neighbors(node):
            if neighbor in visited:
                continue
            edge_data = kg.get_edge_data(node, neighbor) or {}
            causal_w = edge_data.get("causal_confidence", 0.0) if edge_data.get("is_causal") else 0.0
            etype = edge_data.get("edge_type", "entity")
            score = _node_score(neighbor, etype, causal_w)
            heapq.heappush(heap, (-score, neighbor))

    log.info(f"Gated BFS expansion: retrieved {len(retrieved)} entities.")

    # Aggregate scores to tree node IDs (same logic as graph_reranker)
    target_tree_ids = {node.index_id for node in subtree_nodes}
    tree_node_scores: Dict[int, float] = {tid: 0.0 for tid in target_tree_ids}

    # Use BFS order as a proxy for priority: earlier = higher score
    n = len(retrieved)
    for rank, entity_name in enumerate(retrieved):
        score = (n - rank) / n  # descending from 1.0 to ~0
        node_attrs = kg.nodes.get(entity_name, {})
        source_ids = node_attrs.get("source_ids", set())
        for src_id in source_ids:
            if src_id in target_tree_ids:
                tree_node_scores[src_id] += score

    sorted_ranked_list = sorted(
        tree_node_scores.items(), key=lambda x: x[1], reverse=True
    )

    # Return top max_nodes entities as res_entities
    res_entities = retrieved[:max_nodes]

    return sorted_ranked_list, res_entities

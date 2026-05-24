import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from Core.Index.Graph import Graph
from Core.provider.llm import LLM
from Core.prompts.hugrag_prompt import CAUSAL_EDGE_LABELING_PROMPT, CausalEdgeLabel

log = logging.getLogger(__name__)


def label_causal_edges(graph_index: Graph, llm: LLM, max_workers: int = 8) -> Graph:
    """
    Label every edge in the knowledge graph as causal or associative via LLM.
    Skips edges already labeled. Writes is_causal and causal_confidence back
    to each edge attribute in graph_index.kg.
    """
    edges = list(graph_index.kg.edges(data=True))
    total = len(edges)
    log.info(f"Causal edge labeling: {total} edges to process.")

    to_label = []
    skipped = 0
    for u, v, data in edges:
        if "is_causal" in data and data.get("edge_type", "entity") != "entity":
            skipped += 1
            continue
        # skip edges that were already labeled in a previous run
        if data.get("causal_confidence", 0.0) > 0.0:
            skipped += 1
            continue
        to_label.append((u, v, data))

    log.info(f"Skipping {skipped} already-labeled edges. Labeling {len(to_label)} edges.")

    def _label_one(u, v, data):
        relation = data.get("relation_name", data.get("relation", "related to"))
        description = data.get("description", "")
        context = description[:300] if description else ""

        prompt = CAUSAL_EDGE_LABELING_PROMPT.format(
            entity_a=u,
            relation=relation,
            entity_b=v,
            context=context,
        )
        try:
            result: CausalEdgeLabel = llm.get_json_completion(prompt, CausalEdgeLabel)
            if result is None:
                return u, v, False, 0.0
            return u, v, result.is_causal, result.confidence
        except Exception as e:
            log.warning(f"Causal labeling failed for edge ({u}, {v}): {e}")
            return u, v, False, 0.0

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_label_one, u, v, data): (u, v)
            for u, v, data in to_label
        }
        for i, future in enumerate(as_completed(futures)):
            u, v, is_causal, confidence = future.result()
            results.append((u, v, is_causal, confidence))
            if (i + 1) % 50 == 0:
                log.info(f"Labeled {i + 1}/{len(to_label)} edges...")

    # Write results back to graph
    for u, v, is_causal, confidence in results:
        if graph_index.kg.has_edge(u, v):
            graph_index.kg[u][v]["is_causal"] = is_causal
            graph_index.kg[u][v]["causal_confidence"] = confidence if is_causal else 0.0

    causal_count = sum(1 for _, _, c, _ in results if c)
    log.info(
        f"Causal edge labeling complete. "
        f"{causal_count}/{len(to_label)} new edges labeled causal "
        f"({100 * causal_count / max(len(to_label), 1):.1f}%)."
    )

    # Print summary including already-labeled
    all_edges = list(graph_index.kg.edges(data=True))
    total_causal = sum(1 for _, _, d in all_edges if d.get("is_causal", False))
    log.info(
        f"Total graph: {len(all_edges)} edges, {total_causal} causal "
        f"({100 * total_causal / max(len(all_edges), 1):.1f}%)."
    )

    return graph_index

import logging
from typing import List

import networkx as nx

from Core.provider.llm import LLM
from Core.prompts.hugrag_prompt import (
    CAUSAL_PATH_PROMPT,
    CAUSAL_ANSWER_PROMPT,
    CausalPathResult,
)

log = logging.getLogger(__name__)


def linearize_subgraph(subgraph: nx.Graph) -> str:
    """Convert subgraph to a text list of triples with [CAUSAL]/[ASSOC] markers."""
    lines = []
    for u, v, data in subgraph.edges(data=True):
        relation = data.get("relation_name", data.get("relation", "relates to"))
        marker = "[CAUSAL]" if data.get("is_causal", False) else "[ASSOC]"
        lines.append(f"- {u} --{marker}--> {v} (via: {relation})")
    return "\n".join(lines) if lines else "(no edges in subgraph)"


def identify_causal_path(
    query: str, subgraph: nx.Graph, llm: LLM
) -> CausalPathResult:
    """Ask the LLM to identify the causal chain through the subgraph."""
    subgraph_text = linearize_subgraph(subgraph)
    prompt = CAUSAL_PATH_PROMPT.format(query=query, subgraph_text=subgraph_text)
    try:
        result: CausalPathResult = llm.get_json_completion(prompt, CausalPathResult)
        if result is None:
            return CausalPathResult()
        log.info(
            f"Causal path identified: {len(result.causal_path)} nodes, "
            f"{len(result.spurious_nodes)} spurious."
        )
        return result
    except Exception as e:
        log.warning(f"Causal path identification failed: {e}")
        return CausalPathResult()


def build_causal_answer_prompt(
    query: str, causal_result: CausalPathResult, subgraph: nx.Graph
) -> str:
    """Build the final answer prompt grounded only in causal-path nodes."""
    path_lines = []
    for node_name in causal_result.causal_path:
        if node_name in subgraph.nodes:
            desc = subgraph.nodes[node_name].get("description", "")
            entity_name = subgraph.nodes[node_name].get("entity_name", node_name)
            path_lines.append(f"- {entity_name}: {desc}" if desc else f"- {entity_name}")
        else:
            path_lines.append(f"- {node_name}")

    causal_path_text = "\n".join(path_lines) if path_lines else "(empty causal path)"
    spurious_nodes = (
        ", ".join(causal_result.spurious_nodes)
        if causal_result.spurious_nodes
        else "none"
    )

    return CAUSAL_ANSWER_PROMPT.format(
        query=query,
        causal_path_text=causal_path_text,
        spurious_nodes=spurious_nodes,
    )


def answer_with_causal_path(
    query: str,
    subgraph: nx.Graph,
    llm: LLM,
):
    """
    Full HugRAG generation step:
    1. Identify causal path through retrieved subgraph.
    2. Generate answer grounded only in causal-path nodes.

    Returns (answer_str, partial_answers_list).
    """
    causal_result = identify_causal_path(query, subgraph, llm)
    prompt = build_causal_answer_prompt(query, causal_result, subgraph)

    try:
        answer = llm.get_completion(prompt)
    except Exception as e:
        log.error(f"HugRAG answer generation failed: {e}")
        answer = "Not Answerable."

    partial_answers = [
        {
            "source": "hugrag_causal_path",
            "content": answer,
            "causal_path": causal_result.causal_path,
            "spurious_nodes": causal_result.spurious_nodes,
            "path_explanation": causal_result.path_explanation,
        }
    ]
    return answer, partial_answers

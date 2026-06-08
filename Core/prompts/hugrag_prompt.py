from typing import List
from pydantic import BaseModel, Field


# --- Pydantic schemas for structured LLM responses ---

class CausalEdgeLabel(BaseModel):
    is_causal: bool
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = Field(default="")


class CausalGateResult(BaseModel):
    has_causal_link: bool
    direction: str = Field(default="A->B")  # "A->B" | "B->A" | "bidirectional"
    description: str = Field(default="")


class CausalPathResult(BaseModel):
    causal_path: List[str] = Field(default_factory=list)
    spurious_nodes: List[str] = Field(default_factory=list)
    path_explanation: str = Field(default="")


# --- Prompt templates ---

CAUSAL_EDGE_LABELING_PROMPT = """You are given two entities and a relationship extracted from a document.
Determine whether the relationship represents a CAUSAL relationship or merely ASSOCIATIVE.

A CAUSAL relationship means A directly causes, leads to, produces, enables, requires, drives,
results in, or influences B in a mechanistic or functional sense.
An ASSOCIATIVE relationship means A and B co-occur, are mentioned together, or are related
without one directly causing the other.

When in doubt, prefer marking the relationship as CAUSAL (is_causal: true) with lower confidence.

Entity A: {entity_a}
Relation: {relation}
Entity B: {entity_b}
Context: {context}

Respond with a single valid JSON object only. No extra text.

JSON structure:
{{
  "is_causal": true or false,
  "confidence": 0.0 to 1.0,
  "reason": "one sentence explanation"
}}
"""

CROSS_MODULE_GATE_PROMPT = """Given two knowledge modules (groups of related entities), determine if there is
a meaningful CAUSAL link between them — i.e., events in Module A causally influence events in Module B.

Module A summary: {module_a_summary}
Module B summary: {module_b_summary}

Respond with a single valid JSON object only. No extra text.

JSON structure:
{{
  "has_causal_link": true or false,
  "direction": "A->B" or "B->A" or "bidirectional",
  "description": "one sentence describing the causal link, or empty string if no link"
}}
"""

CAUSAL_PATH_PROMPT = """You are given a query and a knowledge subgraph with two types of edges:
- [CAUSAL]: a causal relationship (A directly causes B)
- [ASSOC]: an associative relationship (A and B are related, but not necessarily causal)

Query: {query}

Subgraph:
{subgraph_text}

Task:
1. Identify the causal chain of nodes that most directly answers the query.
   Focus on [CAUSAL] edges. Only use [ASSOC] edges if no causal path exists.
2. List any nodes you believe are SPURIOUS (co-occur with the answer but are not
   part of the causal explanation).

Respond with a single valid JSON object only. No extra text.

JSON structure:
{{
  "causal_path": ["node1", "node2", ...],
  "spurious_nodes": ["nodeA", "nodeB", ...],
  "path_explanation": "one sentence describing the causal chain"
}}
"""

TREE_CROSS_SECTION_GATE_PROMPT = """Given two document sections, determine if there is a meaningful CAUSAL link between them — i.e., content in Section A causally influences, explains, or leads to the content in Section B.

Section A summary: {section_a_summary}
Section B summary: {section_b_summary}

Respond with a single valid JSON object only. No extra text.

JSON structure:
{{
  "has_causal_link": true or false,
  "direction": "A->B" or "B->A" or "bidirectional",
  "description": "one sentence describing the causal link, or empty string if no link"
}}
"""

CAUSAL_ANSWER_PROMPT = """You are a precise question-answering assistant.
Answer the query based on the retrieved knowledge below.

Query: {query}

Primary context (causal path — prefer this):
{causal_path_text}

Additional context (use if the primary context is insufficient):
{additional_context}

Nodes less likely to be relevant (treat with lower priority):
{spurious_nodes}

Instructions:
- Prefer information from the primary context (causal path).
- If the primary context is insufficient, draw on the additional context.
- Keep your answer concise and factual.
- If none of the context contains the answer, give your best estimate based on what is available.

Answer:
"""

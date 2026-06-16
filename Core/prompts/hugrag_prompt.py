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


class CausalTreePathResult(BaseModel):
    causal_node_ids: List[int] = Field(default_factory=list)
    spurious_node_ids: List[int] = Field(default_factory=list)
    explanation: str = Field(default="")


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

TREE_CROSS_SECTION_GATE_PROMPT = """Given two document sections, determine if there is a DIRECT CAUSAL link between them.

A causal link means: the content of one section mechanistically causes, produces, requires, or directly enables the content of the other. This is NOT satisfied by mere topical relatedness, co-occurrence, or sequential ordering in a paper.

Be conservative: if you are uncertain, answer false.

Examples of TRUE causal links:
- Section A introduces a method → Section B reports results that are directly produced by that method
- Section A identifies a problem/limitation → Section B proposes a solution that directly addresses it
- Section A defines a hypothesis → Section B provides experimental evidence that directly tests it

Examples of FALSE (topically related but not causal):
- Both sections discuss the same dataset or domain
- Both sections are about the same general topic or background
- Section B appears after Section A in the document
- Section B cites or references Section A without being produced by it
- Section A and B share terminology but neither produces the other

Section A summary: {section_a_summary}
Section B summary: {section_b_summary}

Respond with a single valid JSON object only. No extra text.

JSON structure:
{{
  "has_causal_link": true or false,
  "direction": "A->B" or "B->A" or "bidirectional",
  "description": "one sentence describing the specific causal mechanism, or empty string if no link"
}}
"""

TREE_CAUSAL_PATH_PROMPT = """You are given a user query and a list of document sections retrieved from a paper. Your task is to identify which sections are on the CAUSAL PATH to answering the query, and which are SPURIOUS (co-occurring but not causally necessary for the answer).

A section is on the causal path if it directly contains or causally enables the information needed to answer the query. A section is spurious if it is topically related but does not contribute to or produce the answer.

Query: {query}

Retrieved sections:
{nodes_text}

Respond with a single valid JSON object only. No extra text.

JSON structure:
{{
  "causal_node_ids": [list of integer node IDs on the causal path],
  "spurious_node_ids": [list of integer node IDs that are spurious],
  "explanation": "one sentence describing the causal chain"
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

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
Determine whether the relationship represents a CAUSAL relationship
(A directly causes, leads to, or produces B) or merely ASSOCIATIVE
(A and B co-occur or are related, but one doesn't cause the other).

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

CAUSAL_ANSWER_PROMPT = """You are a precise question-answering assistant.
Answer the query using ONLY the information in the verified causal path.
Do NOT use information from the spurious nodes listed below.

Query: {query}

Verified causal path:
{causal_path_text}

Spurious nodes to ignore:
{spurious_nodes}

Instructions:
- Base your answer strictly on the causal path.
- If the causal path does not contain enough information, say so explicitly.
- Keep your answer concise and factual.

Answer:
"""

"""
Lightweight multi-hop reasoning utilities.

This module contains pure-logic helpers and the decision prompt template
used by MultiHopReasoner.  It has **no** heavy dependencies (no langchain,
no neo4j, no dotenv) so that unit tests can import it directly.
"""

import json
import re

# ---------------------------------------------------------------------------
# Prompt template (plain string – no PromptTemplate wrapper)
# ---------------------------------------------------------------------------
MULTI_HOP_DECISION_TEMPLATE = """Task: You are a knowledge graph reasoning agent performing multi-hop reasoning.
Based on the original question, the current node context, and accumulated evidence so far,
choose ONE of the following actions:

A. CONTINUE - Continue exploring the current node to gather more information (e.g., different properties or relationships).
B. JUMP - Jump to a different node to explore related information. You MUST specify which node to jump to (provide type and identifier).
C. ANSWER - Sufficient evidence has been collected. Terminate and produce the final answer.
D. OVERVIEW - Do not focus on any specific node. Instead, perform a global overview query across the knowledge graph.

Instructions:
- Only output JSON, no markdown.
- For option B, you MUST provide "jump_target" with "node_type" and "identifier".
- For option A, optionally suggest "focus_hint" to guide what aspect to explore next.
- For option D, optionally suggest "overview_hint" to guide what global pattern to look for.

Return exactly one JSON object:
{{"action": "A" or "B" or "C" or "D", "reason": "...", "jump_target": {{"node_type": "...", "identifier": "..."}} or null, "focus_hint": "..." or null, "overview_hint": "..." or null}}

Original Question:
{question}

Current Node Context:
{current_node}

Accumulated Evidence (from previous hops):
{evidence}

Step {step} of maximum {max_steps} steps.
"""


# ---------------------------------------------------------------------------
# Pure-logic helpers (no external dependencies)
# ---------------------------------------------------------------------------

def parse_decision(raw_text: str) -> dict:
    """Parse the LLM decision JSON, tolerating minor formatting issues."""
    try:
        return json.loads(raw_text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    return {"action": "C", "reason": "Failed to parse decision, defaulting to ANSWER"}


def summarize_evidence(evidence: list) -> str:
    """Summarize collected evidence for the prompt context."""
    if not evidence:
        return "No evidence collected yet."
    try:
        return json.dumps(evidence[:10], ensure_ascii=False, indent=1)
    except Exception:
        return str(evidence[:10])

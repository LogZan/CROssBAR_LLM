"""
Query Examples Library for CROssBARv2 Knowledge Graph.

Provides common Cypher query patterns and examples to guide LLM generation.
"""

from typing import Dict, List

# Common query patterns with explanations
QUERY_EXAMPLES: Dict[str, Dict] = {
    "protein_by_gene": {
        "description": "Find protein encoded by a gene",
        "question": "Find the protein encoded by gene EGFR",
        "cypher": """MATCH (g:Gene {geneName: 'EGFR'})-[:Gene_encodes_protein]->(p:Protein)
RETURN p.primaryAccession, p.geneName, p.uniProtkbId""",
        "explanation": "Use Gene node's geneName property (searchable) for exact match",
    },
    "protein_by_accession": {
        "description": "Find protein by UniProt accession",
        "question": "Get information about protein P00533",
        "cypher": """MATCH (p:Protein {primaryAccession: 'P00533'})
RETURN p.primaryAccession, p.geneName, p.fullName, p.sequence""",
        "explanation": "Use primaryAccession (primary key) for direct protein lookup",
    },
    "enzyme_classification": {
        "description": "Query protein's enzyme classification",
        "question": "What is the enzyme classification of protein P00533?",
        "cypher": """MATCH (p:Protein {primaryAccession: 'P00533'})
      -[:Protein_has_catalytic_activity]->(ca:CatalyticActivityAnnotation)
RETURN ca.ecNumber, ca.texts""",
        "explanation": "Use Protein_has_catalytic_activity relationship to find EC numbers",
    },
    "gene_disease_association": {
        "description": "Find diseases associated with a gene",
        "question": "What diseases are associated with gene BRCA1?",
        "cypher": """MATCH (g:Gene {geneName: 'BRCA1'})
      -[:Gene_associated_with_disease]->(d:Disease)
RETURN d.diseaseId, d.diseaseName""",
        "explanation": "Direct gene-disease association through relationship",
    },
    "protein_function": {
        "description": "Query protein's molecular function",
        "question": "What is the molecular function of EGFR protein?",
        "cypher": """MATCH (g:Gene {geneName: 'EGFR'})-[:Gene_encodes_protein]->(p:Protein)
OPTIONAL MATCH (p)-[:Protein_has_function]->(f:FunctionAnnotation)
RETURN f.texts AS functions""",
        "explanation": "Use OPTIONAL MATCH for annotations that may not exist",
    },
    "subunit_structure": {
        "description": "Query protein's oligomeric state",
        "question": "What is the oligomeric state of FGFR1?",
        "cypher": """MATCH (g:Gene {geneName: 'FGFR1'})-[:Gene_encodes_protein]->(p:Protein)
OPTIONAL MATCH (p)-[r:Protein_has_subunit]->(sa:SubunitAnnotation)
RETURN r.texts AS subunit_description, sa.texts AS subunit_info""",
        "explanation": "Subunit information can be in relationship or annotation node",
    },
    "protein_location": {
        "description": "Find protein subcellular location",
        "question": "Where is protein P12345 located in the cell?",
        "cypher": """MATCH (p:Protein {primaryAccession: 'P12345'})
      -[:Protein_has_subcellular_location]->(loc:SubcellularLocationAnnotation)
RETURN loc.texts AS locations""",
        "explanation": "Subcellular location through specific relationship",
    },
    "drug_targets": {
        "description": "Find drugs targeting a protein",
        "question": "What drugs target the EGFR protein?",
        "cypher": """MATCH (g:Gene {geneName: 'EGFR'})-[:Gene_encodes_protein]->(p:Protein)
      <-[:Drug_targets]-(d:Drug)
RETURN d.drugId, d.drugName""",
        "explanation": "Note the reverse relationship direction (<-)",
    },
    "pathway_membership": {
        "description": "Find pathways containing a gene",
        "question": "Which pathways include gene TP53?",
        "cypher": """MATCH (g:Gene {geneName: 'TP53'})
      -[:Gene_participates_in_pathway]->(pw:Pathway)
RETURN pw.pathwayId, pw.pathwayName""",
        "explanation": "Gene-pathway relationships for functional context",
    },
    "handle_query_failure": {
        "description": "Correct approach when identifier type is unknown",
        "wrong_cypher": "MATCH (p:Protein {id: 'INSR'}) RETURN p",
        "error": "Protein node does not support 'id' property",
        "corrected_cypher": """MATCH (p:Protein)
WHERE p.geneName = 'INSR' OR p.primaryAccession CONTAINS 'INSR'
RETURN p.primaryAccession, p.geneName, p.fullName
LIMIT 10""",
        "explanation": "When unsure, use WHERE clause to search multiple properties",
    },
    "fuzzy_search": {
        "description": "Search with partial matching",
        "question": "Find proteins with 'kinase' in their name",
        "cypher": """MATCH (p:Protein)
WHERE p.fullName CONTAINS 'kinase'
RETURN p.primaryAccession, p.geneName, p.fullName
LIMIT 20""",
        "explanation": "Use CONTAINS for substring matching (case-sensitive)",
    },
    "multi_hop_exploration": {
        "description": "Navigate multiple relationships",
        "question": "Find diseases linked to proteins interacting with EGFR",
        "cypher": """MATCH (g:Gene {geneName: 'EGFR'})-[:Gene_encodes_protein]->(p1:Protein)
      -[:Protein_interacts_with_protein]->(p2:Protein)
      <-[:Gene_encodes_protein]-(g2:Gene)
      -[:Gene_associated_with_disease]->(d:Disease)
RETURN DISTINCT d.diseaseId, d.diseaseName
LIMIT 10""",
        "explanation": "Chain relationships for multi-hop queries, use DISTINCT to avoid duplicates",
    },
}


def get_relevant_examples(question: str, max_examples: int = 3) -> str:
    """
    Return relevant query examples based on the question.

    Args:
        question: The user's question or query intent
        max_examples: Maximum number of examples to return

    Returns:
        Formatted examples string for LLM prompt
    """
    # Simple keyword matching (can be enhanced with embeddings)
    keywords_map = {
        "enzyme": ["enzyme_classification"],
        "catalytic": ["enzyme_classification"],
        "ec number": ["enzyme_classification"],
        "gene": ["protein_by_gene", "gene_disease_association"],
        "protein": ["protein_by_accession", "protein_by_gene"],
        "disease": ["gene_disease_association"],
        "function": ["protein_function"],
        "oligomer": ["subunit_structure"],
        "dimer": ["subunit_structure"],
        "subunit": ["subunit_structure"],
        "location": ["protein_location"],
        "subcellular": ["protein_location"],
        "drug": ["drug_targets"],
        "target": ["drug_targets"],
        "pathway": ["pathway_membership"],
        "interact": ["multi_hop_exploration"],
        "classification": ["enzyme_classification"],
        "accession": ["protein_by_accession"],
        "where": ["protein_location"],
        "what": ["protein_function", "enzyme_classification"],
    }

    relevant_keys = set()
    question_lower = question.lower()

    # Match keywords
    for keyword, example_keys in keywords_map.items():
        if keyword in question_lower:
            relevant_keys.update(example_keys)

    # Always include error handling example
    relevant_keys.add("handle_query_failure")

    # If no matches, provide basic examples
    if len(relevant_keys) == 1:  # Only error handling
        relevant_keys.update(["protein_by_gene", "protein_function"])

    # Limit to max_examples
    relevant_keys = list(relevant_keys)[:max_examples]

    # Format as prompt
    lines = ["## Query Examples\n"]

    for key in relevant_keys:
        ex = QUERY_EXAMPLES[key]
        lines.append(f"### Example: {ex['description']}")

        if "question" in ex:
            lines.append(f"**Question**: {ex['question']}")

        if "wrong_cypher" in ex:
            lines.append(f"**❌ Wrong**:\n```cypher\n{ex['wrong_cypher']}\n```")
            lines.append(f"**Error**: {ex['error']}")

        lines.append(f"**✅ Correct**:\n```cypher\n{ex['cypher'].strip()}\n```")
        lines.append(f"**💡 Tip**: {ex['explanation']}\n")

    return "\n".join(lines)


def get_all_examples() -> str:
    """
    Return all examples formatted for documentation or reference.

    Returns:
        Formatted string with all examples
    """
    lines = ["# CROssBARv2 Cypher Query Examples\n"]

    for key, ex in QUERY_EXAMPLES.items():
        lines.append(f"\n## {ex['description']}")

        if "question" in ex:
            lines.append(f"**Question**: {ex['question']}")

        if "wrong_cypher" in ex:
            lines.append(f"**Wrong**:\n```cypher\n{ex['wrong_cypher']}\n```")
            lines.append(f"**Error**: {ex['error']}")

        if "cypher" in ex:
            lines.append(f"**Correct**:\n```cypher\n{ex['cypher'].strip()}\n```")

        lines.append(f"**Explanation**: {ex['explanation']}\n")

    return "\n".join(lines)

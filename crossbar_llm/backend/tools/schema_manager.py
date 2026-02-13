"""
Schema Manager for CROssBARv2 Knowledge Graph.

This module provides schema management capabilities including:
- Loading and parsing graph schema
- Schema validation for Cypher queries
- Schema-based prompt generation for LLMs
"""

import json
import logging
import os
import re
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class SchemaManager:
    """Manages knowledge graph schema and provides query assistance."""

    def __init__(self, schema_path: Optional[str] = None):
        """
        Initialize SchemaManager.

        Args:
            schema_path: Path to graph_schema.json. If None, looks in current directory.
        """
        if schema_path is None:
            schema_path = os.path.join(os.getcwd(), "graph_schema.json")

        self.schema_path = schema_path
        self.schema = self._load_schema()

        # Build lookup structures for fast access
        self._build_lookups()

    def _load_schema(self) -> Dict:
        """Load schema from JSON file."""
        if not os.path.exists(self.schema_path):
            logger.warning(
                f"Schema file not found at {self.schema_path}. "
                "Schema validation will be limited."
            )
            return {
                "nodes": [],
                "node_properties": [],
                "edges": [],
                "edge_properties": [],
            }

        try:
            with open(self.schema_path, "r") as f:
                schema = json.load(f)
            logger.info(f"Loaded schema from {self.schema_path}")
            return schema
        except Exception as e:
            logger.error(f"Failed to load schema: {e}")
            return {
                "nodes": [],
                "node_properties": [],
                "edges": [],
                "edge_properties": [],
            }

    def _build_lookups(self):
        """Build fast lookup structures from schema."""
        # Extract node labels
        nodes_data = self.schema.get("nodes", [])
        if nodes_data and isinstance(nodes_data[0], dict):
            self.node_labels = set(nodes_data[0].get("labels", []))
        else:
            self.node_labels = set()

        # Build node properties map: {node_label: [property_names]}
        self.node_props_map: Dict[str, Set[str]] = {}
        for node_prop in self.schema.get("node_properties", []):
            label = node_prop.get("labels")
            if label:
                props = node_prop.get("properties", [])
                prop_names = {p.get("property") for p in props if p.get("property")}
                self.node_props_map[label] = prop_names

        # Build edge map: {edge_type: True}
        self.edge_types = set()
        for edge in self.schema.get("edges", []):
            if isinstance(edge, str):
                # Extract edge type from pattern like "(Protein)-[:edge_type]->(Node)"
                match = re.search(r"\[:([^\]]+)\]", edge)
                if match:
                    self.edge_types.add(match.group(1))

        # Build edge properties map
        self.edge_props_map: Dict[str, Set[str]] = {}
        for edge_prop in self.schema.get("edge_properties", []):
            edge_type = edge_prop.get("type")
            if edge_type:
                props = edge_prop.get("properties", [])
                prop_names = {p.get("property") for p in props if p.get("property")}
                self.edge_props_map[edge_type] = prop_names

    def get_node_properties(self, node_type: str) -> List[str]:
        """
        Return all valid properties for a node type.

        Args:
            node_type: The node label (e.g., "Protein", "Gene")

        Returns:
            List of property names
        """
        return list(self.node_props_map.get(node_type, set()))

    def get_relationships(
        self, source_node: Optional[str] = None, target_node: Optional[str] = None
    ) -> List[Dict]:
        """
        Return relationship types, optionally filtered by source/target nodes.

        Args:
            source_node: Optional source node type to filter
            target_node: Optional target node type to filter

        Returns:
            List of relationship dictionaries
        """
        relationships = []
        for edge in self.schema.get("edges", []):
            if isinstance(edge, str):
                # Parse pattern: (SourceNode)-[:EdgeType]->(TargetNode)
                match = re.match(
                    r"\(:(\w+)\)-\[:([^\]]+)\]->\(:(\w+)\)", edge
                )
                if match:
                    src, edge_type, tgt = match.groups()
                    if source_node and src != source_node:
                        continue
                    if target_node and tgt != target_node:
                        continue
                    relationships.append({
                        "type": edge_type,
                        "source": src,
                        "target": tgt,
                        "properties": list(self.edge_props_map.get(edge_type, set())),
                    })
        return relationships

    def get_primary_key(self, node_type: str) -> Optional[str]:
        """
        Return the primary key property for a node type.

        Args:
            node_type: The node label

        Returns:
            Primary key property name, or None if not found

        Note:
            This uses heuristics. Common primary keys:
            - Protein: primaryAccession
            - Gene: geneName
            - Other nodes: typically first unique property
        """
        # Known primary keys for common node types
        primary_keys = {
            "Protein": "primaryAccession",
            "Gene": "geneName",
            "Disease": "diseaseId",
            "Drug": "drugId",
            "Pathway": "pathwayId",
        }

        if node_type in primary_keys:
            return primary_keys[node_type]

        # Fallback: look for common ID patterns in properties
        props = self.get_node_properties(node_type)
        for prop in props:
            prop_lower = prop.lower()
            if any(keyword in prop_lower for keyword in ["id", "accession", "name"]):
                return prop

        # Return first property if available
        return props[0] if props else None

    def get_searchable_properties(self, node_type: str) -> List[str]:
        """
        Return properties commonly used for searching/matching.

        Args:
            node_type: The node label

        Returns:
            List of searchable property names
        """
        # Known searchable properties for common types
        searchable_map = {
            "Protein": ["primaryAccession", "geneName", "uniProtkbId"],
            "Gene": ["geneName", "ensemblId"],
            "Disease": ["diseaseId", "diseaseName"],
            "Drug": ["drugId", "drugName"],
        }

        if node_type in searchable_map:
            # Filter to only properties that exist in schema
            all_props = set(self.get_node_properties(node_type))
            return [p for p in searchable_map[node_type] if p in all_props]

        # Fallback: return primary key and name-like properties
        props = self.get_node_properties(node_type)
        primary = self.get_primary_key(node_type)
        searchable = []

        if primary:
            searchable.append(primary)

        for prop in props:
            prop_lower = prop.lower()
            if "name" in prop_lower or "symbol" in prop_lower:
                if prop not in searchable:
                    searchable.append(prop)

        return searchable if searchable else props[:3]

    def generate_schema_prompt(
        self, relevant_nodes: Optional[List[str]] = None
    ) -> str:
        """
        Generate schema description for LLM prompts.

        Args:
            relevant_nodes: Optional list of node types to focus on

        Returns:
            Formatted schema description string
        """
        if not self.schema or not self.node_labels:
            return "# CROssBARv2 Knowledge Graph Schema\n\nSchema not available."

        lines = ["# CROssBARv2 Knowledge Graph Schema\n"]

        # Filter nodes if requested
        nodes_to_show = (
            relevant_nodes
            if relevant_nodes
            else sorted(list(self.node_labels))[:10]  # Limit to first 10 alphabetically
        )

        # Node information
        for node_type in nodes_to_show:
            if node_type not in self.node_labels:
                continue

            lines.append(f"\n## {node_type} Node")
            primary = self.get_primary_key(node_type)
            if primary:
                lines.append(f"Primary Key: {primary}")

            searchable = self.get_searchable_properties(node_type)
            if searchable:
                lines.append(f"Searchable Properties: {', '.join(searchable)}")

            all_props = self.get_node_properties(node_type)
            if all_props and len(all_props) <= 10:
                lines.append(f"All Properties: {', '.join(all_props)}")
            elif all_props:
                lines.append(f"Properties: {', '.join(all_props[:10])} (+ {len(all_props) - 10} more)")

        # Relationship information
        lines.append("\n## Relationships")
        relationships = self.get_relationships()
        if relationships:
            shown = 0
            for rel in relationships[:20]:  # Limit to avoid huge prompts
                lines.append(
                    f"({rel['source']})-[:{rel['type']}]->({rel['target']})"
                )
                shown += 1
            if len(relationships) > shown:
                lines.append(f"... and {len(relationships) - shown} more relationships")
        else:
            lines.append("(No relationship information available)")

        # IMPORTANT: Annotation data storage guidance
        lines.append("\n## IMPORTANT: Annotation Data Storage Location")
        lines.append("\nMost annotation data is stored in **relationship properties**, NOT in target node properties.")
        lines.append("\n### Correct Query Pattern")
        lines.append("```cypher")
        lines.append("// ✅ CORRECT: Get data from relationship properties")
        lines.append("MATCH (p:Protein {primaryAccession: 'P00533'})")
        lines.append("      -[rel:Protein_has_catalytic_activity]->()")
        lines.append("RETURN rel.ecNumber, rel.name, rel.database")
        lines.append("")
        lines.append("// ❌ WRONG: Trying to get from node properties (returns null)")
        lines.append("MATCH (p:Protein {primaryAccession: 'P00533'})")
        lines.append("      -[:Protein_has_catalytic_activity]->(ca)")
        lines.append("RETURN ca.ecNumber  // Returns null - property doesn't exist on node")
        lines.append("```")
        lines.append("\n### Key Annotation Relationships and Their Properties")
        lines.append("\n**1. Protein_has_catalytic_activity**")
        lines.append("   - Relationship properties: `ecNumber`, `name`, `database`, `commentType`")
        lines.append("   - Example: rel.ecNumber = '2.7.10.1', rel.name = 'L-tyrosyl-[protein] + ATP = ...'")
        lines.append("\n**2. Protein_has_binding_site_feature**")
        lines.append("   - Relationship properties: `name`, `ligandPart_name`, `database`, `label`, `start`, `end`")
        lines.append("   - Example: rel.name = 'chloride', rel.start = '100', rel.end = '105'")
        lines.append("\n**3. Protein_has_cofactor**")
        lines.append("   - Relationship properties: `name`, `database`, `texts`")
        lines.append("   - Example: rel.name = 'Zn(2+)', rel.database = 'ChEBI'")
        lines.append("\n**4. Protein_has_signal_feature**")
        lines.append("   - Relationship properties: `description`, `start`, `end`, `modifier`")
        lines.append("   - Example: rel.description = 'Signal peptide', rel.start = '1', rel.end = '25'")
        lines.append("\n**5. Protein_has_chain_feature**")
        lines.append("   - Relationship properties: `description`, `start`, `end`, `modifier`")
        lines.append("   - Example: rel.description = 'Mature chain', rel.start = '26', rel.end = '500'")
        lines.append("\n**6. Protein_has_subunit**")
        lines.append("   - Relationship properties: `texts`")
        lines.append("   - Example: rel.texts = 'Monomer. Homodimer after ligand binding...'")

        # Query guidelines
        lines.append("\n## Query Guidelines")
        lines.append("1. ALWAYS use searchable properties for node matching")
        lines.append("2. Use primary keys when the exact identifier is known")
        lines.append("3. NEVER use 'id' property unless explicitly listed")
        lines.append("4. **For annotation data: Use -[rel:RelType]->() pattern to access relationship properties**")
        lines.append("5. Keep queries simple (avoid excessive OPTIONAL MATCH)")
        lines.append("6. Validate property names against the schema above")

        return "\n".join(lines)

    def validate_cypher(
        self, cypher_query: str
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Validate a Cypher query against the schema.

        Args:
            cypher_query: The Cypher query to validate

        Returns:
            (is_valid, error_message, suggestion)
        """
        errors = []
        suggestions = []

        if not self.schema or not self.node_labels:
            # Schema not available, cannot validate
            return True, None, None

        # Extract node patterns: (variable:Label {property: value})
        node_patterns = re.findall(
            r"\((\w*):(\w+)\s*(?:\{([^}]+)\})?\)", cypher_query, re.IGNORECASE
        )

        for var, label, props in node_patterns:
            # Check if node label exists
            if label not in self.node_labels:
                errors.append(f"Unknown node type '{label}'")
                similar = self._find_similar_label(label)
                if similar:
                    suggestions.append(f"Did you mean '{similar}'?")
                continue

            # Check properties if specified
            if props:
                valid_props = set(self.get_node_properties(label))
                # Extract property names from the pattern
                prop_names = re.findall(r'(\w+)\s*:', props)

                for prop_name in prop_names:
                    if prop_name not in valid_props:
                        errors.append(
                            f"{label} node does not support property '{prop_name}'"
                        )
                        # Suggest searchable properties
                        searchable = self.get_searchable_properties(label)
                        if searchable:
                            suggestions.append(
                                f"For {label}, use: {', '.join(searchable)}"
                            )

        # Check relationship types: -[:RelType]->
        rel_patterns = re.findall(r'-\[:(\w+)\]->', cypher_query, re.IGNORECASE)

        for rel_type in rel_patterns:
            if rel_type not in self.edge_types:
                errors.append(f"Unknown relationship type '{rel_type}'")
                # Try to find similar
                similar = self._find_similar_edge(rel_type)
                if similar:
                    suggestions.append(f"Did you mean '{similar}'?")

        # Check for common mistakes
        if re.search(r'\{\s*id\s*:', cypher_query, re.IGNORECASE):
            # Check if any node actually has 'id' property
            has_id = any('id' in self.node_props_map.get(node, set())
                        for node in self.node_labels)
            if not has_id:
                errors.append("Using 'id' property, but most nodes don't support it")
                suggestions.append("Use node-specific identifiers (e.g., geneName, primaryAccession)")

        # Check for excessive OPTIONAL MATCH
        optional_count = len(re.findall(r'\bOPTIONAL\s+MATCH\b', cypher_query, re.IGNORECASE))
        if optional_count > 10:
            errors.append(f"Query too complex ({optional_count} OPTIONAL MATCH clauses)")
            suggestions.append("Consider simplifying or splitting into multiple queries")

        if errors:
            return False, "; ".join(errors), "; ".join(suggestions) if suggestions else None

        return True, None, None

    # Abstract / parent node labels that should be excluded when
    # presenting the schema to the LLM.  These appear in the graph
    # because Neo4j stores the full label hierarchy, but using them in
    # Cypher queries leads to overly broad or incorrect results.
    ABSTRACT_NODE_LABELS: Set[str] = {
        "NamedThing", "Entity", "BiologicalEntity",
        "BiologicalProcessOrActivity", "ChemicalEntity",
        "ChemicalMixture", "MolecularMixture",
        "DiseaseOrPhenotypicFeature", "PhenotypicFeature",
        "Polypeptide", "Description", "FunctionalAnnotation",
        "SequenceFeature", "GeneOntology",
    }

    def format_schema_for_llm(self) -> str:
        """
        Produce a clean, deduplicated schema description for LLM prompts.

        Compared to the raw ``graph_schema.json`` dump this method:

        * Removes abstract / parent-class labels (e.g. ``NamedThing``,
          ``Entity``) from node lists and relationship endpoints so the LLM
          only sees concrete types.
        * Deduplicates relationship patterns – for each relationship type only
          the most specific (source, target) combination is kept.
        * Clearly separates **node types** (with their properties) from
          **relationship types** (with their properties) so the LLM does not
          confuse nodes and relationships.

        Returns:
            A formatted string suitable for injection into an LLM prompt.
        """
        if not self.schema:
            return ""

        lines: List[str] = []

        # ---- 1. Concrete node types with properties ----
        concrete_nodes = sorted(
            label for label in self.node_labels
            if label not in self.ABSTRACT_NODE_LABELS
        )

        lines.append("## Node types (use these as node labels in MATCH patterns)")
        lines.append("The following are NODE types. Use them ONLY inside (parentheses) in Cypher, e.g. (p:Protein).")
        lines.append("NEVER use a node type inside square brackets [] – those are for relationships only.\n")

        for node in concrete_nodes:
            props = sorted(self.node_props_map.get(node, set()))
            if props:
                lines.append(f"  {node}: {', '.join(props)}")
            else:
                lines.append(f"  {node}")

        # ---- 2. Deduplicated relationships ----
        lines.append("\n## Relationship types (use these as relationship labels in MATCH patterns)")
        lines.append("The following are RELATIONSHIP types. Use them ONLY inside [square brackets] in Cypher, e.g. -[:Drug_targets_protein]->.")
        lines.append("NEVER use a relationship type inside (parentheses) – those are for nodes only.\n")

        # Build deduplicated edge list: for each relationship type keep
        # only edges whose source AND target are both concrete.
        # When all edges for a relationship type involve only abstract labels,
        # fall back to the most specific abstract label for each endpoint.
        edge_map: Dict[str, List[Tuple[str, str]]] = {}
        edge_map_all: Dict[str, List[Tuple[str, str]]] = {}
        for edge_str in self.schema.get("edges", []):
            if not isinstance(edge_str, str):
                continue
            match = re.match(
                r"\(:(\w+)\)-\[:([^\]]+)\]->\(:(\w+)\)", edge_str
            )
            if not match:
                continue
            src, rel_type, tgt = match.groups()
            edge_map_all.setdefault(rel_type, []).append((src, tgt))
            if src in self.ABSTRACT_NODE_LABELS or tgt in self.ABSTRACT_NODE_LABELS:
                continue
            edge_map.setdefault(rel_type, []).append((src, tgt))

        # For relationship types with no concrete edges, infer the best
        # concrete pair from the relationship name and available abstract edges.
        for rel_type, all_pairs in edge_map_all.items():
            if rel_type in edge_map:
                continue  # already has concrete edges
            best_src = self._infer_concrete_label(
                [s for s, _ in all_pairs], rel_type, position="source"
            )
            best_tgt = self._infer_concrete_label(
                [t for _, t in all_pairs], rel_type, position="target"
            )
            if best_src and best_tgt:
                edge_map[rel_type] = [(best_src, best_tgt)]

        # Deduplicate within each relationship type
        for rel_type in sorted(edge_map):
            pairs = sorted(set(edge_map[rel_type]))
            props = sorted(self.edge_props_map.get(rel_type, set()))
            pair_strs = ", ".join(f"({s})->({t})" for s, t in pairs)
            if props:
                lines.append(f"  {rel_type}  [{pair_strs}]  properties: {', '.join(props)}")
            else:
                lines.append(f"  {rel_type}  [{pair_strs}]")

        # ---- 3. Critical rules ----
        lines.append("\n## Critical rules for Cypher generation")
        lines.append("1. Use ONLY the node types listed above inside (parentheses).")
        lines.append("2. Use ONLY the relationship types listed above inside [square brackets].")
        lines.append("3. Do NOT use a relationship type as a node label and vice versa.")
        lines.append("4. Use ONLY the properties listed for each node/relationship type.")
        lines.append("5. Do NOT use properties from one node type on a different node type.")
        lines.append("6. Do NOT use properties from a relationship on a node or vice versa.")
        lines.append("7. For annotation data (catalytic activity, cofactor, binding site, etc.),")
        lines.append("   the data is stored in RELATIONSHIP properties, NOT in the target node.")
        lines.append("   Use  -[rel:Relationship_type]->()  and access  rel.property_name.")

        return "\n".join(lines)

    def _find_similar_label(self, label: str) -> Optional[str]:
        """Find similar node label using simple string matching."""
        label_lower = label.lower()
        for node_label in self.node_labels:
            if label_lower in node_label.lower() or node_label.lower() in label_lower:
                return node_label
        return None

    def _find_similar_edge(self, edge_type: str) -> Optional[str]:
        """Find similar edge type using simple string matching."""
        edge_lower = edge_type.lower()
        for et in self.edge_types:
            if edge_lower in et.lower() or et.lower() in edge_lower:
                return et
        return None

    def _infer_concrete_label(
        self, labels: List[str], rel_type: str, position: str
    ) -> Optional[str]:
        """Infer the best concrete node label for a relationship endpoint.

        When all edge endpoints for a relationship type are abstract, try to
        derive the intended concrete label from the relationship name (e.g.
        ``Protein_has_toxic_dose`` → source is ``Protein``) and from the
        set of available labels (pick the most specific non-abstract one).
        """
        # 1. Try to pick a non-abstract label from the list
        concrete = [l for l in labels if l not in self.ABSTRACT_NODE_LABELS]
        if concrete:
            return max(set(concrete), key=len)

        # 2. Infer from relationship name for the source position
        if position == "source":
            # E.g. "Protein_has_toxic_dose" → "Protein"
            prefix = rel_type.split("_")[0]
            if prefix in self.node_labels and prefix not in self.ABSTRACT_NODE_LABELS:
                return prefix

        # 3. For target: pick the label that is NOT in
        #    ABSTRACT_NODE_LABELS *and* exists as a node type.
        #    Fall back to the least-generic abstract label by
        #    preferring labels that appear fewer times (more specific).
        if labels:
            unique = sorted(set(labels))
            # Prefer non-abstract first (should have been caught above, but be safe)
            for c in unique:
                if c not in self.ABSTRACT_NODE_LABELS and c in self.node_labels:
                    return c
            # All labels are abstract: rank by how many edges use them
            #   (more niche abstract labels are more specific)
            label_freq = {}
            for l in labels:
                label_freq[l] = label_freq.get(l, 0) + 1
            # Least frequent = most specific
            candidates = sorted(unique, key=lambda x: label_freq.get(x, 0))
            for c in candidates:
                if c in self.node_labels:
                    return c

        return None

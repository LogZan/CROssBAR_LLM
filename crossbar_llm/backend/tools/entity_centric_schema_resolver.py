"""
EntityCentricSchemaResolver: Entity-centric schema filtering for improved Cypher query generation.

This module implements a three-step process:
1. Entity Identification: Detect protein sequences and extract entity information
2. Node Schema Extraction: Locate specific nodes and query their properties/edges
3. Schema Merging: Combine multiple node schemas with limits
"""

import json
import logging
import os
import time
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .qa_templates import ENTITY_IDENTIFICATION_PROMPT, NODE_SCHEMA_EXTRACTION_PROMPT


class EntityCentricSchemaResolver:
    """
    Resolves entity-centric schema by identifying specific entities in questions
    and extracting their node-level schema from the knowledge graph.
    """

    def __init__(
        self,
        llm_client,
        neo4j_helper,
        cache_dir: str = "cache",
        config: Optional[Dict] = None
    ):
        """
        Initialize the EntityCentricSchemaResolver.

        Args:
            llm_client: LLM client for entity identification (gpt-oss-120b)
            neo4j_helper: Neo4jGraphHelper instance for querying the graph
            cache_dir: Base directory for caching schemas
            config: Configuration dict with max_entities, max_edge_types, etc.
        """
        self.llm_client = llm_client
        self.neo4j_helper = neo4j_helper
        self.cache_dir = Path(cache_dir)
        self.config = config or {}

        # Configuration parameters
        self.max_entities = self.config.get("max_entities", 5)
        self.max_edge_types = self.config.get("max_edge_types", 200)
        self.cache_ttl_hours = self.config.get("cache_ttl_hours", 24)
        self.similarity_threshold = self.config.get("similarity_threshold", 0.7)
        self.schema_config_path = self.config.get(
            "schema_config_path",
            "/GenSIvePFS/users/clzeng/workspace/CROssBARv2-KG/config/schema_config.yaml",
        )
        self.label_hierarchy, self.label_aliases = self._load_label_hierarchy(self.schema_config_path)

        # Ensure cache directory exists
        self.entity_cache_dir = self.cache_dir / "entity_schema_cache"
        self.entity_cache_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"EntityCentricSchemaResolver initialized with cache_dir={self.entity_cache_dir}")

    def resolve(self, question: str, max_retries: int = 3) -> Optional[Dict]:
        """
        Main entry point: Resolve entity-centric schema for a question.

        Args:
            question: User's question
            max_retries: Maximum retry attempts

        Returns:
            Filtered schema dict or None (triggers fallback to full schema)
        """
        for attempt in range(max_retries):
            try:
                # Step 1: Identify entities
                entity_info = self._identify_entity(question)
                if not entity_info or entity_info.get("confidence", 0) < 0.5:
                    logging.info("Entity identification failed or low confidence, using full schema")
                    return None

                # Step 2: Extract node schemas
                node_schemas = []
                anchor_entities: List[Dict] = []
                entities = entity_info.get("entities", [])[:self.max_entities]

                for entity in entities:
                    # Try to locate the node
                    node_info = self._locate_node(entity)
                    if not node_info:
                        logging.warning(f"Could not locate node for entity: {entity}")
                        continue
                    node_id = node_info.get("id")
                    node_type = node_info.get("type") or entity.get("type", "Protein")
                    if not node_id:
                        logging.warning(f"Located node missing id for entity: {entity}")
                        continue
                    anchor = {
                        "id": node_id,
                        "type": node_type,
                        "identifier": entity.get("identifier"),
                        "name": entity.get("name"),
                        "sequence": entity.get("sequence"),
                    }
                    anchor_entities.append(anchor)

                    # Check cache first
                    cached_schema = self._load_cached_schema(node_id)
                    if cached_schema:
                        logging.info(f"Using cached schema for node: {node_id}")
                        schema = dict(cached_schema)
                        if "target_node_context" not in schema:
                            schema["target_node_context"] = self._build_target_context(schema, node_id, node_type)
                        schema.setdefault("anchor_entities", []).append(anchor)
                        node_schemas.append(schema)
                    else:
                        # Extract schema from Neo4j
                        schema = self._extract_node_schema(node_id, node_type)
                        if schema:
                            schema["target_node_context"] = self._build_target_context(schema, node_id, node_type)
                            self._cache_schema(node_id, schema)
                            schema = dict(schema)
                            schema.setdefault("anchor_entities", []).append(anchor)
                            node_schemas.append(schema)

                if not node_schemas:
                    logging.warning("No node schemas extracted, using full schema")
                    return None

                # Step 3: Merge schemas
                merged_schema = self._merge_schemas(node_schemas)
                if anchor_entities:
                    merged_schema["anchor_entities"] = anchor_entities
                logging.info(f"Successfully resolved entity-centric schema with {len(merged_schema.get('edges', []))} edge types")
                return merged_schema

            except Exception as e:
                logging.error(f"Entity resolution attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    logging.error("Max retries exceeded, falling back to full schema")
                    return None
                time.sleep(2 ** attempt)  # Exponential backoff

        return None

    def _load_label_hierarchy(self, config_path: str) -> tuple[dict, dict]:
        parent_map = {}
        aliases = {}
        if not config_path or not os.path.exists(config_path):
            return parent_map, aliases
        try:
            with open(config_path, "r") as f:
                data = yaml.safe_load(f) or {}
            for key, value in data.items():
                if not isinstance(value, dict):
                    continue
                norm_key = self._normalize_label(key)
                label_in_input = value.get("label_in_input")
                if label_in_input:
                    aliases[self._normalize_label(label_in_input)] = norm_key
                synonym_for = value.get("synonym_for")
                is_a = value.get("is_a")
                parent = synonym_for or is_a
                if parent:
                    parent_map[norm_key] = self._normalize_label(parent)
        except Exception as e:
            logging.warning(f"Failed to load schema_config.yaml for label hierarchy: {e}")
        return parent_map, aliases

    @staticmethod
    def _normalize_label(label: str) -> str:
        return "".join(ch for ch in label.lower() if ch.isalnum())

    def _canonical_label(self, label: str) -> str:
        return self.label_aliases.get(label, label)

    def _label_depth(self, label: str) -> int:
        seen = set()
        depth = 0
        current = label
        while current in self.label_hierarchy and current not in seen:
            seen.add(current)
            current = self.label_hierarchy[current]
            depth += 1
        return depth

    def _select_leaf_label(self, labels: list[str]) -> str:
        if not labels:
            return ""
        best_label = labels[0]
        best_depth = -1
        for label in labels:
            norm = self._normalize_label(label)
            canon = self._canonical_label(norm)
            depth = self._label_depth(canon) if canon in self.label_hierarchy or canon in self.label_aliases.values() else -1
            if depth > best_depth:
                best_depth = depth
                best_label = label
            elif depth == best_depth and len(label) > len(best_label):
                best_label = label
        return best_label

    def _identify_entity(self, question: str) -> Optional[Dict]:
        """
        Step 1: Use LLM to identify entities in the question.

        Args:
            question: User's question

        Returns:
            Dict with entity information or None
        """
        try:
            prompt = ENTITY_IDENTIFICATION_PROMPT.format(question=question)
            response = self.llm_client.invoke(prompt)

            # Extract content from response
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)

            # Parse JSON response
            response_text = response_text.strip()
            # Remove markdown code blocks if present
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                response_text = "\n".join(lines[1:-1]) if len(lines) > 2 else response_text

            entity_info = json.loads(response_text)
            logging.info(f"Entity identification result: {entity_info}")
            return entity_info

        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse entity identification response: {e}")
            return None
        except Exception as e:
            logging.error(f"Entity identification failed: {e}")
            return None

    def _locate_node(self, entity: Dict) -> Optional[Dict]:
        """
        Step 2a: Locate the specific node in the knowledge graph.

        Strategy:
        1. If entity has sequence, try exact sequence match
        2. If no match, try vector search (for sequences)
        3. If entity has identifier/name, try property match
        4. Return None if all fail

        Args:
            entity: Entity dict with type, sequence, identifier, name

        Returns:
            Dict with node id/type or None
        """
        entity_type = entity.get("type", "Protein")
        sequence = entity.get("sequence")
        identifier = entity.get("identifier")
        name = entity.get("name")

        # Strategy 1: Exact sequence match
        if sequence:
            node_info = self._find_node_by_sequence(sequence)
            if node_info:
                return node_info

        # Strategy 2: Property match (name or identifier)
        if identifier or name:
            node_id = self._find_node_by_property(identifier or name, entity_type)
            if node_id:
                return {"id": node_id, "type": entity_type}

        logging.warning(f"Could not locate node for entity: {entity}")
        return None

    def _find_node_by_sequence(self, sequence: str) -> Optional[Dict]:
        """
        Find node by exact sequence match across all labels.

        Args:
            sequence: Protein sequence

        Returns:
            Dict with node id/type or None
        """
        try:
            # Escape single quotes in sequence
            escaped_sequence = sequence.replace("'", "\\'")
            query = f"""
            MATCH (n)
            WHERE n.sequence = '{escaped_sequence}'
            WITH n, labels(n)[0] AS node_type
            RETURN n.id AS node_id, node_type
            ORDER BY CASE WHEN node_type = 'Protein' THEN 0 ELSE 1 END
            LIMIT 1
            """
            result = self.neo4j_helper.execute(query, top_k=1)

            if result and result != "Given cypher query did not return any result":
                node_id = result[0].get("node_id")
                node_type = result[0].get("node_type")
                if node_id:
                    logging.info(f"Found node by sequence: {node_id} ({node_type})")
                    return {"id": node_id, "type": node_type}

        except Exception as e:
            logging.error(f"Sequence match failed: {e}")

        return None

    def _find_node_by_property(self, identifier: str, entity_type: str = "Protein") -> Optional[str]:
        """
        Find node by name or ID property.

        Args:
            identifier: Entity name or ID
            entity_type: Node type

        Returns:
            Node ID or None
        """
        try:
            query = f"""
            MATCH (n:{entity_type})
            WHERE n.id = $identifier OR n.name CONTAINS $identifier OR n.primary_protein_name CONTAINS $identifier
            RETURN n.id AS node_id
            LIMIT 1
            """
            result = self.neo4j_helper.execute(query, top_k=1)

            if result and result != "Given cypher query did not return any result":
                node_id = result[0].get("node_id")
                logging.info(f"Found node by property: {node_id}")
                return node_id

        except Exception as e:
            logging.error(f"Property match failed: {e}")

        return None

    def _extract_node_schema(self, node_id: str, node_type: str) -> Optional[Dict]:
        """
        Step 2b: Extract schema (properties and edges) for a specific node.

        Args:
            node_id: Node ID
            node_type: Node type

        Returns:
            Schema dict with node_id, node_type, properties, edges
        """
        try:
            # Escape single quotes in node_id
            escaped_node_id = node_id.replace("'", "\\'")
            non_empty_check = "v IS NOT NULL AND toString(v) <> '' AND toString(v) <> '[]' AND toString(v) <> '{}' "
            node_query = f"""
            MATCH (n:{node_type} {{id: '{escaped_node_id}'}})
            WITH n, keys(n) AS props
            UNWIND props AS p
            WITH n, p, n[p] AS v
            WHERE {non_empty_check}
            RETURN collect(DISTINCT p) AS non_null_props,
                   collect(DISTINCT CASE
                     WHEN toLower(p) CONTAINS 'name'
                       OR toLower(p) CONTAINS 'label'
                       OR toLower(p) CONTAINS 'symbol'
                     THEN {{key: p, value: v}}
                     ELSE NULL
                   END) AS name_like
            """
            node_result = self.neo4j_helper.execute(node_query, top_k=200)

            if not node_result or node_result == "Given cypher query did not return any result":
                return None

            non_null_props = node_result[0].get("non_null_props") or []
            name_like = [x for x in (node_result[0].get("name_like") or []) if x]

            edge_query = f"""
            MATCH (n:{node_type} {{id: '{escaped_node_id}'}})-[r]-()
            WITH type(r) AS rel_type, keys(r) AS props, r
            UNWIND props AS p
            WITH rel_type, p, r[p] AS v
            WHERE {non_empty_check}
            RETURN rel_type, collect(DISTINCT p) AS non_null_props
            """
            edge_result = self.neo4j_helper.execute(edge_query, top_k=500)
            edges = []
            if edge_result and edge_result != "Given cypher query did not return any result":
                for row in edge_result:
                    rel_type = row.get("rel_type")
                    props = row.get("non_null_props") or []
                    if rel_type:
                        edges.append({
                            "type": rel_type,
                            "properties": sorted(set(props)),
                        })

            neighbor_query = f"""
            MATCH (n:{node_type} {{id: '{escaped_node_id}'}})--(m)
            WITH labels(m) AS labels, keys(m) AS props, m
            UNWIND props AS prop
            WITH labels, prop, m[prop] AS v
            WHERE {non_empty_check}
            RETURN labels,
                   collect(DISTINCT prop) AS properties,
                   collect(DISTINCT CASE
                     WHEN toLower(prop) CONTAINS 'name'
                       OR toLower(prop) CONTAINS 'label'
                       OR toLower(prop) CONTAINS 'symbol'
                     THEN {{key: prop, value: v}}
                     ELSE NULL
                   END) AS name_like
            """
            neighbor_result = self.neo4j_helper.execute(neighbor_query, top_k=500)
            neighbor_props = {}
            neighbor_core = {}
            if neighbor_result and neighbor_result != "Given cypher query did not return any result":
                for row in neighbor_result:
                    labels = row.get("labels") or []
                    props = row.get("properties") or []
                    name_like = [x for x in (row.get("name_like") or []) if x]
                    leaf_label = self._select_leaf_label(labels)
                    if leaf_label:
                        if props:
                            neighbor_props[leaf_label] = sorted(set(props))
                        if name_like:
                            neighbor_core.setdefault(leaf_label, [])
                            neighbor_core[leaf_label].extend(name_like)

            schema = {
                "node_id": node_id,
                "node_type": node_type,
                "properties": sorted(set(non_null_props)),
                "edges": edges,
                "neighbor_node_properties": neighbor_props,
                "core_values": {
                    "node": {
                        "id": node_id,
                        "name_like": name_like,
                    },
                    "neighbors": neighbor_core,
                    "edges": {},
                },
            }

            logging.info(
                f"Extracted schema for node {node_id}: {len(schema.get('edges', []))} edges, "
                f"{len(neighbor_props)} neighbor types"
            )
            return schema

        except Exception as e:
            logging.error(f"Schema extraction failed for node {node_id}: {e}")

        return None

    def _merge_schemas(self, node_schemas: List[Dict]) -> Dict:
        """
        Step 3: Merge multiple node schemas.

        Args:
            node_schemas: List of schema dicts

        Returns:
            Merged schema dict
        """
        if len(node_schemas) == 1:
            return self._convert_to_full_schema_format(node_schemas[0])

        merged = {
            "nodes": set(),
            "node_properties": [],
            "edges": [],
            "edge_properties": [],
            "anchor_entities": [],
            "target_node_context": "",
        }

        seen_node_props = set()
        seen_edges = set()
        seen_edge_props = set()

        # Merge up to max_entities schemas
        for schema in node_schemas[:self.max_entities]:
            # Merge node types
            node_type = schema.get("node_type")
            if node_type:
                merged["nodes"].add(node_type)

            # Merge node properties
            properties = schema.get("properties", [])
            if properties and node_type:
                prop_key = (node_type, tuple(sorted(properties)))
                if prop_key not in seen_node_props:
                    merged["node_properties"].append({
                        "labels": node_type,
                        "properties": [{"property": p, "type": "STRING"} for p in properties]
                    })
                    seen_node_props.add(prop_key)

            # Merge neighbor node properties
            neighbor_props = schema.get("neighbor_node_properties", {})
            for neighbor_label, props in neighbor_props.items():
                if neighbor_label:
                    merged["nodes"].add(neighbor_label)
                if props and neighbor_label:
                    prop_key = (neighbor_label, tuple(sorted(props)))
                    if prop_key not in seen_node_props:
                        merged["node_properties"].append({
                            "labels": neighbor_label,
                            "properties": [{"property": p, "type": "STRING"} for p in props]
                        })
                        seen_node_props.add(prop_key)

            # Merge edges (limit to max_edge_types)
            for edge in schema.get("edges", []):
                if len(merged["edges"]) >= self.max_edge_types:
                    break

                edge_type = edge.get("type")
                source = edge.get("source")
                target = edge.get("target")

                if not edge_type:
                    continue

                # Create edge string representation
                if source:
                    edge_str = f"(:{source})-[:{edge_type}]->(:{node_type})"
                elif target:
                    edge_str = f"(:{node_type})-[:{edge_type}]->(:{target})"
                else:
                    continue

                if edge_str not in seen_edges:
                    merged["edges"].append(edge_str)
                    seen_edges.add(edge_str)

                    # Add edge properties
                    edge_props = edge.get("properties", [])
                    if edge_props and edge_type not in seen_edge_props:
                        merged["edge_properties"].append({
                            "type": edge_type,
                            "properties": [{"property": p, "type": "STRING"} for p in edge_props]
                        })
                        seen_edge_props.add(edge_type)

            for anchor in schema.get("anchor_entities", []) or []:
                if anchor not in merged["anchor_entities"]:
                    merged["anchor_entities"].append(anchor)

            context = schema.get("target_node_context")
            if context:
                if merged["target_node_context"]:
                    merged["target_node_context"] += "\n\n" + context
                else:
                    merged["target_node_context"] = context

        # Convert sets to lists
        merged["nodes"] = [{"labels": list(merged["nodes"])}]

        logging.info(f"Merged {len(node_schemas)} schemas: {len(merged['edges'])} edges, {len(merged['node_properties'])} node types")
        return merged

    def _convert_to_full_schema_format(self, node_schema: Dict) -> Dict:
        """
        Convert single node schema to full schema format.

        Args:
            node_schema: Single node schema dict

        Returns:
            Schema in full format
        """
        node_type = node_schema.get("node_type")
        properties = node_schema.get("properties", [])
        edges_data = node_schema.get("edges", [])

        neighbor_props = node_schema.get("neighbor_node_properties", {})
        node_labels = {node_type}
        for label in neighbor_props.keys():
            if label:
                node_labels.add(label)

        schema = {
            "nodes": [{"labels": sorted(node_labels)}],
            "node_properties": [{
                "labels": node_type,
                "properties": [{"property": p, "type": "STRING"} for p in properties]
            }],
            "edges": [],
            "edge_properties": [],
            "anchor_entities": node_schema.get("anchor_entities", []),
            "target_node_context": node_schema.get("target_node_context", ""),
        }

        for neighbor_label, props in neighbor_props.items():
            if neighbor_label and props:
                schema["node_properties"].append({
                    "labels": neighbor_label,
                    "properties": [{"property": p, "type": "STRING"} for p in props]
                })

        seen_edge_types = set()

        for edge in edges_data[:self.max_edge_types]:
            edge_type = edge.get("type")
            source = edge.get("source")
            target = edge.get("target")

            if not edge_type:
                continue

            # Create edge string
            if source:
                edge_str = f"(:{source})-[:{edge_type}]->(:{node_type})"
            elif target:
                edge_str = f"(:{node_type})-[:{edge_type}]->(:{target})"
            else:
                continue

            schema["edges"].append(edge_str)

            # Add edge properties
            if edge_type not in seen_edge_types:
                edge_props = edge.get("properties", [])
                if edge_props:
                    schema["edge_properties"].append({
                        "type": edge_type,
                        "properties": [{"property": p, "type": "STRING"} for p in edge_props]
                    })
                seen_edge_types.add(edge_type)

        return schema

    def _build_target_context(self, schema: Dict, node_id: str, node_type: str) -> str:
        lines = []
        lines.append(f"Target node: {node_type} id={node_id}")

        core_values = schema.get("core_values", {})
        node_values = core_values.get("node", {}) or {}
        name_like = node_values.get("name_like") or []
        if name_like:
            lines.append("Target node name-like values:")
            for item in name_like:
                key = item.get("key")
                value = item.get("value")
                if key and value is not None:
                    lines.append(f"- {key}: {value}")

        props = schema.get("properties", []) or []
        if props:
            lines.append("Node properties:")
            lines.append(", ".join(sorted(props)))

        edges = schema.get("edges", []) or []
        if edges:
            lines.append("Edges:")
            for edge in edges:
                edge_type = edge.get("type")
                edge_props = edge.get("properties") or []
                edge_str = f"[:{edge_type}]"
                if edge_props:
                    lines.append(f"- {edge_str} props={sorted(edge_props)}")
                    desc_props = [p for p in edge_props if self._is_desc_property(p)]
                    if desc_props:
                        desc_values = self._edge_desc_samples(node_type, node_id, edge_type, desc_props)
                        if desc_values:
                            core_values.setdefault("edges", {})
                            core_values["edges"][edge_type] = desc_values
                            lines.append(f"  desc_values={desc_values}")
                else:
                    lines.append(f"- {edge_str}")

        neighbor_props = schema.get("neighbor_node_properties", {}) or {}
        if neighbor_props:
            lines.append("Neighbor node properties:")
            for label, props in sorted(neighbor_props.items()):
                if props:
                    lines.append(f"- {label}: {sorted(props)}")

        neighbor_core = core_values.get("neighbors", {}) or {}
        if neighbor_core:
            lines.append("Neighbor node name-like values:")
            for label, items in sorted(neighbor_core.items()):
                values = []
                for item in items:
                    key = item.get("key")
                    value = item.get("value")
                    if key and value is not None:
                        values.append(f"{key}: {value}")
                if values:
                    lines.append(f"- {label}: {values}")

        return "\n".join(lines).strip()

    @staticmethod
    def _is_desc_property(prop_name: str) -> bool:
        name = prop_name.lower()
        return "desc" in name or "comment" in name or "annotation" in name

    def _edge_desc_samples(self, node_type: str, node_id: str, edge_type: str, desc_props: list[str]) -> dict:
        escaped_node_id = node_id.replace("'", "\\'")
        samples = {}
        for prop in desc_props:
            query = f"""
            MATCH (n:{node_type} {{id: '{escaped_node_id}'}})-[r:{edge_type}]-()
            WHERE r.{prop} IS NOT NULL AND toString(r.{prop}) <> '' AND toString(r.{prop}) <> '[]' AND toString(r.{prop}) <> '{{}}'
            RETURN r.{prop} AS v
            LIMIT 3
            """
            result = self.neo4j_helper.execute(query, top_k=3)
            if result and result != "Given cypher query did not return any result":
                values = [row.get("v") for row in result if isinstance(row, dict)]
                if values:
                    samples[prop] = values
        return samples

    def _cache_schema(self, node_id: str, schema: Dict) -> None:
        """
        Cache node schema to file.

        Args:
            node_id: Node ID
            schema: Schema dict
        """
        try:
            # Sanitize node_id for filename (replace colons with underscores)
            safe_id = node_id.replace(":", "_").replace("/", "_")
            cache_file = self.entity_cache_dir / f"{safe_id}.json"

            cache_data = {
                "schema": schema,
                "timestamp": datetime.now().isoformat(),
                "ttl_hours": self.cache_ttl_hours
            }

            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)

            logging.info(f"Cached schema for node {node_id}")

        except Exception as e:
            logging.error(f"Failed to cache schema for {node_id}: {e}")

    def _load_cached_schema(self, node_id: str) -> Optional[Dict]:
        """
        Load cached schema if valid (within TTL).

        Args:
            node_id: Node ID

        Returns:
            Cached schema or None
        """
        try:
            safe_id = node_id.replace(":", "_").replace("/", "_")
            cache_file = self.entity_cache_dir / f"{safe_id}.json"

            if not cache_file.exists():
                return None

            with open(cache_file, 'r') as f:
                cached = json.load(f)

            # Check TTL
            cached_time = datetime.fromisoformat(cached["timestamp"])
            age_hours = (datetime.now() - cached_time).total_seconds() / 3600

            if age_hours > self.cache_ttl_hours:
                logging.info(f"Cache expired for {node_id}, deleting")
                cache_file.unlink()
                return None

            return cached["schema"]

        except Exception as e:
            logging.error(f"Failed to load cached schema for {node_id}: {e}")
            return None

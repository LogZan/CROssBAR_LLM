import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class PropertyContextBuilder:
    def __init__(self, neo4j_helper, config: Optional[Dict[str, Any]] = None):
        self.neo4j_helper = neo4j_helper
        self.config = config or {}

        self.enabled = self.config.get("enabled", True)
        self.cache_dir = Path(self.config.get("cache_dir", "cache"))
        self.cache_ttl_hours = float(self.config.get("cache_ttl_hours", 24))
        self.max_node_labels = int(self.config.get("max_node_labels", 50))
        self.max_node_properties = int(self.config.get("max_node_properties", 50))
        self.max_edge_types = int(self.config.get("max_edge_types", 200))
        self.max_edge_properties = int(self.config.get("max_edge_properties", 50))
        self.sample_per_property = int(self.config.get("sample_per_property", 10))
        self.max_context_chars = int(self.config.get("max_context_chars", 12000))

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.cache_dir / "property_context.json"

    def build_context(self, schema: Dict[str, Any]) -> str:
        if not self.enabled:
            return ""

        cached = self._load_cached_context()
        if cached:
            return cached

        context = self._build_context_from_schema(schema)
        if context:
            self._cache_context(context)
        return context

    def _load_cached_context(self) -> Optional[str]:
        if not self.cache_path.exists():
            return None

        try:
            with open(self.cache_path, "r") as f:
                payload = json.load(f)
            ts = float(payload.get("timestamp", 0))
            if ts and (time.time() - ts) <= self.cache_ttl_hours * 3600:
                return payload.get("context", "")
        except Exception as e:
            logging.warning(f"Failed to load property context cache: {e}")

        return None

    def _cache_context(self, context: str) -> None:
        try:
            with open(self.cache_path, "w") as f:
                json.dump({"timestamp": time.time(), "context": context}, f)
        except Exception as e:
            logging.warning(f"Failed to cache property context: {e}")

    def _build_context_from_schema(self, schema: Dict[str, Any]) -> str:
        try:
            node_props = schema.get("node_properties", [])
            edge_props = schema.get("edge_properties", [])
        except Exception:
            return ""

        node_props = node_props[: self.max_node_labels]
        edge_props = edge_props[: self.max_edge_types]

        lines: List[str] = []
        lines.append("Property examples (non-exhaustive):")

        for node in node_props:
            label = node.get("labels")
            if not label:
                continue
            props = node.get("properties", [])[: self.max_node_properties]
            if not props:
                continue

            lines.append(f"Node: {label}")
            for prop in props:
                prop_name = prop.get("property") if isinstance(prop, dict) else prop
                if not prop_name:
                    continue
                samples = self._sample_node_property(label, prop_name)
                if samples:
                    lines.append(f"- {prop_name}: {samples}")

        for rel in edge_props:
            rel_type = rel.get("type")
            if not rel_type:
                continue
            props = rel.get("properties", [])[: self.max_edge_properties]
            if not props:
                continue

            lines.append(f"Relationship: {rel_type}")
            for prop in props:
                prop_name = prop.get("property") if isinstance(prop, dict) else prop
                if not prop_name:
                    continue
                samples = self._sample_edge_property(rel_type, prop_name)
                if samples:
                    lines.append(f"- {prop_name}: {samples}")

        context = "\n".join(lines).strip()
        if self.max_context_chars and len(context) > self.max_context_chars:
            context = context[: self.max_context_chars].rstrip() + "..."

        return context

    def _sample_node_property(self, label: str, prop_name: str) -> str:
        query = (
            f"MATCH (n:{label}) "
            f"WHERE n.{prop_name} IS NOT NULL "
            f"RETURN n.{prop_name} AS v "
            f"LIMIT {self.sample_per_property}"
        )
        return self._format_samples(self._run_query(query))

    def _sample_edge_property(self, rel_type: str, prop_name: str) -> str:
        query = (
            f"MATCH ()-[r:{rel_type}]-() "
            f"WHERE r.{prop_name} IS NOT NULL "
            f"RETURN r.{prop_name} AS v "
            f"LIMIT {self.sample_per_property}"
        )
        return self._format_samples(self._run_query(query))

    def _run_query(self, query: str) -> List[Any]:
        try:
            result = self.neo4j_helper.execute(query, top_k=self.sample_per_property)
            if not result or result == "Given cypher query did not return any result":
                return []
            values = []
            for row in result:
                if isinstance(row, dict) and "v" in row:
                    values.append(row["v"])
            return values
        except Exception as e:
            logging.debug(f"Property context sample query failed: {e}")
            return []

    def _format_samples(self, values: List[Any]) -> str:
        if not values:
            return ""

        formatted = []
        for v in values:
            if isinstance(v, list):
                formatted.append(v[:5])
            elif isinstance(v, str):
                formatted.append(v)
            else:
                formatted.append(v)

        try:
            return json.dumps(formatted, ensure_ascii=False)
        except Exception:
            return str(formatted)

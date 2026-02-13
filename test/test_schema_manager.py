"""
Unit tests for SchemaManager.

Tests schema loading, validation, and prompt generation without requiring
actual Neo4j connection or graph_schema.json file.
"""

import importlib.util
import json
import os
import tempfile
import unittest

# Add backend to path
import sys

_backend_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "crossbar_llm",
    "backend",
)
sys.path.insert(0, _backend_dir)

# Import schema_manager directly by file path to bypass tools/__init__.py
_schema_path = os.path.join(_backend_dir, "tools", "schema_manager.py")
_spec = importlib.util.spec_from_file_location("schema_manager", _schema_path)
_schema_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_schema_module)

SchemaManager = _schema_module.SchemaManager


class TestSchemaManager(unittest.TestCase):
    """Test SchemaManager functionality."""

    def setUp(self):
        """Set up test schema."""
        self.test_schema = {
            "nodes": [
                {"labels": ["Protein", "Gene", "Disease", "Drug"]}
            ],
            "node_properties": [
                {
                    "labels": "Protein",
                    "properties": [
                        {"property": "primaryAccession"},
                        {"property": "geneName"},
                        {"property": "uniProtkbId"},
                        {"property": "sequence"},
                    ]
                },
                {
                    "labels": "Gene",
                    "properties": [
                        {"property": "geneName"},
                        {"property": "ensemblId"},
                    ]
                },
                {
                    "labels": "Disease",
                    "properties": [
                        {"property": "diseaseId"},
                        {"property": "diseaseName"},
                    ]
                },
            ],
            "edges": [
                "(:Gene)-[:Gene_encodes_protein]->(:Protein)",
                "(:Protein)-[:Protein_has_catalytic_activity]->(:CatalyticActivityAnnotation)",
                "(:Gene)-[:Gene_associated_with_disease]->(:Disease)",
            ],
            "edge_properties": [
                {
                    "type": "Gene_encodes_protein",
                    "properties": []
                }
            ],
        }

        # Create temporary schema file
        self.temp_dir = tempfile.mkdtemp()
        self.schema_path = os.path.join(self.temp_dir, "test_schema.json")
        with open(self.schema_path, "w") as f:
            json.dump(self.test_schema, f)

    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.schema_path):
            os.remove(self.schema_path)
        os.rmdir(self.temp_dir)

    def test_schema_loading(self):
        """Test that schema loads correctly."""
        manager = SchemaManager(self.schema_path)
        self.assertIsNotNone(manager.schema)
        self.assertIn("nodes", manager.schema)
        self.assertIn("node_properties", manager.schema)

    def test_get_node_properties(self):
        """Test retrieving node properties."""
        manager = SchemaManager(self.schema_path)
        
        # Test Protein properties
        protein_props = manager.get_node_properties("Protein")
        self.assertIn("primaryAccession", protein_props)
        self.assertIn("geneName", protein_props)
        self.assertIn("uniProtkbId", protein_props)
        
        # Test Gene properties
        gene_props = manager.get_node_properties("Gene")
        self.assertIn("geneName", gene_props)
        self.assertIn("ensemblId", gene_props)
        
        # Test non-existent node
        unknown_props = manager.get_node_properties("UnknownNode")
        self.assertEqual(unknown_props, [])

    def test_get_primary_key(self):
        """Test primary key identification."""
        manager = SchemaManager(self.schema_path)
        
        # Known primary keys
        self.assertEqual(manager.get_primary_key("Protein"), "primaryAccession")
        self.assertEqual(manager.get_primary_key("Gene"), "geneName")
        self.assertEqual(manager.get_primary_key("Disease"), "diseaseId")

    def test_get_searchable_properties(self):
        """Test searchable properties extraction."""
        manager = SchemaManager(self.schema_path)
        
        # Protein searchable properties
        protein_searchable = manager.get_searchable_properties("Protein")
        self.assertIn("primaryAccession", protein_searchable)
        self.assertIn("geneName", protein_searchable)

    def test_get_relationships(self):
        """Test relationship extraction."""
        manager = SchemaManager(self.schema_path)
        
        # All relationships
        all_rels = manager.get_relationships()
        self.assertGreater(len(all_rels), 0)
        
        # Filter by source
        gene_rels = manager.get_relationships(source_node="Gene")
        self.assertGreater(len(gene_rels), 0)
        for rel in gene_rels:
            self.assertEqual(rel["source"], "Gene")
        
        # Filter by target
        protein_rels = manager.get_relationships(target_node="Protein")
        self.assertGreater(len(protein_rels), 0)
        for rel in protein_rels:
            self.assertEqual(rel["target"], "Protein")

    def test_validate_cypher_valid(self):
        """Test validation of valid Cypher queries."""
        manager = SchemaManager(self.schema_path)
        
        # Valid query with correct properties
        valid_query = "MATCH (p:Protein {primaryAccession: 'P00533'}) RETURN p"
        is_valid, error, suggestion = manager.validate_cypher(valid_query)
        self.assertTrue(is_valid)
        self.assertIsNone(error)
        
        # Valid query with Gene
        valid_gene_query = "MATCH (g:Gene {geneName: 'EGFR'}) RETURN g"
        is_valid, error, suggestion = manager.validate_cypher(valid_gene_query)
        self.assertTrue(is_valid)

    def test_validate_cypher_invalid_property(self):
        """Test validation catches invalid property names."""
        manager = SchemaManager(self.schema_path)
        
        # Query with 'id' property (common mistake)
        invalid_query = "MATCH (p:Protein {id: 'INSR'}) RETURN p"
        is_valid, error, suggestion = manager.validate_cypher(invalid_query)
        self.assertFalse(is_valid)
        self.assertIsNotNone(error)
        self.assertIn("id", error.lower())

    def test_validate_cypher_invalid_node_label(self):
        """Test validation catches invalid node labels."""
        manager = SchemaManager(self.schema_path)
        
        # Query with non-existent node type
        invalid_query = "MATCH (x:UnknownNode {name: 'test'}) RETURN x"
        is_valid, error, suggestion = manager.validate_cypher(invalid_query)
        self.assertFalse(is_valid)
        self.assertIn("Unknown node type", error)

    def test_validate_cypher_invalid_relationship(self):
        """Test validation catches invalid relationship types."""
        manager = SchemaManager(self.schema_path)
        
        # Query with non-existent relationship
        invalid_query = "MATCH (p:Protein)-[:INVALID_REL]->(x) RETURN p"
        is_valid, error, suggestion = manager.validate_cypher(invalid_query)
        self.assertFalse(is_valid)
        self.assertIn("Unknown relationship type", error)

    def test_validate_cypher_too_many_optionals(self):
        """Test validation warns about excessive OPTIONAL MATCH."""
        manager = SchemaManager(self.schema_path)
        
        # Query with too many OPTIONAL MATCH clauses
        excessive_query = "MATCH (p:Protein {primaryAccession: 'P00533'}) " + \
                         " ".join(["OPTIONAL MATCH (p)-[:REL{}]->(n{})".format(i, i) 
                                  for i in range(15)])
        is_valid, error, suggestion = manager.validate_cypher(excessive_query)
        self.assertFalse(is_valid)
        self.assertIn("too complex", error.lower())

    def test_generate_schema_prompt(self):
        """Test schema prompt generation."""
        manager = SchemaManager(self.schema_path)
        
        # Generate full prompt
        prompt = manager.generate_schema_prompt()
        self.assertIn("CROssBARv2", prompt)
        self.assertIn("Protein", prompt)
        self.assertIn("Primary Key", prompt)
        self.assertIn("Searchable Properties", prompt)
        
        # Generate filtered prompt
        filtered_prompt = manager.generate_schema_prompt(relevant_nodes=["Protein"])
        self.assertIn("Protein", filtered_prompt)
        
    def test_annotation_guidance_in_prompt(self):
        """Test that annotation data storage guidance is included in schema prompt."""
        manager = SchemaManager(self.schema_path)
        
        # Generate prompt
        prompt = manager.generate_schema_prompt()
        
        # Check for annotation section
        self.assertIn("IMPORTANT: Annotation Data Storage Location", prompt)
        self.assertIn("relationship properties", prompt.lower())
        
        # Check for specific annotation relationships
        self.assertIn("Protein_has_catalytic_activity", prompt)
        self.assertIn("ecNumber", prompt)
        self.assertIn("Protein_has_cofactor", prompt)
        self.assertIn("Protein_has_binding_site_feature", prompt)
        self.assertIn("Protein_has_signal_feature", prompt)
        self.assertIn("Protein_has_subunit", prompt)
        
        # Check for correct pattern example
        self.assertIn("-[rel:", prompt)
        self.assertIn("rel.ecNumber", prompt)
        
        # Check for wrong pattern example
        self.assertIn("WRONG", prompt)
        self.assertIn("ca.ecNumber", prompt)
        self.assertIn("Returns null", prompt)

    def test_missing_schema_file(self):
        """Test behavior when schema file doesn't exist."""
        manager = SchemaManager(schema_path="/nonexistent/path/schema.json")
        
        # Should still initialize but with empty schema
        self.assertIsNotNone(manager.schema)
        
        # Validation should pass (no schema to validate against)
        is_valid, error, suggestion = manager.validate_cypher(
            "MATCH (x:Any {anything: 'value'}) RETURN x"
        )
        self.assertTrue(is_valid)


class TestSchemaManagerEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_empty_schema(self):
        """Test handling of empty schema."""
        temp_dir = tempfile.mkdtemp()
        schema_path = os.path.join(temp_dir, "empty_schema.json")
        
        with open(schema_path, "w") as f:
            json.dump({"nodes": [], "node_properties": [], "edges": [], "edge_properties": []}, f)
        
        manager = SchemaManager(schema_path)
        
        # Should handle gracefully
        props = manager.get_node_properties("AnyNode")
        self.assertEqual(props, [])
        
        # Cleanup
        os.remove(schema_path)
        os.rmdir(temp_dir)

    def test_malformed_schema(self):
        """Test handling of malformed schema data."""
        temp_dir = tempfile.mkdtemp()
        schema_path = os.path.join(temp_dir, "malformed_schema.json")
        
        # Schema with unexpected structure
        with open(schema_path, "w") as f:
            json.dump({"unexpected": "structure"}, f)
        
        # Should handle gracefully without crashing
        manager = SchemaManager(schema_path)
        self.assertIsNotNone(manager.schema)
        
        # Cleanup
        os.remove(schema_path)
        os.rmdir(temp_dir)


class TestFormatSchemaForLLM(unittest.TestCase):
    """Test the format_schema_for_llm method."""

    def setUp(self):
        """Set up test schema with abstract and concrete labels."""
        self.test_schema = {
            "nodes": [
                {"labels": [
                    "Protein", "Gene", "Disease", "Drug",
                    "NamedThing", "Entity", "BiologicalEntity",
                    "CatalyticActivityAnnotation",
                ]}
            ],
            "node_properties": [
                {
                    "labels": "Protein",
                    "properties": [
                        {"property": "primaryAccession"},
                        {"property": "geneName"},
                    ]
                },
                {
                    "labels": "Gene",
                    "properties": [
                        {"property": "geneName"},
                        {"property": "ensemblId"},
                    ]
                },
                {
                    "labels": "Drug",
                    "properties": [
                        {"property": "name"},
                        {"property": "id"},
                    ]
                },
                {
                    "labels": "CatalyticActivityAnnotation",
                    "properties": [
                        {"property": "id"},
                    ]
                },
            ],
            "edges": [
                "(:Gene)-[:Gene_encodes_protein]->(:Protein)",
                "(:NamedThing)-[:Gene_encodes_protein]->(:Protein)",
                "(:Drug)-[:Drug_targets_protein]->(:Protein)",
                "(:NamedThing)-[:Drug_targets_protein]->(:NamedThing)",
                "(:NamedThing)-[:Drug_targets_protein]->(:Protein)",
                "(:Entity)-[:Drug_targets_protein]->(:Protein)",
                "(:Protein)-[:Protein_has_catalytic_activity]->(:CatalyticActivityAnnotation)",
                "(:NamedThing)-[:Protein_has_catalytic_activity]->(:CatalyticActivityAnnotation)",
                "(:BiologicalEntity)-[:Protein_has_catalytic_activity]->(:CatalyticActivityAnnotation)",
                # Edge where all sources and targets are abstract
                "(:NamedThing)-[:Protein_has_allergen]->(:NamedThing)",
                "(:BiologicalEntity)-[:Protein_has_allergen]->(:Entity)",
            ],
            "edge_properties": [
                {
                    "type": "Drug_targets_protein",
                    "properties": [{"property": "source"}, {"property": "score"}]
                },
                {
                    "type": "Protein_has_catalytic_activity",
                    "properties": [{"property": "ecNumber"}, {"property": "name"}]
                },
            ],
        }

        self.temp_dir = tempfile.mkdtemp()
        self.schema_path = os.path.join(self.temp_dir, "test_schema.json")
        with open(self.schema_path, "w") as f:
            json.dump(self.test_schema, f)

    def tearDown(self):
        if os.path.exists(self.schema_path):
            os.remove(self.schema_path)
        os.rmdir(self.temp_dir)

    def test_format_schema_separates_nodes_and_relationships(self):
        """Schema output should have separate Node types and Relationship types sections."""
        manager = SchemaManager(self.schema_path)
        result = manager.format_schema_for_llm()
        self.assertIn("## Node types", result)
        self.assertIn("## Relationship types", result)

    def test_format_schema_excludes_abstract_node_labels(self):
        """Abstract labels like NamedThing, Entity should not appear as node types."""
        manager = SchemaManager(self.schema_path)
        result = manager.format_schema_for_llm()
        # Should include concrete nodes
        self.assertIn("Protein:", result)
        self.assertIn("Gene:", result)
        self.assertIn("Drug:", result)
        # NamedThing should NOT be listed as a node type
        # (it may appear in the text instructions, so check carefully)
        lines = result.split("\n")
        node_section_lines = []
        in_node_section = False
        for line in lines:
            if "## Node types" in line:
                in_node_section = True
                continue
            if "## Relationship types" in line:
                in_node_section = False
            if in_node_section and line.strip().startswith("NamedThing"):
                self.fail("NamedThing should not appear as a node type entry")

    def test_format_schema_deduplicates_edges(self):
        """Edges with abstract source/target should be filtered, only concrete kept."""
        manager = SchemaManager(self.schema_path)
        result = manager.format_schema_for_llm()
        # Drug_targets_protein should show (Drug)->(Protein), NOT (NamedThing)->(Protein)
        self.assertIn("Drug_targets_protein", result)
        # Find the Drug_targets_protein line and verify it only shows concrete types
        rel_section = result.split("## Relationship types")[1].split("## Critical rules")[0]
        dtp_lines = [l.strip() for l in rel_section.split("\n")
                     if l.strip().startswith("Drug_targets_protein")]
        self.assertTrue(len(dtp_lines) > 0, "Drug_targets_protein relationship not found")
        self.assertIn("(Drug)->(Protein)", dtp_lines[0])
        self.assertNotIn("NamedThing", dtp_lines[0])

    def test_format_schema_includes_relationship_properties(self):
        """Relationship properties should be listed alongside the relationship type."""
        manager = SchemaManager(self.schema_path)
        result = manager.format_schema_for_llm()
        self.assertIn("ecNumber", result)
        self.assertIn("Drug_targets_protein", result)
        self.assertIn("score", result)

    def test_format_schema_includes_critical_rules(self):
        """Critical rules section should be present."""
        manager = SchemaManager(self.schema_path)
        result = manager.format_schema_for_llm()
        self.assertIn("Critical rules", result)
        self.assertIn("NEVER use a relationship type", result)
        self.assertIn("square brackets", result)
        self.assertIn("parentheses", result)

    def test_format_schema_node_properties_listed(self):
        """Node properties should be listed with each node type."""
        manager = SchemaManager(self.schema_path)
        result = manager.format_schema_for_llm()
        self.assertIn("primaryAccession", result)
        self.assertIn("geneName", result)

    def test_format_schema_infers_concrete_label_for_all_abstract_edges(self):
        """Relationship types with only abstract edges should infer the source from the name."""
        manager = SchemaManager(self.schema_path)
        result = manager.format_schema_for_llm()
        # Protein_has_allergen has only abstract edges but the source should be
        # inferred as Protein from the relationship name prefix.
        self.assertIn("Protein_has_allergen", result)
        rel_section = result.split("## Relationship types")[1].split("## Critical rules")[0]
        allergen_line = [l for l in rel_section.split("\n") if "Protein_has_allergen" in l]
        self.assertTrue(len(allergen_line) > 0)
        # Source should be "Protein" (inferred from "Protein_has_allergen")
        self.assertIn("(Protein)->", allergen_line[0])

    def test_format_schema_empty_schema(self):
        """format_schema_for_llm should return empty string for empty schema."""
        temp_dir = tempfile.mkdtemp()
        schema_path = os.path.join(temp_dir, "empty.json")
        with open(schema_path, "w") as f:
            json.dump({"nodes": [], "node_properties": [], "edges": [], "edge_properties": []}, f)
        manager = SchemaManager(schema_path)
        # Empty schema should still return something (header lines at minimum)
        result = manager.format_schema_for_llm()
        self.assertIsInstance(result, str)
        os.remove(schema_path)
        os.rmdir(temp_dir)


if __name__ == "__main__":
    unittest.main()

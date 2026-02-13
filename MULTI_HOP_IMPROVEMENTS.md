# Multi-Hop Reasoning System Improvements

## Overview

This document describes the improvements made to the multi-hop reasoning system to address issues with schema understanding, context management, and query validation.

## Architecture

### Core Modules

1. **SchemaManager** (`tools/schema_manager.py`)
   - Manages knowledge graph schema
   - Validates Cypher queries against schema
   - Generates schema-aware prompts for LLM
   - Suggests corrections for common errors

2. **ContextManager** (`tools/context_manager.py`)
   - Manages reasoning trace with automatic compression
   - Prevents token overflow (90K token limit with buffer)
   - Detects reasoning loops and terminates early
   - Provides formatted context summaries for LLM

3. **QueryExamples** (`tools/query_examples.py`)
   - Library of common Cypher query patterns
   - Context-aware example selection
   - Error correction examples

4. **ReasoningDiagnostics** (`tools/reasoning_diagnostics.py`)
   - Diagnoses empty query results
   - Detects reasoning loops
   - Analyzes failure patterns
   - Generates debug reports

### Integration Points

The new modules integrate with `MultiHopReasoner` in `langchain_llm_qa_trial.py`:

```python
class MultiHopReasoner:
    def __init__(self, ..., schema_path=None):
        # Initialize schema and context management
        self.schema_manager = SchemaManager(schema_path)
        self.context_manager = ContextManager(max_tokens=90000)
    
    def run(self, question, top_k=5):
        # Uses context_manager for trace management
        # Uses schema_manager for validation
        # Returns enhanced results with statistics
```

## Key Features

### 1. Schema Validation

Cypher queries are now validated before execution:

```python
# Validates against schema
is_valid, error, suggestion = schema_manager.validate_cypher(cypher)

# Catches common errors:
# - Using 'id' property (should be 'geneName', 'primaryAccession', etc.)
# - Invalid node labels
# - Invalid relationship types
# - Too many OPTIONAL MATCH clauses
```

### 2. Context Compression

Automatically compresses trace when token limit approached:

```python
# Last 3 steps kept uncompressed (full detail)
# Older steps compressed (summary only)
# Token estimation: ~1 token per 4 characters
```

### 3. Loop Detection

Prevents infinite reasoning loops:

```python
# Detects patterns:
# - Repeated visits to same node (A → A)
# - Alternating pattern (A → B → A)
# - Cycles in recent steps
```

### 4. Enhanced Prompts

LLM prompts now include:
- Relevant schema information
- Common query patterns
- Error correction examples
- Schema reminders

## Usage

### Basic Usage (Backward Compatible)

No code changes required for existing usage:

```python
reasoner = MultiHopReasoner(llm, neo4j_connection, query_chain_factory)
result = reasoner.run(question="Find proteins encoded by gene EGFR")

# Returns familiar structure:
# {
#   "evidence": [...],
#   "trace": [...],
#   "final_action": "C"
# }
```

### Enhanced Usage (With New Features)

Take advantage of new features:

```python
# Specify custom schema path
reasoner = MultiHopReasoner(
    llm, 
    neo4j_connection, 
    query_chain_factory,
    schema_path="/path/to/graph_schema.json"
)

result = reasoner.run(question, top_k=5)

# Returns enhanced structure:
# {
#   "evidence": [...],           # Kept for compatibility
#   "trace": [...],              # From ContextManager
#   "final_action": "C",
#   "compressed": True/False,    # Whether compression occurred
#   "statistics": {              # Session statistics
#       "total_steps": 5,
#       "successful_steps": 4,
#       "empty_results": 1,
#       "estimated_tokens": 45000,
#       "compressed_steps": 2
#   }
# }
```

### Diagnostics

Generate debug reports for failed reasoning:

```python
from tools.reasoning_diagnostics import ReasoningDiagnostics

# Analyze failure patterns
analysis = ReasoningDiagnostics.analyze_failure_pattern(trace)
print(f"Pattern: {analysis['pattern']}")
print(f"Recommendation: {analysis['recommendation']}")

# Diagnose empty results
if result_count == 0:
    diagnosis = ReasoningDiagnostics.diagnose_empty_result(cypher)
    print(diagnosis)

# Generate full debug report
report = ReasoningDiagnostics.generate_debug_report(trace, final_answer)
```

## Schema File Format

The `graph_schema.json` file should follow this structure:

```json
{
  "nodes": [
    {"labels": ["Protein", "Gene", "Disease", ...]}
  ],
  "node_properties": [
    {
      "labels": "Protein",
      "properties": [
        {"property": "primaryAccession", "type": "STRING"},
        {"property": "geneName", "type": "STRING"},
        ...
      ]
    }
  ],
  "edges": [
    "(:Gene)-[:Gene_encodes_protein]->(:Protein)",
    ...
  ],
  "edge_properties": [
    {
      "type": "Gene_encodes_protein",
      "properties": []
    }
  ]
}
```

A sample schema file is provided at `graph_schema.json` in the repository root.

## Configuration

### Context Manager Settings

Adjust token limits and termination rules:

```python
context_manager = ContextManager(
    max_tokens=90000  # Adjust based on model's context window
)

# Termination rules (in ContextManager.should_terminate()):
# - Maximum 8 steps
# - 3 consecutive empty results
# - Repeated node visits (loop detection)
# - 3+ validation failures
```

### Schema Manager Settings

Customize validation behavior by modifying `schema_manager.py`:

```python
# Known primary keys (in get_primary_key)
primary_keys = {
    "Protein": "primaryAccession",
    "Gene": "geneName",
    # Add custom mappings
}

# Searchable properties (in get_searchable_properties)
searchable_map = {
    "Protein": ["primaryAccession", "geneName", "uniProtkbId"],
    # Add custom mappings
}
```

## Testing

Run unit tests:

```bash
# Schema Manager tests (14 tests)
python test/test_schema_manager.py

# Context Manager tests (16 tests)
python test/test_context_manager.py

# Existing multi-hop tests (19 tests)
python test/test_multi_hop_reasoning.py
```

## Performance Improvements

Based on the problem statement, expected improvements:

| Metric | Before | After |
|--------|--------|-------|
| Pass Rate | 60% (3/5) | 100% (5/5) |
| Average Steps | 4.6 | 2-3 |
| Cypher Success Rate | ~20% | ~70% |
| Context Overflow | 1/5 queries | 0/5 queries |
| Loop Detection | None | Automatic |

## Troubleshooting

### Schema Not Loading

If schema validation isn't working:
1. Check that `graph_schema.json` exists in the working directory
2. Verify JSON format is correct
3. SchemaManager will log warnings if schema file is missing

### Context Still Overflowing

If token limits are still exceeded:
1. Reduce `max_tokens` parameter to trigger compression earlier
2. Check token estimation is working (look for compression logs)
3. Reduce `max_steps` to limit reasoning depth

### Validation Too Strict

If validation rejects valid queries:
1. Update schema file with missing properties/relationships
2. Schema validation can be disabled by not providing schema_path
3. Check SchemaManager logs for validation details

## Migration Guide

### From Old to New System

No breaking changes - migration is optional:

1. **Minimal Migration** (Use new features without code changes)
   - Place `graph_schema.json` in working directory
   - New features activate automatically

2. **Full Migration** (Use all features)
   ```python
   # Old code:
   reasoner = MultiHopReasoner(llm, neo4j, query_factory)
   
   # New code (add optional parameter):
   reasoner = MultiHopReasoner(
       llm, neo4j, query_factory,
       schema_path="/path/to/schema.json"
   )
   ```

3. **Using New Return Fields**
   ```python
   result = reasoner.run(question)
   
   # Old fields (still available):
   evidence = result["evidence"]
   trace = result["trace"]
   
   # New fields:
   if result["compressed"]:
       print("Context was compressed")
   print(f"Stats: {result['statistics']}")
   ```

## Contributing

When extending the system:

1. **Adding New Node Types**: Update schema file and `primary_keys`/`searchable_map` in SchemaManager
2. **Adding Query Examples**: Add to `QUERY_EXAMPLES` dict in `query_examples.py`
3. **Adding Diagnostics**: Add methods to `ReasoningDiagnostics` class
4. **Adding Tests**: Follow existing test patterns in `test/test_*.py`

## References

- Original Issue: "改进多跳推理系统的Schema集成和上下文管理"
- Related Modules:
  - `langchain_llm_qa_trial.py` - Main reasoning engine
  - `neo4j_query_executor_extractor.py` - Neo4j integration
  - `entity_centric_schema_resolver.py` - Entity-specific schema filtering

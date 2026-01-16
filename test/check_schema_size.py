"""
检查 Neo4j schema 大小
"""
import os
import sys
from dotenv import load_dotenv

sys.path.insert(0, '/GenSIvePFS/users/clzeng/workspace/CROssBAR_LLM/crossbar_llm/backend')
load_dotenv('/GenSIvePFS/users/clzeng/workspace/CROssBAR_LLM/.env')

from tools.neo4j_query_executor_extractor import Neo4jGraphHelper

print("=" * 80)
print("CHECKING NEO4J SCHEMA SIZE")
print("=" * 80)

try:
    helper = Neo4jGraphHelper(
        URI=os.getenv("NEO4J_URI"),
        user=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("MY_NEO4J_PASSWORD"),
        db_name=os.getenv("NEO4J_DATABASE_NAME"),
        reset_schema=False,
        create_vector_indexes=False
    )
    
    schema = helper.create_graph_schema_variables()
    
    print(f"\n1. Schema components:")
    print(f"   - Nodes: {len(schema['nodes'])} types")
    print(f"   - Edges: {len(schema['edges'])} types")
    print(f"   - Node properties: {len(str(schema['node_properties']))} characters")
    print(f"   - Edge properties: {len(str(schema['edge_properties']))} characters")
    
    print(f"\n2. Total schema size when formatted:")
    from tools.qa_templates import CYPHER_GENERATION_TEMPLATE
    
    formatted_prompt = CYPHER_GENERATION_TEMPLATE.format(
        node_types=schema['nodes'],
        node_properties=schema['node_properties'],
        edge_properties=schema['edge_properties'],
        edges=schema['edges'],
        question="What is 2+2?"
    )
    
    print(f"   - Total characters: {len(formatted_prompt)}")
    print(f"   - Estimated tokens (chars/4): ~{len(formatted_prompt) // 4}")
    
    if len(formatted_prompt) > 50000:
        print(f"\n⚠️  WARNING: Prompt is very large!")
        print(f"   This could cause:")
        print(f"   - Slow API responses")
        print(f"   - Timeouts")
        print(f"   - Higher costs")    
        
    print(f"\n3. First 500 chars of formatted prompt:")
    print(formatted_prompt[:500])
    print("...")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

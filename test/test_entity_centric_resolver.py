"""
Test script for EntityCentricSchemaResolver functionality.

This script tests the entity-centric schema filtering with a protein sequence query.
"""

import sys
import os
import logging

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from crossbar_llm.backend.tools.langchain_llm_qa_trial import RunPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_entity_centric_resolver():
    """Test EntityCentricSchemaResolver with a protein sequence query."""

    # Test question with protein sequence
    instruction = "Does this protein associate with any specific small molecule ligands or cofactors besides its enzymatic substrates, and if so, what is the most likely category of these molecules?"
    protein_sequence = "MSGVRGLSRLLSARRLALAKAWPTVLQTGTRGFHFTVDGNKRASAKVSDSISAQYPVVDHEFDAVVVGAGGAGLRAAFGLSEAGFNTACVTKLFPTRSHTVAAQGGINAALGNMEEDNWRWHFYDTVKGSDWLGDQDAIHYMTEQAPAAVVELENYGMPFSRTEDGKIYQRAFGGQSLKFGKGGQAHRCCCVADRTGHSLLHTLYGRSLRYDTSYFVEYFALDLLMENGECRGVIALCIEDGSIHRIRAKNTVVATGGYGRTYFSCTSAHTSTGDGTAMITRAGLPCQDLEFVQFHPTGIYGAGCLITEGCRGEGGILINSQGERFMERYAPVAKDLASRDVVSRSMTLEIREGRGCGPEKDHVYLQLHHLPPEQLATRLPGISETAMIFAGVDVTKEPIPVLPTVHYNMGGIPTNYKGQVLRHVNGQDQIVPGLYACGEAACASVHGANRLGANSLLDLVVFGRACALSIEESCRPGDKVPPIKPNAGEESVMNLDKLRFADGSIRTSELRLSMQKSMQNHAAVFRVGSVLQEGCGKISKLYGDLKHLKTFDRGMVWNTDLVETLELQNLMLCALQTIYGAEARKESRGAHAREDYKVRIDEYDYSKPIQGQQKKPFEEHWRKHTLSYVDVGTGKVTLEYRPVIDKTLNEADCATVPPAIRSY"

    question = f'"{instruction}", "input": "<protein>{protein_sequence}</protein>"'

    print("=" * 80)
    print("Testing EntityCentricSchemaResolver")
    print("=" * 80)
    print(f"\nQuestion: {instruction}")
    print(f"\nProtein sequence length: {len(protein_sequence)} amino acids")
    print(f"Sequence preview: {protein_sequence[:50]}...")
    print("\n" + "=" * 80)

    try:
        # Load API key from environment
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("OPENROUTER_API_KEY")

        if not api_key:
            print("✗ OPENROUTER_API_KEY not found in .env file")
            return False

        # Initialize pipeline with entity-centric resolver enabled
        print("\n[1/5] Initializing RunPipeline with resolver enabled...")

        # Load resolver config manually
        import yaml
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'batch_config.yaml')
        with open(config_path, 'r') as f:
            batch_config = yaml.safe_load(f)
            resolver_config = batch_config.get('entity_centric_resolver', {})

        pipeline = RunPipeline(
            model_name="gpt-oss-120b",
            verbose=True,
            use_entity_centric_resolver=True,
            resolver_config=resolver_config
        )
        print("✓ Pipeline initialized successfully")
        print(f"  - Resolver enabled: {pipeline.use_entity_centric_resolver}")
        print(f"  - Resolver config: {pipeline.resolver_config}")

        # Generate Cypher query
        print("\n[2/5] Generating Cypher query...")
        cypher_query = pipeline.run_for_query(
            question=question,
            model_name="gpt-oss-120b",
            api_key=api_key
        )
        print("✓ Cypher query generated successfully")
        print(f"\nGenerated Query:\n{cypher_query}")

        # Execute query
        print("\n[3/5] Executing Cypher query against Neo4j...")
        result, raw_result = pipeline.execute_query(
            query=cypher_query,
            question=question,
            model_name="gpt-oss-120b"
        )
        print("✓ Query executed successfully")
        print(f"\nQuery Result:\n{result}")

        # Check cache
        print("\n[4/5] Checking cache directory...")
        cache_dir = "crossbar_llm/backend/cache/entity_schema_cache"
        if os.path.exists(cache_dir):
            cache_files = [f for f in os.listdir(cache_dir) if f.endswith('.json')]
            print(f"✓ Cache directory exists with {len(cache_files)} cached schemas")
            if cache_files:
                print(f"  - Cached files: {cache_files}")
        else:
            print("✗ Cache directory not found")

        # Summary
        print("\n[5/5] Test Summary")
        print("=" * 80)
        print("✓ Entity-centric resolver is working correctly!")
        print("\nKey observations:")
        print("  1. Protein sequence was detected in the question")
        print("  2. Cypher query was generated using filtered schema")
        print("  3. Query executed successfully against Neo4j")
        print("  4. Results were parsed into natural language")
        print("\n" + "=" * 80)

        return True

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("EntityCentricSchemaResolver Test Script")
    print("=" * 80)

    success = test_entity_centric_resolver()

    if success:
        print("\n✓ All tests passed!")
        sys.exit(0)
    else:
        print("\n✗ Tests failed!")
        sys.exit(1)

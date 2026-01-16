import os
import sys
from dotenv import load_dotenv

# Add backend to path
sys.path.insert(0, '/GenSIvePFS/users/clzeng/workspace/CROssBAR_LLM/crossbar_llm/backend')
load_dotenv('/GenSIvePFS/users/clzeng/workspace/CROssBAR_LLM/.env')

from tools.neo4j_query_executor_extractor import Neo4jGraphHelper

def initialize_vector_indexes():
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    password = os.getenv("MY_NEO4J_PASSWORD")
    db_name = os.getenv("NEO4J_DATABASE_NAME", "neo4j")

    print(f"Initializing Neo4jGraphHelper for {uri}...")
    
    try:
        # Initializing the helper with create_vector_indexes=True
        helper = Neo4jGraphHelper(
            URI=uri,
            user=user,
            password=password,
            db_name=db_name,
            reset_schema=False,
            create_vector_indexes=True  # This is the key
        )
        
        print("✅ Vector indexes creation triggered successfully.")
        
        # Verify
        import neo4j
        with neo4j.GraphDatabase.driver(uri, auth=(user, password)) as driver:
            with driver.session(database=db_name) as session:
                print("\n--- Current Vector Indexes ---")
                result = session.run("SHOW VECTOR INDEXES")
                for record in result:
                    print(f"Index Name: {record['name']}")
                    
    except Exception as e:
        print(f"❌ Error during initialization: {e}")

if __name__ == "__main__":
    initialize_vector_indexes()

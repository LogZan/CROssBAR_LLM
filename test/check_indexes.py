import os
import neo4j
from dotenv import load_dotenv

load_dotenv()

def check_indexes():
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    password = os.getenv("MY_NEO4J_PASSWORD")
    db_name = os.getenv("NEO4J_DATABASE_NAME", "neo4j")

    print(f"Connecting to: {uri}")
    try:
        with neo4j.GraphDatabase.driver(uri, auth=(user, password)) as driver:
            with driver.session(database=db_name) as session:
                print("\n--- Existing Indexes ---")
                result = session.run("SHOW INDEXES")
                for record in result:
                    print(f"Name: {record['name']}, Type: {record['type']}, State: {record['state']}, Labels: {record.get('labelsOrTypes')}, Properties: {record.get('properties')}")
                
                print("\n--- Specifically checking for Vector Indexes ---")
                # Neo4j 5.x command for vector indexes
                result = session.run("SHOW VECTOR INDEXES")
                found = False
                for record in result:
                    found = True
                    print(f"Vector Index Name: {record['name']}")
                
                if not found:
                    print("No vector indexes found.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_indexes()

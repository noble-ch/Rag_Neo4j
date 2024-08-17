from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import pinecone
import networkx as nx
import matplotlib.pyplot as plt

# Neo4j setup
uri = "bolt://localhost:7687"
driver = GraphDatabase.driver(uri, auth=("neo4j", "password"))

# Pinecone setup
pinecone.init(api_key='api-key')
index_name = 'neo4j-embedding-index'
pinecone.create_index(index_name, dimension=384)
index = pinecone.Index(index_name)

# Embedding model setup
model = SentenceTransformer('all-MiniLM-L6-v2')

def execute_query(query):
    with driver.session() as session:
        session.run(query)

def fetch_data():
    query = "MATCH (n)-[r]->(m) RETURN n, r, m"
    with driver.session() as session:
        result = session.run(query)
        return result.data()

def embed_data(data):
    texts = [f"{record['n']['name']} {record['r'].type} {record['m']['name']}" for record in data]
    embeddings = model.encode(texts)
    return embeddings

def upload_to_pinecone(embeddings, data):
    for i, embedding in enumerate(embeddings):
        index.upsert(vectors=[(str(i), embedding)], namespace='neo4j-data')

def nl_to_neoquery(nl_query):
    if "person" in nl_query.lower():
        return "MATCH (n:Person) RETURN n"
    else:
        return "MATCH (n) RETURN n"

def fetch_neo_data(nl_query):
    cypher_query = nl_to_neoquery(nl_query)
    with driver.session() as session:
        result = session.run(cypher_query)
        return result.data()

def visualize_graph(data):
    G = nx.Graph()
    for record in data:
        n = record['n']
        m = record['m']
        r = record['r']
        G.add_node(n['name'], label=n.labels)
        G.add_node(m['name'], label=m.labels)
        G.add_edge(n['name'], m['name'], label=r.type)
    
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True)
    labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()

def fetch_combined_results(nl_query):
    # Fetch from Neo4j
    neo_data = fetch_neo_data(nl_query)
    # Fetch from Pinecone
    query_embedding = model.encode(nl_query)
    pinecone_result = index.query(queries=[query_embedding], top_k=5, namespace='neo4j-data')
    # Combine and display results
    visualize_graph(neo_data)
    return pinecone_result

# Example workflow

# Populate Neo4j (example cypher query)
cypher_query = """
CREATE (a:Person {name: 'Nob'})-[:KNOWS]->(b:Person {name: 'Biru'})
"""
execute_query(cypher_query)

# Fetch, Transform, and Embed Data
data = fetch_data()
embeddings = embed_data(data)

# Upload Embeddings to Pinecone
upload_to_pinecone(embeddings, data)

# Query with Natural Language
nl_query = "Find all persons"
combined_results = fetch_combined_results(nl_query)
print(combined_results)


# name : noble biru
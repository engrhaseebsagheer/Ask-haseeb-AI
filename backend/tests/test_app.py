import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX_NAME")

# Init clients
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# Your test query
query = "Why is coding important in data science?"

# Generate embedding for the query
query_embedding = client.embeddings.create(
    model="text-embedding-3-large",
    input=query
).data[0].embedding

# Search in Pinecone
results = index.query(
    vector=query_embedding,
    top_k=5,
    include_metadata=True
)

# Print results
print("\n🔍 Top results for:", query)
for match in results["matches"]:
    score = match["score"]
    title = match["metadata"].get("title", "Untitled")
    source = match["metadata"].get("source", "Unknown")
    print(f"\nScore: {score:.4f}")
    print(f"Title: {title}")
    print(f"Source: {source}")

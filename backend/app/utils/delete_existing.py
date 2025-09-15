from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX_NAME")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

pc = Pinecone(api_key=PINECONE_API_KEY)

# --- Delete existing index if it exists ---
if PINECONE_INDEX in pc.list_indexes().names():
    print(f"🗑️ Deleting old index: {PINECONE_INDEX}")
    pc.delete_index(PINECONE_INDEX)

# --- Recreate index with same config ---
pc.create_index(
    name=PINECONE_INDEX,
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION)
)

print(f"✅ Created fresh index: {PINECONE_INDEX}")

import os
import json
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from tqdm import tqdm

# ------------------ Load Environment Variables ------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX_NAME")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

# ------------------ Init OpenAI & Pinecone ------------------
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# ------------------ Create Index if Missing ------------------
if PINECONE_INDEX not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION)
    )

index = pc.Index(PINECONE_INDEX)

# ------------------ Load Chunks ------------------
chunks_file = "backend/data/processed/chunks/all_chunks.jsonl"

if not os.path.exists(chunks_file):
    print(f"❌ File {chunks_file} not found!")
    exit()

with open(chunks_file, "r") as f:
    chunks = [json.loads(line) for line in f]

print(f"📄 Loaded {len(chunks)} chunks from {chunks_file}")

# ------------------ Generate & Store Embeddings ------------------
def get_embedding(text: str):
    """Generate embedding for a single text chunk using OpenAI v1.x API."""
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"⚠️ Error generating embedding: {e}")
        return None

print("🚀 Generating embeddings and pushing to Pinecone...")
success_count = 0
for chunk in tqdm(chunks, desc="Processing Chunks"):
    text = chunk.get("text", "")
    chunk_id = str(chunk.get("id", ""))
    metadata = {
        "title": chunk.get("title", "Untitled"),
        "source": chunk.get("source", "Unknown")
    }

    emb = get_embedding(text)
    if emb:
        index.upsert(vectors=[(chunk_id, emb, metadata)])
        success_count += 1

print(f"✅ Successfully stored {success_count}/{len(chunks)} chunks in Pinecone index '{PINECONE_INDEX}'.")

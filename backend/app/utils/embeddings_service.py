import os
import json
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

# -------------------------------
# 1) Load environment variables
# -------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX_NAME")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

# -------------------------------
# 2) Initialize clients
# -------------------------------
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# -------------------------------
# 3) Recreate index if exists
# -------------------------------
if PINECONE_INDEX in pc.list_indexes().names():
    print(f"🗑️ Deleting old index: {PINECONE_INDEX}")
    pc.delete_index(PINECONE_INDEX)

print(f"🆕 Creating new index: {PINECONE_INDEX}")
pc.create_index(
    name=PINECONE_INDEX,
    dimension=3072,  # using text-embedding-3-large
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION)
)

index = pc.Index(PINECONE_INDEX)

# -------------------------------
# 4) Load all chunks
# -------------------------------
chunks_file = "backend/data/processed/chunks/all_chunks.jsonl"

if not os.path.exists(chunks_file):
    print(f"❌ File {chunks_file} not found!")
    exit()

with open(chunks_file, "r") as f:
    chunks = [json.loads(line) for line in f]

print(f"📄 Loaded {len(chunks)} chunks from {chunks_file}")

# -------------------------------
# 5) Generate embeddings in batches
# -------------------------------
def get_embeddings_batch(texts):
    """Generate embeddings for a list of texts in one API call."""
    try:
        response = client.embeddings.create(
            model="text-embedding-3-large",  # More accurate than -small
            input=texts
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        print(f"⚠️ Error generating embeddings: {e}")
        return [None] * len(texts)

# -------------------------------
# 6) Upsert to Pinecone
# -------------------------------
BATCH_SIZE = 32
print("🚀 Generating embeddings and pushing to Pinecone...")

success_count = 0
for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Processing Batches"):
    batch = chunks[i:i+BATCH_SIZE]
    texts = [ch["text"] for ch in batch]
    embeddings = get_embeddings_batch(texts)

    vectors = []
    for ch, emb in zip(batch, embeddings):
        if emb is not None:
            vectors.append((
                ch["id"],
                emb,
                {
                    "title": ch.get("title", "Untitled"),
                    "source": ch.get("source", "Unknown")
                }
            ))

    if vectors:
        index.upsert(vectors=vectors)
        success_count += len(vectors)

print(f"\n✅ Successfully stored {success_count}/{len(chunks)} chunks in Pinecone index '{PINECONE_INDEX}'.")

import os
import uuid
from typing import Dict, List

from openai import OpenAI
from pinecone import Pinecone
from pathlib import Path
from backend.app.utils.universal_preprocess import process_file_to_chunks
from backend.app.utils.gdrive_service import list_files_in_folder, download_file
from backend.app.utils.state_store import load_state, save_state

# -------------------------------
# 1) Settings & Clients
# -------------------------------
GOOGLE_DRIVE_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")

RAW_DIR = Path("backend/data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

_oai = OpenAI(api_key=OPENAI_API_KEY)
_pc = Pinecone(api_key=PINECONE_API_KEY)
_index = _pc.Index(PINECONE_INDEX_NAME)

# -------------------------------
# 2) Embedding Utility
# -------------------------------
def embed_batch(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a batch of texts."""
    if not texts:
        return []
    resp = _oai.embeddings.create(
        model=OPENAI_EMBED_MODEL,
        input=texts
    )
    return [d.embedding for d in resp.data]

# -------------------------------
# 3) Upsert into Pinecone
# -------------------------------
def upsert_chunks(chunks: List[Dict], file_id: str):
    vectors = []
    for ch in chunks:
        vectors.append((
            ch.get("id", str(uuid.uuid4())),
            embed_batch([ch["text"]])[0],  # embed single chunk text
            {
                "title": ch.get("title"),
                "source": ch.get("source"),
                "text": ch.get("text"),
                "file_id": file_id
            }
        ))
    if vectors:
        _index.upsert(vectors=vectors)

# -------------------------------
# 4) Process a single new file
# -------------------------------
def process_single_file(file_id: str, name: str, mime: str, mtime: str):
    print(f"[INFO] Processing new file: {name}")
    base_name = name.replace("/", "_").strip()
    local_path = RAW_DIR / base_name

    # Download file from Google Drive
    downloaded_path = download_file(file_id, name, mime, str(local_path))

    # Run universal preprocessing → get chunks
    chunks = process_file_to_chunks(downloaded_path)

    # Embed & upsert chunks into Pinecone
    if chunks:
        upsert_chunks(chunks, file_id)
        print(f"[OK] {name}: {len(chunks)} chunks upserted to Pinecone")
    else:
        print(f"[SKIP] {name}: No chunks found")

    return chunks

# -------------------------------
# 5) Main Pipeline
# -------------------------------
def process_new_drive_files() -> Dict:
    """
    Detects new/updated files in Google Drive, downloads them,
    preprocesses into chunks, generates embeddings, and upserts to Pinecone.
    """
    if not GOOGLE_DRIVE_FOLDER_ID:
        print("[WARN] GOOGLE_DRIVE_FOLDER_ID not set. Skipping ingest.")
        return {"processed": 0, "skipped": 0}

    state = load_state()  # {file_id: modifiedTime}
    files = list_files_in_folder(GOOGLE_DRIVE_FOLDER_ID)

    new_files = []
    for f in files:
        fid, mtime = f["id"], f["modifiedTime"]
        if fid not in state or state[fid] != mtime:
            new_files.append(f)

    processed, skipped = 0, 0
    for f in new_files:
        fid = f["id"]
        name = f["name"]
        mime = f["mimeType"]
        mtime = f["modifiedTime"]

        chunks = process_single_file(fid, name, mime, mtime)
        if chunks:
            processed += 1
            state[fid] = mtime
        else:
            skipped += 1
            state[fid] = mtime  # Mark even if skipped, so we don't retry endlessly

    save_state(state)
    return {"processed": processed, "skipped": skipped, "found": len(new_files)}

"""Empty the Pinecone index and forget which Drive files were processed.

Use this once before re-ingesting, to clear out old content (for example the old ML project
documents). After it runs, restart the app: it will read every file in the Drive folder again.

Run from the project root:   python scripts/reset_index.py
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()
name = os.getenv("PINECONE_INDEX_NAME", "")
key = os.getenv("PINECONE_API_KEY", "")
if not key or not name:
    sys.exit("PINECONE_API_KEY and PINECONE_INDEX_NAME must be set in .env.")

index = Pinecone(api_key=key).Index(name)
stats = index.describe_index_stats()
print(f"Index {name!r} holds {stats.get('total_vector_count', 0)} vectors, dimension {stats.get('dimension')}.")

if input("Type DELETE to remove all of them: ").strip() != "DELETE":
    sys.exit("Nothing was changed.")

index.delete(delete_all=True)
state = Path("backend/data/processed/processed_files.json")
if state.exists():
    state.write_text("{}", encoding="utf-8")
print("Index emptied and the processed-files list cleared. Restart the app to re-ingest the Drive folder.")

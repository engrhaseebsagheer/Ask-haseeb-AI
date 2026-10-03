"""Check the OpenAI key, chat model and embedding model without starting the app.

Run from the project root:   python scripts/check_chat.py
Reads OPENAI_API_KEY, OPENAI_CHAT_MODEL, OPENAI_REASONING_EFFORT and OPENAI_EMBED_MODEL from .env.
"""
import os
import sys

from dotenv import load_dotenv
from openai import BadRequestError, OpenAI

load_dotenv()
key = os.getenv("OPENAI_API_KEY", "")
model = os.getenv("OPENAI_CHAT_MODEL", "gpt-6-luna")
effort = os.getenv("OPENAI_REASONING_EFFORT", "none")
embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")

if not key:
    sys.exit("OPENAI_API_KEY is empty. Add it to .env first.")

client = OpenAI(api_key=key, timeout=40)
messages = [
    {"role": "system", "content": "Answer in one short sentence."},
    {"role": "user", "content": "Say that the Ask Haseeb AI chat model is working."},
]
base = dict(model=model, messages=messages, max_completion_tokens=200)
try:
    resp = client.chat.completions.create(**base, reasoning_effort=effort)
    print(f"Chat model: {model} (reasoning effort: {effort})")
except BadRequestError as e:
    print(f"Note: {model} rejected reasoning_effort={effort!r} ({e}). Retrying without it.")
    resp = client.chat.completions.create(**base)
    print(f"Chat model: {model}")
print(f"Reply: {resp.choices[0].message.content}")
if resp.usage:
    print(f"Tokens: {resp.usage.prompt_tokens} in, {resp.usage.completion_tokens} out")

emb = client.embeddings.create(model=embed_model, input="test")
print(f"Embedding model: {embed_model}, {len(emb.data[0].embedding)} dimensions (the Pinecone index must match)")

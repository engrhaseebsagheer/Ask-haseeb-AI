# Ask Haseeb AI

Ask Haseeb AI is a retrieval-augmented generation (RAG) assistant built by Haseeb Sagheer. It ingests documents from Google Drive, stores embeddings in Pinecone, and answers questions through a FastAPI backend and a chat interface. It runs at ask.haseebsagheer.com and the code is public.

Type: Automation tool. Status: Live. Built by Haseeb Sagheer, solo.
What the status means: The assistant is running at https://ask.haseebsagheer.com. It was first built in September 2025, went offline for a period, and was rebuilt and relaunched in October 2026. The source code is public on GitHub.
Links: https://github.com/engrhaseebsagheer/ask-haseeb-ai
Portfolio page: https://haseebsagheer.com/projects/ask-haseeb-ai/

## Overview
Ask Haseeb AI is an assistant that answers questions about my skills, projects and experience. It does not rely on what a language model happens to know. It retrieves the relevant parts of my own documents and writes the answer from those.

Documents live in a Google Drive folder. When a new file appears, the system picks it up, cleans it, splits it into overlapping chunks of about 1,000 characters, embeds each chunk, and stores the vectors in Pinecone. A question is embedded the same way, the closest chunks are retrieved, and an OpenAI chat model writes the answer from that context.

Around that sits a FastAPI backend with REST endpoints, a simple chat front end, and a production deployment on a VPS behind Nginx with HTTPS. It is the project where I built every layer of a retrieval system myself.

## The problem Ask Haseeb AI solves
Recruiters and clients ask the same questions about skills, projects and experience. The assistant answers them from the actual documents instead of a static page.

## Who Ask Haseeb AI is for
Recruiters, clients and collaborators who want answers about my work without reading every page.

## What Ask Haseeb AI does
- Detects new files in Google Drive and processes them automatically
- Handles PDF, Markdown, HTML, plain text and Google Docs
- Chunks text at about 1,000 characters with a 200-character overlap to keep context
- Stores OpenAI embeddings in Pinecone
- Retrieves relevant chunks and generates an answer with an OpenAI chat model
- Serves queries through REST endpoints and a chat interface

## How Ask Haseeb AI works, step by step
1. Ingest: Google Drive is watched for new documents.
2. Chunk: Documents are cleaned and split into overlapping chunks.
3. Embed: Chunks are embedded and stored in Pinecone.
4. Retrieve: A question pulls the most relevant chunks.
5. Answer: An OpenAI chat model writes the answer from that context.

## Engineering notes for Ask Haseeb AI
- Ingestion: The Google Drive API detects new files and triggers processing. PDF, Markdown, HTML, plain text and Google Docs are supported. Edited files are re-indexed and deleted files are removed from the index.
- Chunking: Documents are split into chunks of about 1,000 characters with a 200-character overlap, so meaning carries across chunk boundaries. The first version used about 350 tokens per chunk.
- Retrieval: OpenAI embeddings (3072 dimensions) are stored in Pinecone and searched for each question.
- Serving: A FastAPI backend exposes REST endpoints for queries and background jobs. It was deployed on a VPS with Gunicorn, Uvicorn and Nginx over HTTPS.
- Index management: The index is recreated automatically and document metadata is managed alongside it.

## History of Ask Haseeb AI
Version 1 was built in six days in September 2025:
1. Day 1: preprocessing pipeline for PDF, text, Markdown and HTML, with cleaning and chunking.
2. Day 2: embeddings with OpenAI text-embedding-3-large (3072 dimensions) stored in Pinecone, with cosine similarity search.
3. Day 3: the RAG pipeline and FastAPI endpoints.
4. Day 4: automatic ingestion from Google Drive on a schedule.
5. Day 5: the chat front end in plain HTML, CSS and JavaScript.
6. Day 6: deployment on an Ubuntu VPS with Gunicorn, Nginx, HTTPS and a systemd service.

Version 2, October 2026, changed these things:
- A new interface that matches haseebsagheer.com, with answers that stream as they are written.
- Follow-up questions keep the context of the conversation.
- A choice of short or detailed answers.
- Sources shown with each answer.
- Edited and deleted Drive files are now updated in or removed from the index.
- A per-visitor rate limit to keep running costs low.
- A cheaper, newer OpenAI chat model.
- A rewritten knowledge base that contains only professional information.

## Built with
Python, FastAPI, OpenAI, Pinecone, Google Drive API, APScheduler, Gunicorn, Nginx

## Haseeb's role on Ask Haseeb AI
I built the whole system: ingestion, preprocessing, embeddings, the RAG chain, the API, the chat front end and the deployment.

## What is next for Ask Haseeb AI
- Role-based access for public and private data
- Query logs and document usage statistics
- Translation for wider access
- A React or Next.js front end

## Questions about Ask Haseeb AI
### What is Ask Haseeb AI?
A RAG assistant that answers questions about Haseeb Sagheer’s skills, projects and experience from his own documents.

### Is the demo live?
Yes, at https://ask.haseebsagheer.com. The code is on GitHub.

### Which models and tools does it use?
OpenAI embeddings and an OpenAI chat model for answers, Pinecone for vector search, and FastAPI for the backend. The first version used LangChain for text splitting; the current version uses its own splitter.

### Where does its knowledge come from?
From documents in a Google Drive folder, which are ingested automatically when new files appear.


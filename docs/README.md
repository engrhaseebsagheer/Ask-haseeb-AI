# Ask Haseeb AI

Ask Haseeb AI is a **Retrieval-Augmented Generation (RAG)** powered personal AI assistant designed to answer questions about Haseeb Sagheer's portfolio, projects, and professional background.  
The system leverages **LangChain**, **OpenAI GPT-4**, **Pinecone**, and **FastAPI** to deliver a highly accurate, dynamic, and interactive experience.  
It will be deployed live at **[ask.haseebsagheer.com](https://ask.haseebsagheer.com/)**.

---

## 🚀 Project Overview
Ask Haseeb AI allows users to:
- Ask questions about Haseeb Sagheer's projects, skills, and experience.
- Retrieve answers directly from curated documents (resume, LinkedIn posts, blogs, portfolio).
- Interact with a conversational AI powered by **RAG architecture** for accurate, contextual responses.

---

## 🏗️ Tech Stack

### Core Tools
- **Backend:** FastAPI  
- **RAG Framework:** LangChain  
- **Vector Database:** Pinecone  
- **LLM:** OpenAI GPT-4 API  
- **Frontend:** React.js  
- **Data Storage:** AWS S3 (for dynamic document uploads)  
- **Deployment:** Docker, Gunicorn, Nginx, VPS (Ubuntu)  
- **Authentication:** Auth0  
- **Analytics:** Weights & Biases  
- **Project Management:** Jira  

---

## 📂 Features

- **RAG-powered Q&A:** Retrieve and answer queries using curated personal data.  
- **Dynamic Data Upload:** Add new documents without redeployment.  
- **Conversation Memory:** Context-aware multi-turn conversations.  
- **Citations:** Answers include source references for transparency.  
- **Authentication:** Restrict access if needed using Auth0.  
- **Analytics:** Track usage metrics and most asked questions with W&B.  
- **Deployment:** Production-ready setup on VPS with custom domain.  

---

## 📅 Development Plan

### Phase 1 – Core MVP
- Document ingestion → Embeddings → Pinecone storage  
- RAG pipeline → LLM integration → Basic Q&A interface  
- Deployment on VPS with custom domain  

### Phase 2 – Advanced Features
- Dynamic data upload  
- Conversation memory & citations  
- Authentication & analytics integration  
- UI polish & branding  

---

## 🧰 Installation & Setup

### Prerequisites
- Python 3.10+
- Node.js & npm
- Docker
- OpenAI API Key
- Pinecone API Key
- Auth0 Credentials (optional)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/username/ask-haseeb-ai.git
cd ask-haseeb-ai

# 2. Setup backend (FastAPI)
cd backend
pip install -r requirements.txt

# 3. Setup frontend (React)
cd ../frontend
npm install

# 4. Environment variables
Create a `.env` file in the backend folder with:
OPENAI_API_KEY=your_key
PINECONE_API_KEY=your_key

# 5. Run backend
uvicorn main:app --reload

# 6. Run frontend
npm start
```

---

## 🌐 Deployment
- Containerize backend + frontend using **Docker**  
- Use **Gunicorn + Nginx** on VPS for production readiness  
- Set up DNS for **ask.haseebsagheer.com**  

---

## 📊 Analytics
- Integrated with **Weights & Biases** for query logging and usage analysis.

---

## 🔒 Authentication
- Optional **Auth0** integration for private access before public launch.

---

## 🤝 Contributing
Contributions are welcome! Feel free to open issues or pull requests for feature suggestions or improvements.

---

## 📜 License
This project is licensed under the MIT License.

---

## 👤 Author
**Haseeb Sagheer**  
- Portfolio: [haseebsagheer.com](https://haseebsagheer.com)  
- LinkedIn: [linkedin.com/in/haseebsagheer](https://linkedin.com/in/haseebsagheer)  

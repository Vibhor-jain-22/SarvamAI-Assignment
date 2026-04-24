# 🚀 Bike Troubleshooting Assistant (RAG-Based)

## 🧠 Overview

This project implements a **Retrieval-Augmented Generation (RAG) based AI assistant** that answers user queries **strictly from a provided bike manual (PDF)**.

The system is designed with a **zero-hallucination principle**:

> If the answer is not present in the manual, the assistant refuses to respond.

---

## 🎯 Key Capabilities

* 📄 PDF ingestion and processing
* 🔍 Semantic search using embeddings (ChromaDB)
* 🤖 LLM-powered answer generation (Mistral)
* 🛑 Strict grounding (no hallucinations)
* 📌 Source attribution with page numbers

---

### 🔄 End-to-End Flow

1. **PDF Upload**

   * User uploads a bike manual
   * Text is extracted and chunked

2. **Embedding Pipeline**

   * Text chunks are converted into embeddings
   * Stored in **ChromaDB (persistent storage)**

3. **Query Processing**

   * User submits a question via UI

4. **Retrieval**

   * Relevant chunks retrieved using similarity search

5. **LLM Response Generation**

   * Mistral LLM generates answer strictly from retrieved context
   * If context is insufficient → refusal response

---

## 🧰 Tech Stack

### 🔹 Backend

* Python + FastAPI
* Uvicorn

### 🔹 LLM

* Mistral API (`mistral-small-latest`)

### 🔹 Embeddings & Retrieval

* Mistral Embeddings (`mistral-embed`)
* ChromaDB (persistent vector store)

### 🔹 Frontend

* Vanilla JavaScript + HTML

---

## 📁 Project Structure

SarvamAI-Assignment/
│
├── backend/
│   ├── main.py              # FastAPI app
│   ├── storage/             # Chroma DB + logs
│   └── requirements.txt
│
├── frontend/
│   └── index.html           # UI
│
├── embedding.py             # Embedding logic
└── README.md
```

---

⚙️ Setup Instructions

1️⃣ Create virtual environment

python3 -m venv .venv
source .venv/bin/activate
```

---

2️⃣ Install dependencies

pip install -r backend/requirements.txt
```

---

### 3️⃣ Set environment variables

export MISTRAL_API_KEY="YOUR_KEY"
export MISTRAL_CHAT_MODEL="mistral-small-latest"
export MISTRAL_EMBED_MODEL="mistral-embed"
```

---

4️⃣ Run backend

uvicorn backend.main:app --reload --port 8000
```

---

### 5️⃣ Open frontend

Open:

frontend/index.html
```

---

## 🧪 API Endpoints

### 📄 Upload Manual

POST /upload
```

### ❓ Query

POST /query
```

---

## 🧪 Sample Queries

* “What is the recommended engine oil grade?”
* “How do I adjust the clutch lever free play?”
* “What does the ABS warning light indicate?”

### ❌ Out-of-scope example

> “What is the capital of France?”
> → Returns:
> `Sorry, this information is not available in the manual.`

---

## 🧠 Design Decisions

### 🔒 Grounded Answering

* Responses are strictly based on retrieved context
* Prevents hallucinations

### 📦 Persistent Vector Store

* ChromaDB stores embeddings for fast retrieval

### ⚙️ Simple, Modular Backend

* Clear separation between ingestion, retrieval, and generation

---

## ⚠️ Limitations

* No multimodal support (text-only queries)
* Single-LLM dependency (Mistral)
* Retrieval quality depends on chunking and embeddings

---

## 🚀 Future Improvements

* Add support for multiple LLM providers
* Improve retrieval with hybrid search (keyword + semantic)
* Add re-ranking for better answer quality
* Build hosted UI instead of static frontend
* Add evaluation metrics

---

## 💼 Business Impact

* Reduces time spent reading manuals
* Enables instant troubleshooting
* Improves accessibility of technical information

---

## 👤 Author

**Vibhor Jain**

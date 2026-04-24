# Bike Troubleshooting Assistant (RAG)

A FastAPI + vanilla JS web app that answers questions **strictly** from an uploaded manual PDF using Retrieval Augmented Generation (RAG). If the manual doesn’t contain the answer, it refuses with:

`Sorry, this information is not available in the manual.`

## Setup

### 1) Create venv + install dependencies

From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

### 2) Set environment variables

```bash
export MISTRAL_API_KEY="YOUR_KEY"

# Optional:
export MISTRAL_CHAT_MODEL="mistral-small-latest"
export MISTRAL_EMBED_MODEL="mistral-embed"

# If you prefer OpenAI instead:
# export OPENAI_API_KEY="YOUR_KEY"
# export OPENAI_CHAT_MODEL="gpt-4.1-mini"
# export OPENAI_EMBED_MODEL="text-embedding-3-small"
# export OPENAI_VISION_MODEL="gpt-4o-mini"

# If you prefer Gemini instead:
# export GEMINI_API_KEY="YOUR_KEY"
# export GEMINI_CHAT_MODEL="gemini-2.0-flash"
# export GEMINI_EMBED_MODEL="gemini-embedding-001"
# export GEMINI_VISION_MODEL="gemini-2.0-flash"

export LOG_LEVEL="INFO"
```

### 3) Run the backend

```bash
uvicorn backend.main:app --reload --port 8000
```

### 4) Open the frontend

- Open `frontend/index.html` in your browser.
- Ensure the backend is running at `http://127.0.0.1:8000`.

## Usage

1) Click **Upload manual PDF** and upload a bike manual.
2) Ask questions in chat; the response includes **sources with page numbers**.
3) Bonus: click **Ask with image** to upload an image; it will be turned into a text query before RAG.
   - If you’re using **Mistral** as the provider, image queries currently return a 400 because vision isn’t implemented in this app for Mistral.

## API

- `POST /upload` (multipart): field `pdf` (PDF file)
- `POST /query`:
  - JSON: `{ "question": "..." }`
  - or multipart: field `question` (string) and/or `image` (image file)

### curl examples

Upload a PDF:

```bash
curl -sS -X POST "http://127.0.0.1:8000/upload" \
  -F "pdf=@RAG/bullet-350.pdf"
```

Ask a question (JSON):

```bash
curl -sS -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question":"How do I adjust the clutch lever free play?"}'
```

Ask with an image (multipart):

```bash
curl -sS -X POST "http://127.0.0.1:8000/query" \
  -F "image=@/path/to/bike_issue.jpg"
```

## Where data is stored

- Chroma persistence: `backend/storage/chroma/`
- Manifest: `backend/storage/manifest.json`
- Evaluation logs (JSONL): `backend/storage/eval_logs.jsonl`

## Sample test queries

After uploading a manual, try:

- “What is the recommended engine oil grade?”
- “How do I adjust the clutch lever free play?”
- “What does the ABS warning light indicate?”
- Out-of-manual check: “What is the capital of France?” (must refuse)


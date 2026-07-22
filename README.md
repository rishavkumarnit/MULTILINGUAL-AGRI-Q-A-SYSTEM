# Multilingual Agri Q&A System

Three independently runnable services form the assistant:

- `agri-assistant-frontend` — React + TypeScript user interface; calls only the Node API.
- `agri-assistant-backend` — Node.js + TypeScript + Express; validates public requests, manages conversations, and calls the internal AI service.
- `agri-assistant-ai` — Python + FastAPI; owns translation, embeddings, semantic search, RAG, LangGraph, tools, and answer generation.

## Run locally

Start each service in a separate terminal.

```powershell
cd agri-assistant-ai
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

```powershell
cd agri-assistant-backend
pnpm install
pnpm run dev
```

```powershell
cd agri-assistant-frontend
pnpm install
pnpm run dev
```

The frontend runs at `http://localhost:5173`, Node at `http://localhost:4001`, and FastAPI at `http://localhost:8000`.

## Request flow

`React → POST /api/chat → Node API → POST /internal/chat → Python AI service → Node API → React`

Only the Node API is public. The Python service is an internal boundary and must not be called directly by the browser.

## MongoDB Atlas and semantic reuse

Set the same `MONGODB_URI` and `MONGODB_DATABASE` in both `agri-assistant-backend/.env` and `agri-assistant-ai/.env`. The Node API stores conversation messages in the `conversations` collection.

The AI service embeds English questions with `text-embedding-3-small` and runs `$vectorSearch` against the `verified_answers` collection. This learning project directly reuses `DEMO_VERIFIED` and `EXPERT_VERIFIED` documents when their crop and location fields are compatible. Use `DEMO_VERIFIED` for your own sample answers; reserve `EXPERT_VERIFIED` for real reviewed content.

In Atlas, create a Vector Search index named `verified_answers_vector` on `verified_answers`, using [agri-assistant-ai/atlas/verified_answers_vector_index.json](agri-assistant-ai/atlas/verified_answers_vector_index.json). The embedding dimension is `1536` for the configured default embedding model.

Each expert-reviewed document must use this shape (normalize `crop` and `location` to lowercase):

```json
{
  "questionEnglish": "How often should wheat be irrigated in Delhi?",
  "answerEnglish": "Approved expert answer goes here.",
  "crop": "wheat",
  "location": "delhi",
  "status": "DEMO_VERIFIED",
  "embedding": [0.0],
  "embeddingModel": "text-embedding-3-small"
}
```

Populate `embedding` with the OpenAI embedding for `questionEnglish` before inserting the document. Do not label generated content as `EXPERT_VERIFIED`.

To create a vetted record safely, use the ingestion helper from `agri-assistant-ai`:

```powershell
python -m scripts.ingest_verified_answer --question "How often should wheat be irrigated in Delhi?" --answer "Demo answer for semantic-search testing" --crop wheat --location delhi
```

## RAG document ingestion

When a question has no crop+location-matched verified answer, the AI service falls back to retrieval-augmented generation over trusted agricultural documents stored in the `document_chunks` collection.

In Atlas, create a second Vector Search index named `document_chunks_vector` on `document_chunks`, using [agri-assistant-ai/atlas/document_chunks_vector_index.json](agri-assistant-ai/atlas/document_chunks_vector_index.json).

Sample documents live in `agri-assistant-ai/data/documents/`. Ingest each one with the chunking + embedding CLI:

```powershell
python -m scripts.ingest_document --path data/documents/wheat_irrigation.md --title "Wheat Irrigation Guide" --crop wheat
python -m scripts.ingest_document --path data/documents/rice_pest_management.md --title "Rice Pest Management" --crop rice
python -m scripts.ingest_document --path data/documents/soil_health_composting.md --title "Soil Health and Composting Basics"
python -m scripts.ingest_document --path data/documents/monsoon_crop_planning.md --title "Monsoon Crop Planning"
python -m scripts.ingest_document --path data/documents/fertilizer_application_basics.md --title "Fertilizer Application Basics"
```

The script loads the document with a LangChain `TextLoader`, chunks it with `RecursiveCharacterTextSplitter.from_tiktoken_encoder(...)`, embeds each chunk with `EMBEDDING_MODEL` via `langchain_openai.OpenAIEmbeddings`, and upserts into `document_chunks`. Re-running the same `--title` replaces its existing chunks, so it's safe to re-ingest after editing a document. Retrieval goes through `app/retrievers.py`'s `AtlasVectorRetriever` (a custom LangChain `BaseRetriever`), unfiltered by crop/location — general documents are matched by semantic similarity alone — and answers that draw on retrieved chunks are returned with `source: "rag-generated"` and a `sources` list naming the document titles used.

## MCP server

`agri-assistant-ai/mcp_server.py` is a standalone [MCP](https://modelcontextprotocol.io) server that exposes the weather and mandi price tools (`agri-assistant-ai/app/weather.py`, `agri-assistant-ai/app/crop_price.py`) to any MCP client — it does not sit on the FastAPI request path; the AI service calls those same functions directly. Run it from `agri-assistant-ai` with the venv active:

```powershell
python mcp_server.py
```

It communicates over stdio, so most MCP clients spawn it as a subprocess rather than connecting to a port. Example Claude Desktop / Claude Code config entry:

```json
{
  "mcpServers": {
    "agri-assistant-tools": {
      "command": "C:\\path\\to\\agri-assistant-ai\\.venv\\Scripts\\python.exe",
      "args": ["C:\\path\\to\\agri-assistant-ai\\mcp_server.py"]
    }
  }
}
```

It exposes two tools: `get_weather_forecast(location)` and `get_crop_price(crop, location)` (the latter needs `DATA_GOV_IN_API_KEY` set in `agri-assistant-ai/.env`).

# Multilingual Agri Q&A System

Two independently runnable services form the assistant:

- `agri-assistant-frontend` — React + TypeScript user interface; calls the AI service directly.
- `agri-assistant-ai` — Python + FastAPI; public-facing API (CORS enabled, request validation, conversation persistence) plus translation, embeddings, semantic search, RAG, LangGraph, tools, and answer generation.

There used to be a third, Node/Express service (`agri-assistant-backend`) sitting between
the frontend and the AI service as a public gateway. It was removed — CORS, request
validation, and conversation persistence all moved into FastAPI (`app/main.py`,
`app/conversations.py`) — so the frontend now talks to the AI service directly. CORS
defaults to wide open (`ALLOWED_ORIGINS` unset → `*`); the "Deploy" section below covers
locking it to the real frontend domain.

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
cd agri-assistant-frontend
pnpm install
pnpm run dev
```

The frontend runs at `http://localhost:5173` and proxies `/api` to FastAPI at
`http://localhost:8000` (see `vite.config.ts`).

## Deploy (Render + Vercel)

Push everything first — both platforms deploy from the GitHub repo (`origin/master`),
including the new `render.yaml` at the repo root and `agri-assistant-frontend/.env.example`.

### 1. Backend on Render

1. [render.com](https://render.com) → sign in with GitHub → **New +** → **Blueprint**.
2. Select this repo. Render reads `render.yaml` automatically and proposes one web
   service (`agri-assistant-ai`, free plan, `rootDir: agri-assistant-ai`).
3. Before the first deploy completes, fill in the secret environment variables it lists
   (declared with `sync: false` in `render.yaml`, so Render prompts for values rather
   than reading them from git): `OPENAI_API_KEY`, `MONGODB_URI`, `MONGODB_DATABASE`,
   `DATA_GOV_IN_API_KEY`, plus the optional overrides in
   `agri-assistant-ai/.env.example` if you want non-default values. Copy the values from
   your local `agri-assistant-ai/.env` — leave `ALLOWED_ORIGINS` blank for now.
4. **MongoDB Atlas → Network Access**: add `0.0.0.0/0` ("allow access from anywhere").
   Render's free tier has no static outbound IP, so a specific-IP allowlist won't work.
5. Deploy, then note the resulting URL (e.g. `https://agri-assistant-ai.onrender.com`).
   Free-tier services spin down after ~15 minutes idle — the first request after that can
   take 30-60s while it wakes back up; that's expected, not a bug.

### 2. Frontend on Vercel

1. [vercel.com](https://vercel.com) → sign in with GitHub → **Add New** → **Project** →
   import this repo.
2. Set **Root Directory** to `agri-assistant-frontend` (this is a monorepo). Vercel
   auto-detects the Vite preset; build command `pnpm run build` / output `dist` need no
   changes.
3. Add environment variable `VITE_API_BASE_URL` = the Render URL from step 1 (no
   trailing slash).
4. Deploy, then note the resulting URL (e.g. `https://your-app.vercel.app`).

### 3. Lock down CORS

The backend's `ALLOWED_ORIGINS` env var was left blank in step 1, so it's currently wide
open (`*`) — fine to get both deploys working end-to-end first, but tighten it now:

1. Render dashboard → `agri-assistant-ai` → **Environment** → set `ALLOWED_ORIGINS` to
   the exact Vercel URL from step 2.
2. Saving triggers an automatic redeploy.

From here, both platforms auto-deploy on every push to `master` — no further manual
steps for future changes.

## Request flow

`React → POST /api/chat (or /api/chat/stream) → FastAPI → React`

## Conversations and grounding

Each browser session keeps one `conversationId` (generated on the first message, then
reused for every message after) and sends it on every request. The AI service loads the
conversation's persisted messages and feeds the last `CONTEXT_TURN_LIMIT` (6) turns back
to both the translation/extraction step and the answer-generating agent, so follow-ups
like "and for rice?" resolve pronouns and carried-over crop/location against the actual
prior exchange, not just the current message in isolation (`app/workflow.py`'s
`_format_history`). A conversation is capped at `MAX_CONVERSATION_MESSAGES` (50) stored
messages — once reached, further messages on that `conversationId` are rejected with a
400 asking the user to start a new conversation (`app/conversations.py`).

When no tool call and no verified/RAG match grounds an answer, the agent may still
answer from its own general knowledge rather than refusing outright — but the answer is
forced (in code, not just prompted) to start with the disclaimer "General AI answer (not
verified against our database):" and is tagged `source: "llm-general"`, which the
frontend renders as a distinct warning badge. This is a portfolio/learning project, not
a real advisory service — the UI carries a permanent disclaimer banner saying so.

## MongoDB Atlas and semantic reuse

Set `MONGODB_URI` and `MONGODB_DATABASE` in `agri-assistant-ai/.env`. The AI service stores conversation messages in the `conversations` collection.

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
python -m scripts.ingest_document --path data/documents/mustard_pest_management.md --title "Mustard Pest Management" --crop mustard
python -m scripts.ingest_document --path data/documents/cotton_pest_management.md --title "Cotton Pest Management" --crop cotton
python -m scripts.ingest_document --path data/documents/maize_irrigation_fertilization.md --title "Maize Irrigation and Fertilization" --crop maize
python -m scripts.ingest_document --path data/documents/potato_disease_management.md --title "Potato Disease Management" --crop potato
python -m scripts.ingest_document --path data/documents/sugarcane_irrigation_scheduling.md --title "Sugarcane Irrigation Scheduling" --crop sugarcane
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

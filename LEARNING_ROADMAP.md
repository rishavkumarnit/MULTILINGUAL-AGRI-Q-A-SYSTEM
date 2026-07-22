# AI Learning Roadmap

This project is intentionally a single, portfolio-ready system that demonstrates modern AI application patterns.

| Topic | Where it belongs in this project | Status |
| --- | --- | --- |
| Multilingual LLM application | Question translation and answer localization | Implemented |
| Structured output | Crop/location extraction JSON schema | Implemented |
| Embeddings | English question vectors | Implemented |
| Vector database / semantic search | MongoDB Atlas `$vectorSearch` over verified Q&A | Implemented |
| Context-aware retrieval | Crop and location filters on vector search | Implemented |
| Conversation persistence | Node API + MongoDB `conversations` collection | Implemented |
| RAG | Trusted agricultural document ingestion and chunk retrieval | Implemented |
| LangChain | Document loaders, chunking, retrievers, prompt templates | Planned |
| LangGraph | Explicit state graph for the assistant workflow | Implemented |
| Tool calling | Weather and location tools for time-sensitive questions | In progress (weather done, mandi price pending) |
| MCP | Dedicated MCP server exposing agriculture and weather tools | Planned |
| Agents | Tool-selection policy layered above the LangGraph workflow | Planned |
| Streaming | Token streaming from AI service through Node to React | Planned |
| Evaluations and observability | Test dataset, retrieval metrics, traces, and feedback | Planned |

## Portfolio flow

`React → Node API → FastAPI → translate/extract → embedding + Atlas Vector Search → reuse or RAG → LLM → translate → React`

Each stage is intentionally separate, so you can explain its responsibility, data, and evaluation method in interviews.

## Tool calling

Two tools are planned as graph nodes/branches on top of the LangGraph workflow.

1. **Weather forecast tool** — Implemented. `agri-assistant-ai/app/weather.py` geocodes the
   extracted location and fetches a 5-day forecast from Open-Meteo (no API key needed).
   The extraction step now classifies `intent: "weather" | "general"`; when `intent` is
   `weather` and a location was given, the graph routes to a `fetch_weather` node before
   falling back to the normal semantic-search/RAG pipeline if the location can't be
   resolved or the API call fails. Response `source: "weather-forecast"`.
2. **Mandi price tool** — Not started. Crop market price lookup via the data.gov.in
   Agmarknet API (resource `9ef84268-d588-465a-a308-a864a43d0070`), triggered when the
   question is about price/rate. Deliberately built last. A free API key has already been
   obtained and is set as `DATA_GOV_IN_API_KEY` in `agri-assistant-ai/.env` (gitignored,
   not committed); the placeholder is in `.env.example`.

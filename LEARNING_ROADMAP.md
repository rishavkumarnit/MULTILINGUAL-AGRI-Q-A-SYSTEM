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
| LangChain | Document loaders, chunking, retrievers, prompt templates | Implemented |
| LangGraph | Explicit state graph for the assistant workflow | Implemented |
| Tool calling | Weather and mandi price tools for time-sensitive questions | Implemented |
| MCP | Dedicated MCP server exposing agriculture and weather tools | Implemented |
| Agents | Tool-selection policy layered above the LangGraph workflow | Planned |
| Streaming | Token streaming from AI service through Node to React | Planned |
| Evaluations and observability | Test dataset, retrieval metrics, traces, and feedback | Planned |

## Portfolio flow

`React → Node API → FastAPI → translate/extract → embedding + Atlas Vector Search → reuse or RAG → LLM → translate → React`

Each stage is intentionally separate, so you can explain its responsibility, data, and evaluation method in interviews.

## LangChain

The four named patterns are demonstrated without changing the app's response contract
or behavior — this was a portfolio/interview-value swap of internals, not a feature.

- **Document loaders**: `scripts/ingest_document.py` loads source files via
  `langchain_community.document_loaders.TextLoader` instead of a raw `Path.read_text()`.
  (`langchain-community` carries a "being sunset" deprecation notice upstream but still
  works and is the pattern most tutorials/interviews reference — noted here so it isn't
  a surprise later.)
- **Chunking**: the hand-rolled token-window splitter (`app/chunking.py`, now deleted)
  was replaced with `langchain_text_splitters.RecursiveCharacterTextSplitter
  .from_tiktoken_encoder(...)`, same `chunk_size=300, chunk_overlap=50` defaults.
- **Retrievers**: `agri-assistant-ai/app/retrievers.py` defines `AtlasVectorRetriever`,
  a custom `langchain_core.retrievers.BaseRetriever` subclass wrapping the app's
  existing async MongoDB `$vectorSearch` aggregation. **Deliberately not using**
  `langchain-mongodb`'s prebuilt `MongoDBAtlasVectorSearch`: it only supports a
  synchronous `pymongo.Collection` (no working async-native support), which would have
  required a second, separate sync Mongo connection alongside the app's existing
  `AsyncMongoClient`. `semantic_search.py` and `rag.py` now build an
  `AtlasVectorRetriever` internally but keep their original function signatures and
  return types — no callers changed.
- **Prompt templates**: `langchain_core.prompts.PromptTemplate` builds the text for the
  extraction input, the RAG generation prompt, and the translate-back instructions in
  `workflow.py`/`rag.py`. The actual model calls are still `AsyncOpenAI().responses
  .create(...)` with strict `json_schema` structured outputs — deliberately not switched
  to `langchain_openai.ChatOpenAI`, since that call pattern was only just tuned to fix
  an intent-classification reliability bug and isn't worth re-risking.
- **Embeddings**: `langchain_openai.OpenAIEmbeddings` (async `aembed_query`/
  `aembed_documents`) replaced the manual `client.embeddings.create(...)` calls
  everywhere embeddings are generated (retrievers, both ingestion scripts).

**Dependency note:** `langchain-openai` requires `openai>=1.109.1` — bumped from the
previously pinned `openai==1.99.9` (both are pre-2.0 releases; the `openai` 2.x line
requires `langchain-openai>=1.2`, which was avoided specifically to not risk the tuned
`responses.create`/structured-output call sites on a major SDK version).

## MCP

`agri-assistant-ai/mcp_server.py` is a standalone MCP server (official `mcp` SDK,
pinned `mcp<2` for the stable v1 line — a `2.0` line is still alpha and renames
`FastMCP`→`MCPServer`) that exposes exactly the two tools named in this roadmap row:
`get_weather_forecast(location)` and `get_crop_price(crop, location)`. It reuses
`app/weather.py`/`app/crop_price.py` directly — no logic duplicated — via
`format_forecast`/`format_price` helper functions that were extracted out of
`workflow.py`'s private formatting code so both entrypoints share them.

**Deliberately a separate process, not mounted into the FastAPI app.** The MCP Python
SDK's ASGI-mount path had real, currently-open upstream bugs (redirect loops,
session-init errors) as of 2026-07. Running standalone via `mcp.run(transport="stdio")`
is the reliable path, and is also exactly how Claude Desktop/Claude Code spawn local
MCP servers — no port to configure.

**Deliberately not wired into the LangGraph workflow as a tool-calling client**, either.
`workflow.py`'s `fetch_weather`/`fetch_price` nodes still call the plain Python
functions directly, unchanged. Routing tool selection through the MCP protocol
internally is closer to the separate, still-`Planned` "Agents" row below — adding a
protocol hop and new failure modes to an already-tuned path wasn't worth it just to
reuse the same two tools a different way.

## Tool calling

Both tools are implemented as graph nodes/branches on top of the LangGraph workflow.
The extraction step classifies `intent: "weather" | "price" | "general"` in the same
structured-output call that already extracts crop/location, so routing costs no extra
LLM round trip.

1. **Weather forecast tool** — `agri-assistant-ai/app/weather.py` geocodes the
   extracted location and fetches a 5-day forecast from Open-Meteo (no API key needed).
   When `intent` is `weather` and a location was given, the graph routes to a
   `fetch_weather` node before falling back to the normal semantic-search/RAG pipeline
   if the location can't be resolved or the API call fails. Response
   `source: "weather-forecast"`.
2. **Mandi price tool** — `agri-assistant-ai/app/crop_price.py` looks up the latest
   modal price for a crop via the data.gov.in Agmarknet API (resource
   `9ef84268-d588-465a-a308-a864a43d0070`), trying the extracted location as a district,
   then a state, then falling back to a nationwide lookup. When `intent` is `price` and
   a crop was given, the graph routes to a `fetch_price` node, falling back to the
   normal pipeline if no price data is found. Response `source: "price-lookup"`.
   Requires `DATA_GOV_IN_API_KEY` in `agri-assistant-ai/.env` (already set locally,
   gitignored; placeholder in `.env.example`).
   **Implementation note:** this API's server hangs indefinitely on Python's
   OpenSSL-based TLS handshake (reproduced with both `httpx` and stdlib `urllib`) while
   `curl.exe`'s Schannel TLS stack connects in under a second, so `crop_price.py` shells
   out to `curl` (present by default on Windows and Git for Windows) instead of using
   `httpx` like the other tools.

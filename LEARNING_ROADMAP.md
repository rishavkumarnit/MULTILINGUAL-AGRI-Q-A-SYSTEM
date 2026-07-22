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
| Tool calling | Weather and mandi price tools for time-sensitive questions | Implemented |
| MCP | Dedicated MCP server exposing agriculture and weather tools | Planned |
| Agents | Tool-selection policy layered above the LangGraph workflow | Planned |
| Streaming | Token streaming from AI service through Node to React | Planned |
| Evaluations and observability | Test dataset, retrieval metrics, traces, and feedback | Planned |

## Portfolio flow

`React → Node API → FastAPI → translate/extract → embedding + Atlas Vector Search → reuse or RAG → LLM → translate → React`

Each stage is intentionally separate, so you can explain its responsibility, data, and evaluation method in interviews.

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

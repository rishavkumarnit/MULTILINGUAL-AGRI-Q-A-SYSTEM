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
| Agents | Tool-selection policy layered above the LangGraph workflow | Implemented |
| Streaming | Token streaming from AI service through Node to React | Implemented |
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

**Deliberately not wired into the LangGraph workflow as a tool-calling client**, even
after the "Agents" row below shipped. `app/agent.py` calls `weather.py`/`crop_price.py`'s
plain Python functions directly, the same way `mcp_server.py` does — the two
entrypoints share the underlying functions, not a protocol connection between them.
Routing the *internal* agent through its own MCP server would add a protocol hop and
new failure modes for no benefit; MCP's job here is external exposure only.

## Agents

Tool selection is no longer a hand-coded classification step. `app/agent.py`'s
`run_agent(question_english, crop, location)` gives the model three tools via OpenAI's
native Responses API function-calling and lets it decide which to call — none, one, or
several — instead of us extracting an `intent` field and branching on it:

- `get_weather_forecast(location)` → `weather.fetch_weather_forecast` + `format_forecast`
- `get_crop_price(crop, location)` → `crop_price.fetch_crop_price` + `format_price`
- `search_documents(query)` → `rag.find_relevant_chunks` (same threshold-filtered
  retrieval RAG already used)

The loop: call the model with `tools=[...]` → for each `function_call` item in
`response.output`, execute the real Python function → continue with
`client.responses.create(..., previous_response_id=response.id, input=[{"type":
"function_call_output", ...}], tools=[...])` → repeat (capped at 4 rounds) until the
model stops calling tools. `previous_response_id` handles conversation/reasoning
continuity server-side, so prior turns don't need to be replayed manually.

**Grounding is enforced in code, not just prompted.** The agent's system instructions
say to only state facts backed by a tool result, but that alone isn't a guarantee — so
`run_agent` tracks which tools actually returned real data (a forecast, a price, or any
document chunks) and derives `source` from that (`"weather-forecast"` /
`"price-lookup"` / `"rag-generated"`, reusing the exact same `ChatResponse` literal
values as before — no schema change). If no tool produced anything usable, the model's
own text is discarded entirely and `workflow.py`'s `_agent` node falls back to the exact
same plain status message the old `status_message` node used
(`source: "translation-extraction"`) — the app never presents ungrounded, hallucinated
agricultural advice as if it were sourced.

Verified-answer reuse (`semantic_search.py`) stays a **deterministic pre-check before
the agent runs**, not a tool — expert-reviewed content should be authoritative when it
matches crop+location, not left to model discretion.

This simplified `workflow.py`'s graph from 4 conditional branches (weather/price/RAG/
status, each hand-routed from a classified `intent`) down to 2 (verified-answer match,
then the agent) while being strictly more capable — the agent can combine tools within
a single answer (e.g. weather + irrigation advice) instead of being locked into exactly
one branch per question. One accepted limitation: `ChatResponse.source` is still a
single value, so if multiple tools contribute to one answer, only one "wins" for
attribution (weather > price > rag priority) — a pre-existing schema constraint, not
something this change needed to fix.

*(Since Streaming below shipped, translation moved out of the graph entirely — see that
section — so the graph is now just `translate_extract → semantic_search →(cond)→ END |
agent → END`, 1 conditional branch.)*

## Tool calling

The weather and mandi price lookups themselves (used by both the agent above and the
standalone MCP server) are plain async functions with no LLM involved in the lookup
itself:

1. **Weather forecast** — `agri-assistant-ai/app/weather.py` geocodes a location and
   fetches a 5-day forecast from Open-Meteo (no API key needed).
2. **Mandi price** — `agri-assistant-ai/app/crop_price.py` looks up the latest modal
   price for a crop via the data.gov.in Agmarknet API (resource
   `9ef84268-d588-465a-a308-a864a43d0070`), trying the location as a district, then a
   state, then falling back to a nationwide lookup. Requires `DATA_GOV_IN_API_KEY` in
   `agri-assistant-ai/.env` (already set locally, gitignored; placeholder in
   `.env.example`).
   **Implementation note:** this API's server hangs indefinitely on Python's
   OpenSSL-based TLS handshake (reproduced with both `httpx` and stdlib `urllib`) while
   `curl.exe`'s Schannel TLS stack connects in under a second, so `crop_price.py` shells
   out to `curl` (present by default on Windows and Git for Windows) instead of using
   `httpx` like the other tools.

## Streaming

Scope, confirmed with the user via two clarifying questions: stream only the **final
answer text** (the multi-step `translate_extract`/`semantic_search`/`agent` pipeline
stays invisible, exactly as before — no progress events for tool calls), and **keep
English free** — English answers still return as a single immediate chunk with zero
extra LLM cost or latency, exactly as before this change. Only non-English answers get
a real token-by-token streamed translation.

**Why translation had to leave the LangGraph graph.** `_translate_back` used to be the
graph's last node, reached via one `ainvoke()` call that only returns once everything is
done — incompatible with token-level streaming out of the endpoint. It's now two
module-level functions in `workflow.py`: `translate_answer(...)` (awaited in full, used
by the unchanged `/internal/chat` endpoint) and `translate_answer_stream(...)` (an async
generator yielding text deltas via `client.responses.create(..., stream=True)`, used by
the new streaming endpoint). Verified live during planning: `stream=True` yields
`response.output_text.delta` events with a `.delta` string each — confirmed 512 delta
events for one real Hindi translation, vs. exactly 1 for the English fast path (whole
text, no model call) and exactly 1 for the never-translate-status-messages case
(preserved unchanged from before this phase).

**Three-service plumbing**, all newly added, none of the existing endpoints removed:
- `agri-assistant-ai/app/main.py`: `POST /internal/chat/stream` — `StreamingResponse`
  (`text/event-stream`) wrapping `workflow.stream_question(...)`, which yields
  `{"type": "metadata", ...}` once (crop/location/similarity/source/sources — same
  fields as `ChatResponse`), then one or more `{"type": "delta", "text": ...}`, then
  `{"type": "done"}`.
- `agri-assistant-backend/src/server.ts`: `POST /api/chat/stream` — persists the user
  message as before, then relays FastAPI's SSE stream to the browser chunk-by-chunk
  (buffering only to find `\n\n`-delimited SSE frames, never buffering the whole
  response), injecting `conversationId` into the metadata event and accumulating the
  full answer text so it can call the existing `persistMessage(...)` for the assistant
  role once the stream ends — conversation persistence behavior is unchanged.
- `agri-assistant-frontend/src/App.tsx`: reads `response.body.getReader()` manually
  (not `EventSource`, which can't send a POST body), pushes an empty assistant message
  on the metadata event, and appends each delta's text to it — the answer visibly types
  in for non-English questions, appears instantly for English/status-message ones.

Verified end-to-end through the real browser UI, not just direct script calls: a Hindi
RAG question's answer visibly grew character-by-character; an English ungrounded
question (which is also a status message, so double-covers both fast-path rules)
appeared instantly with no typing effect; Mongo persistence confirmed correct for both
the user and assistant messages after a streamed exchange.

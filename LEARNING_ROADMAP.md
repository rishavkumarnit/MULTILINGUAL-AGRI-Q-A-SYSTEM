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
| LangGraph | Explicit state graph for the assistant workflow | Planned |
| Tool calling | Weather and location tools for time-sensitive questions | Planned |
| MCP | Dedicated MCP server exposing agriculture and weather tools | Planned |
| Agents | Tool-selection policy layered above the LangGraph workflow | Planned |
| Streaming | Token streaming from AI service through Node to React | Planned |
| Evaluations and observability | Test dataset, retrieval metrics, traces, and feedback | Planned |

## Portfolio flow

`React → Node API → FastAPI → translate/extract → embedding + Atlas Vector Search → reuse or RAG → LLM → translate → React`

Each stage is intentionally separate, so you can explain its responsibility, data, and evaluation method in interviews.

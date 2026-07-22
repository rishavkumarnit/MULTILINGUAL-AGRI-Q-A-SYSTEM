"""LangChain retriever wrapping this app's MongoDB Atlas $vectorSearch collections.

Deliberately not using langchain-mongodb's MongoDBAtlasVectorSearch: it requires a
synchronous pymongo.Collection, with no working support for the AsyncMongoClient this
app already uses everywhere else (database.py). Implementing the retriever's async
extension point directly keeps a single async Mongo connection and the same
{"$meta": "vectorSearchScore"} metric the app's score thresholds are already tuned to.
"""

from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForRetrieverRun, CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_openai import OpenAIEmbeddings
from pydantic import ConfigDict

from .database import get_database


class AtlasVectorRetriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    collection_name: str
    vector_index: str
    embedding: OpenAIEmbeddings
    top_k: int
    content_field: str
    metadata_fields: list[str] = []
    extra_filter: dict[str, Any] | None = None

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[Document]:
        database = get_database()
        if database is None:
            return []

        query_vector = await self.embedding.aembed_query(query)
        vector_search: dict[str, Any] = {
            "index": self.vector_index,
            "path": "embedding",
            "queryVector": query_vector,
            "numCandidates": 100,
            "limit": self.top_k,
        }
        if self.extra_filter:
            vector_search["filter"] = self.extra_filter

        projection = {field: 1 for field in [self.content_field, *self.metadata_fields]}
        pipeline = [
            {"$vectorSearch": vector_search},
            {"$project": {**projection, "score": {"$meta": "vectorSearchScore"}}},
        ]
        cursor = await database[self.collection_name].aggregate(pipeline)
        documents = []
        async for doc in cursor:
            metadata = {field: doc[field] for field in self.metadata_fields} | {"score": float(doc["score"])}
            documents.append(Document(page_content=doc[self.content_field], metadata=metadata))
        return documents

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        raise NotImplementedError("AtlasVectorRetriever is async-only; use ainvoke().")

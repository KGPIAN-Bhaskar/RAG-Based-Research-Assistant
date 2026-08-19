"""
ChromaDB Vector Store Manager wrapping EphemeralClient and Gemini Embedding Function.
"""

from typing import List, Dict, Any, Optional
import chromadb
from embedding import GeminiEmbeddingFunction


class VectorDBManager:
    """
    Manages in-memory ChromaDB vector store collections and vector query operations.
    """
    def __init__(self, api_key: str, model_name: str = "gemini-embedding-2") -> None:
        self.api_key = api_key
        self.model_name = model_name
        self.client = chromadb.EphemeralClient()
        self.emb_fn = GeminiEmbeddingFunction(api_key=self.api_key, model_name=self.model_name)
        self.collection: Optional[Any] = None

    def get_or_create_collection(self, collection_name: str) -> Any:
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.emb_fn
        )
        return self.collection

    def add_documents(
        self, documents: List[str], metadatas: List[Dict[str, Any]], ids: List[str]
    ) -> None:
        if self.collection:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )

    def query_similarity(self, query_text: str, k: int = 5) -> Optional[Any]:
        if self.collection:
            return self.collection.query(
                query_texts=[query_text],
                n_results=k
            )
        return None

    def get_documents_by_metadata(
        self, where_clause: Dict[str, Any], limit: Optional[int] = None
    ) -> Optional[Any]:
        if self.collection:
            if limit:
                return self.collection.get(where=where_clause, limit=limit)
            return self.collection.get(where=where_clause)
        return None

    def get_all_documents(self, limit: int = 15) -> Optional[Any]:
        if self.collection:
            return self.collection.get(limit=limit)
        return None

    def reset_db(self) -> None:
        self.collection = None

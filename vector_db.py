import chromadb
from embedding import GeminiEmbeddingFunction

class VectorDBManager:
    def __init__(self, api_key: str, model_name: str = "gemini-embedding-2"):
        self.api_key = api_key
        self.model_name = model_name
        # Initialize EphemeralClient (in-memory for serverless runs)
        self.client = chromadb.EphemeralClient()
        self.emb_fn = GeminiEmbeddingFunction(api_key=self.api_key, model_name=self.model_name)
        self.collection = None

    def get_or_create_collection(self, collection_name: str):
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.emb_fn
        )
        return self.collection

    def add_documents(self, documents: list[str], metadatas: list[dict], ids: list[str]):
        if self.collection:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )

    def query_similarity(self, query_text: str, k: int = 5):
        if self.collection:
            return self.collection.query(
                query_texts=[query_text],
                n_results=k
            )
        return None

    def get_documents_by_metadata(self, where_clause: dict, limit: int = None):
        if self.collection:
            if limit:
                return self.collection.get(where=where_clause, limit=limit)
            return self.collection.get(where=where_clause)
        return None

    def get_all_documents(self, limit: int = 15):
        if self.collection:
            return self.collection.get(limit=limit)
        return None

    def reset_db(self):
        self.collection = None

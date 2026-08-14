from chromadb.api.types import Documents, Embeddings, EmbeddingFunction
from google import genai
from concurrent.futures import ThreadPoolExecutor

class GeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(self, api_key: str, model_name: str = "gemini-embedding-2"):
        self.api_key = api_key
        self.model_name = model_name
        self.client = genai.Client(api_key=self.api_key)

    def __call__(self, input: Documents) -> Embeddings:
        try:
            # We define an embedding worker to run concurrently
            def embed_one(text: str):
                res = self.client.models.embed_content(
                    model=self.model_name,
                    contents=text
                )
                return res.embeddings[0].values

            # Ingest chunks concurrently to speed up the process significantly
            with ThreadPoolExecutor(max_workers=10) as executor:
                embeddings = list(executor.map(embed_one, input))
            return embeddings
        except Exception as e:
            raise e

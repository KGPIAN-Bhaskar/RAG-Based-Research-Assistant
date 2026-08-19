"""
Custom Gemini Embedding Function for ChromaDB with Rate-Limit Handling and Thread Concurrency.
"""

import time
import random
import logging
from typing import List
from chromadb.api.types import Documents, Embeddings, EmbeddingFunction
from google import genai
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class GeminiEmbeddingFunction(EmbeddingFunction):
    """
    ChromaDB compatible Embedding Function using Google Gemini Embedding API.
    Includes exponential backoff retries for 429 Rate Limits and concurrency regulation.
    """
    def __init__(self, api_key: str, model_name: str = "gemini-embedding-2") -> None:
        self.api_key = api_key
        self.model_name = model_name
        self.client = genai.Client(api_key=self.api_key)
        self.last_call_duration: float = 0.0

    def __call__(self, input: Documents) -> Embeddings:
        start_time = time.perf_counter()
        try:
            def embed_one(text: str) -> List[float]:
                max_retries = 6
                base_delay = 1.5
                for attempt in range(max_retries):
                    try:
                        res = self.client.models.embed_content(
                            model=self.model_name,
                            contents=text
                        )
                        return list(res.embeddings[0].values)
                    except Exception as e:
                        err_str = str(e).lower()
                        if (
                            "429" in err_str
                            or "resource_exhausted" in err_str
                            or "quota" in err_str
                            or "rate limit" in err_str
                        ) and attempt < max_retries - 1:
                            sleep_time = (base_delay * (2 ** attempt)) + random.uniform(0.1, 0.5)
                            logger.warning(
                                f"Gemini API rate limit hit (429). Retrying in {sleep_time:.2f}s... "
                                f"(Attempt {attempt + 1}/{max_retries})"
                            )
                            time.sleep(sleep_time)
                        else:
                            raise e

            if len(input) == 1:
                return [embed_one(input[0])]

            # Regulate worker concurrency (max 3 workers) to stay within Gemini Free Tier limits
            max_workers = min(3, len(input))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                embeddings = list(executor.map(embed_one, input))
            return embeddings

        finally:
            self.last_call_duration = time.perf_counter() - start_time

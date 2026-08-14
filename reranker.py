import logging
from typing import List, Dict, Tuple
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

class Reranker:
    """
    Cross-Encoder based Reranker stage for RAG pipelines.
    Evaluates Query ↔ Chunk relevance to select the Top-P most relevant candidates
    out of Top-K vector search results.
    """
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model_name = model_name
        self.model = None

    def _load_model(self):
        if self.model is None:
            logger.info(f"Loading CrossEncoder model: {self.model_name}...")
            self.model = CrossEncoder(self.model_name)

    def rerank(
        self, query: str, documents: List[str], metadatas: List[Dict], top_p: int = 3
    ) -> Tuple[List[str], List[Dict], List[float]]:
        """
        Reranks retrieved candidate chunks based on deep query-document cross-encoding.
        
        Args:
            query: User input query text.
            documents: List of text chunks retrieved from Top-K vector search.
            metadatas: List of metadata dicts corresponding to each text chunk.
            top_p: Desired number of top reranked chunks to return (P <= K).
            
        Returns:
            Tuple of (reranked_documents, reranked_metadatas, relevance_scores)
        """
        if not documents:
            return [], [], []

        # Ensure top_p does not exceed the available candidate count
        effective_p = min(top_p, len(documents))

        try:
            self._load_model()
            pairs = [[query, doc] for doc in documents]
            scores = self.model.predict(pairs)

            # Zip scores, documents, and metadatas together and sort descending by score
            scored_tuples = sorted(
                zip(scores, documents, metadatas),
                key=lambda x: float(x[0]),
                reverse=True
            )

            reranked_scores = [float(s) for s, _, _ in scored_tuples[:effective_p]]
            reranked_docs = [doc for _, doc, _ in scored_tuples[:effective_p]]
            reranked_metadatas = [meta for _, _, meta in scored_tuples[:effective_p]]

            return reranked_docs, reranked_metadatas, reranked_scores

        except Exception as e:
            logger.error(f"Reranker failed with error: {e}. Falling back to standard vector search ordering.")
            # Graceful Fallback: Return top_p items directly from candidate list without breaking execution
            return documents[:effective_p], metadatas[:effective_p], []

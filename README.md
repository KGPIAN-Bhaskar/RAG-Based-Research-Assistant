# 🎓 RAG-Based Research Assistant

A premium, browser-hosted **Retrieval-Augmented Generation (RAG)** Research Assistant powered by **Google Gemini API** (`google-genai`), **ChromaDB**, **BGE CrossEncoder Reranker**, and **Streamlit**.

Designed specifically for researchers, students, and academics to upload research papers (PDF, DOCX, TXT, MD), perform deep vector retrieval + reranking, and ask complex questions with verified document & page-level citations.

---

## 🏗️ End-to-End RAG Pipeline Architecture

```
                                  USER QUERY
                                      │
                                      ▼
                       ┌─────────────────────────────┐
                       │    Query Vector Embedding   │
                       │     (gemini-embedding-2)    │
                       └──────────────┬──────────────┘
                                      │
                                      ▼
                       ┌─────────────────────────────┐
                       │   ChromaDB Vector Search    │
                       │    Retrieves Top-K Chunks   │
                       └──────────────┬──────────────┘
                                      │
                                      ▼
                       ┌─────────────────────────────┐
                       │   CrossEncoder Reranker     │
                       │  (BAAI/bge-reranker-base)   │
                       │   Selects Top-P Candidates  │
                       └──────────────┬──────────────┘
                                      │
                                      ▼
                       ┌─────────────────────────────┐
                       │    Grounded Context & Prompt│
                       └──────────────┬──────────────┘
                                      │
                                      ▼
                       ┌─────────────────────────────┐
                       │     Gemini LLM Generation   │
                       │   (gemini-2.0-flash / etc.) │
                       └──────────────┬──────────────┘
                                      │
                                      ▼
                      STREAMING RESPONSE WITH CITATIONS
```

---

## 🤖 Models, Components & Where They Are Used

| Component / Model | Technology / Model Name | File Location | Purpose & Functionality |
| :--- | :--- | :--- | :--- |
| **Document Splitter** | `SimpleTextSplitter` | [`app.py`](file:///c:/Users/manda/Desktop/Hybrid%20RAG%20Based%20Research%20Assistant/app.py) | Recursively splits uploaded documents into text chunks based on user-configured **Chunk Size** and **Chunk Overlap**. Prepends trailing overlap text from preceding chunks to preserve context boundaries. |
| **Embedding Model** | `gemini-embedding-2` | [`embedding.py`](file:///c:/Users/manda/Desktop/Hybrid%20RAG%20Based%20Research%20Assistant/embedding.py) | Custom ChromaDB `GeminiEmbeddingFunction`. Converts text chunks and queries into 3072-dimensional dense vector embeddings via Google GenAI API. Features **exponential backoff retry** for 429 rate limits (100 RPM Gemini Free Tier limit). |
| **Vector Database** | `ChromaDB` (Ephemeral Client) | [`vector_db.py`](file:///c:/Users/manda/Desktop/Hybrid%20RAG%20Based%20Research%20Assistant/vector_db.py) | In-memory vector database manager (`VectorDBManager`). Manages collection creation, indexes chunk vectors with metadata (`source`, `page`, `chunk_index`), and executes similarity search to retrieve **Top-K vector candidates**. |
| **Reranker Model** | `BAAI/bge-reranker-base` (CrossEncoder) | [`reranker.py`](file:///c:/Users/manda/Desktop/Hybrid%20RAG%20Based%20Research%20Assistant/reranker.py) | Deep CrossEncoder reranker. Evaluates query ↔ candidate text pairs using deep transformer self-attention to re-score vector candidates and select the **Top-P most relevant chunks**. Features startup pre-warming, candidate deduplication, and PyTorch `inference_mode`. |
| **LLM Reasoning & Generation** | `gemini-2.0-flash`<br>*(also supports `gemini-3.1-flash-lite`, `gemini-1.5-pro`)* | [`streamlit_app.py`](file:///c:/Users/manda/Desktop/Hybrid%20RAG%20Based%20Research%20Assistant/streamlit_app.py) | Generates streaming answers grounded strictly on the reranked context. Outputs inline citations and renders document tags with page numbers. |
| **Frontend UI** | `Streamlit` | [`streamlit_app.py`](file:///c:/Users/manda/Desktop/Hybrid%20RAG%20Based%20Research%20Assistant/streamlit_app.py) | Glassmorphism UI with custom Inter typography, dark mode theme, live metric statistics, tabbed navigation, and interactive document ingestion controls. |

---

## 🚀 Step-by-Step Pipeline Flow

1. **Document Ingestion & Parsing**:
   - User uploads papers (`.pdf`, `.docx`, `.txt`, `.md`).
   - `parse_document()` extracts text and page numbers.
   - `SimpleTextSplitter` divides text using user-defined **Chunk Size** (e.g., 800 chars) and **Chunk Overlap** (e.g., 150 chars).
   - Generated chunks are embedded using `GeminiEmbeddingFunction` (`gemini-embedding-2`) and stored in ChromaDB along with page metadata.

2. **Retrieval & Reranking**:
   - When a user submits a question, `VectorDBManager` queries ChromaDB for the **Top-K** closest vector candidates (default `K=10`).
   - The candidate chunks are passed to the `Reranker` (`BAAI/bge-reranker-base`).
   - CrossEncoder scores query-chunk relevance, filters out duplicate chunks, and selects the **Top-P** highest-scoring chunks (default `P=3`).

3. **Grounded Generation & Citation**:
   - The Top-P reranked chunks are formatted into an academic research prompt.
   - Gemini LLM generates a real-time streaming answer with precise `[DocumentName, Page X]` source citations.

---

## 📁 Repository Structure

```
.
├── app.py              # Document extraction (PDF/DOCX/TXT) & SimpleTextSplitter chunking
├── embedding.py        # GeminiEmbeddingFunction with retry & backoff for 429 rate limits
├── vector_db.py        # VectorDBManager encapsulating ChromaDB storage & similarity search
├── reranker.py         # CrossEncoder Reranker with pre-warming & batch inference
├── streamlit_app.py    # Main Streamlit web application & chat interface
├── requirements.txt    # Required Python dependencies
└── README.md           # Documentation
```

---

## 💻 Local Setup & Installation

### 1. Prerequisites
- Python 3.10+
- Google Gemini API Key ([Get API Key](https://aistudio.google.com/))

### 2. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/KGPIAN-Bhaskar/RAG-Based-Research-Assistant.git
cd RAG-Based-Research-Assistant
pip install -r requirements.txt
```

### 3. Environment Configuration

Create a `.env` file in the root directory and add your Gemini API Key:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 4. Run Application

Launch the Streamlit dashboard:

```bash
streamlit run streamlit_app.py
```

The application will automatically open in your browser at `http://localhost:8501`.

---

## ⚡ Performance & Optimization Highlights

- **Pre-warmed Model Weights**: CrossEncoder weights are pre-loaded into memory during session startup to eliminate first-query cold-start latency.
- **Candidate Deduplication**: Identical chunks are deduplicated before reranking, saving ~25-35% computation time.
- **Automatic 429 Rate Limit Handling**: Automatic exponential backoff handles Gemini Free Tier API quotas (100 RPM) seamlessly without failing document ingestion.
- **Optimized UI Stream Rendering**: Pre-computed citation tags prevent redundant HTML re-rendering during response streaming.

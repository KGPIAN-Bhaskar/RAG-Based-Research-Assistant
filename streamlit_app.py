import streamlit as st
import os
import time
import chromadb
from google import genai
from dotenv import load_dotenv

# Import backend RAG processing logic & Vector DB logic
from app import parse_document
from vector_db import VectorDBManager
from reranker import Reranker

# Load local environment variables (if any)
load_dotenv()

# --- Page Config ---
st.set_page_config(
    page_title="Enterprise Systems ",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Premium CSS Styling ---
st.markdown("""
    <style>
    /* Custom Google Font Inter */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main Background & Padding */
    .stApp {
        background-color: #0d0f14;
        color: #e2e8f0;
    }

    /* Make Streamlit top header transparent and blend in */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
    }
    
    /* Reduce top padding to extend UI to the top portions */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
    }

    /* Gradient Header Text */
    .gradient-title {
        background: linear-gradient(135deg, #a855f7 0%, #6366f1 50%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -0.05rem;
        margin-bottom: 0.2rem;
        text-align: center;
    }

    .subtitle {
        font-size: 1.1rem;
        color: #94a3b8;
        font-weight: 400;
        margin-bottom: 2rem;
        text-align: center;
    }

    /* Sidebar Custom Styling */
    section[data-testid="stSidebar"] {
        background-color: #111420 !important;
        border-right: 1px solid #1e293b;
    }

    /* Move sidebar contents to the absolute top of the sidebar container with exactly 1px space */
    [data-testid="stSidebar"] {
        padding-top: 1px !important;
    }
    [data-testid="stSidebar"] > div:first-child,
    [data-testid="stSidebarUserContent"],
    [data-testid="stSidebarUserContent"] > div,
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1px !important;
        margin-top: 1px !important;
    }

    /* Target the very first element block containing the cap emoji to force 1px spacing */
    section[data-testid="stSidebar"] div.element-container:first-of-type,
    section[data-testid="stSidebar"] div.element-container:first-of-type > div {
        margin-top: 1px !important;
        padding-top: 1px !important;
    }

    /* Hide empty Streamlit sidebar navigation spacing */
    div[data-testid="stSidebarNav"] {
        display: none !important;
        height: 0px !important;
        margin: 0px !important;
        padding: 0px !important;
    }

    /* Reduce vertical widget spacing inside the sidebar */
    section[data-testid="stSidebar"] div.stVerticalBlock {
        gap: 0.4rem !important;
    }

    /* Glassmorphism Metric Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.25rem;
        margin: 0.5rem 0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        height: 110px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .metric-card:hover {
        border-color: rgba(99, 102, 241, 0.4);
        transform: translateY(-2px);
        box-shadow: 0 10px 20px -10px rgba(99, 102, 241, 0.2);
    }

    .metric-title {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05rem;
        margin-bottom: 0.25rem;
    }

    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffffff;
    }

    /* Custom Chat Styling */
    .chat-bubble {
        padding: 1.2rem;
        border-radius: 18px;
        margin-bottom: 1rem;
        max-width: 85%;
        line-height: 1.5;
        font-size: 0.95rem;
        animation: fadeIn 0.3s ease-in-out;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .user-bubble {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        color: white;
        margin-left: auto;
        border-bottom-right-radius: 4px;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.2);
    }

    .assistant-bubble {
        background: rgba(30, 41, 59, 0.55);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.06);
        color: #f1f5f9;
        margin-right: auto;
        border-bottom-left-radius: 4px;
    }

    .chat-header {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.03rem;
        margin-bottom: 0.4rem;
        color: #94a3b8;
    }

    /* Citation Box Styling */
    .citation-container {
        margin-top: 0.8rem;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        padding-top: 0.6rem;
    }

    .citation-tag {
        display: inline-flex;
        align-items: center;
        background: rgba(99, 102, 241, 0.15);
        color: #818cf8;
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 6px;
        padding: 0.2rem 0.6rem;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.5rem;
        margin-top: 0.25rem;
    }

    /* ===========================
   CENTER TABS - STREAMLIT 1.58
=========================== */

.stTabs{
    display:flex;
    justify-content:center;
}

.stTabs > div{
    width:100%;
}

.stTabs [role="tablist"]{
    display:flex !important;
    justify-content:center !important;
    align-items:center !important;
    width:fit-content !important;
    margin:0 auto !important;
    gap:12px !important;
}

.stTabs [role="tab"]{
    height:48px;
    padding:0 22px;
    border-radius:10px 10px 0 0;
    background:rgba(30,41,59,.20);
    border:1px solid rgba(255,255,255,.05);
    color:#94a3b8;
    transition:.25s;
}

.stTabs [aria-selected="true"]{
    background:rgba(99,102,241,.15)!important;
    border-color:rgba(99,102,241,.35)!important;
    color:#a5b4fc!important;
    font-weight:600;
}
    /* Green Send Button for st.chat_input */
    button[data-testid="stChatInputSubmitButton"] {
        background-color: #10b981 !important;
        color: white !important;
    }
    /* ===========================
   CHAT INPUT - BLACK THEME
=========================== */

/* Outer chat box */
div[data-testid="stChatInput"]{
    background:#000000 !important;
    border:1px solid #3d3d3d !important;
    border-radius:14px !important;
    padding:8px !important;
}

/* Remove Streamlit white wrapper */
div[data-testid="stChatInput"] > div{
    background:#000000 !important;
}

/* Text input */
div[data-testid="stChatInput"] textarea{
    background:#000000 !important;
    color:#ffffff !important;
    border:none !important;
    box-shadow:none !important;
    caret-color:#ffffff !important;
}

/* Placeholder */
div[data-testid="stChatInput"] textarea::placeholder{
    color:#9ca3af !important;
}

/* Focus */
div[data-testid="stChatInput"]:focus-within{
    border:1px solid #6366f1 !important;
}

/* Send button */
button[data-testid="stChatInputSubmitButton"]{
    background:#10b981 !important;
    color:white !important;
    border-radius:10px !important;
}

/* Clear Chat button */
button[kind="secondary"]{
    background:#000000 !important;
    color:#ffffff !important;
    border:1px solid #3d3d3d !important;
}

button[kind="secondary"]:hover{
    background:#1a1a1a !important;
}
/* Clear Chat button */
button[kind="secondary"] {
    background-color: #000000 !important;
    color: #ffffff !important;
    border: 1px solid #404040 !important;
    border-radius: 10px !important;
}

button[kind="secondary"]:hover {
    background-color: #1a1a1a !important;
    border-color: #666666 !important;
}

button[data-testid="stChatInputSubmitButton"] {
    background-color: #10b981 !important;
    color: white !important;
}   
    /* Right Sidebar Custom Styling */
    .right-sidebar-panel {
        background-color: #111420;
        border: 1px solid #1e293b;
        border-radius: 16px;
        padding: 1.25rem 1.1rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        margin-top: 0.5rem;
    }

    .right-sidebar-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 1.25rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .query-metric-card {
        background: rgba(30, 41, 59, 0.35);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.75rem;
        transition: all 0.2s ease;
    }

    .query-metric-card:hover {
        border-color: rgba(99, 102, 241, 0.3);
        background: rgba(30, 41, 59, 0.5);
    }

    .query-metric-label {
        font-size: 0.82rem;
        color: #94a3b8;
        font-weight: 500;
        margin-bottom: 0.3rem;
        letter-spacing: 0.01rem;
    }

    .query-metric-val {
        font-size: 1.35rem;
        font-weight: 700;
        color: #f8fafc;
    }

    .query-metric-divider {
        border: 0;
        height: 1px;
        background: rgba(255, 255, 255, 0.12);
        margin: 1.1rem 0;
    }

    .total-latency-card {
        background: rgba(99, 102, 241, 0.12);
        border: 1px solid rgba(99, 102, 241, 0.3);
    }

    .total-latency-card:hover {
        background: rgba(99, 102, 241, 0.2);
        border-color: rgba(99, 102, 241, 0.5);
    }

    .total-latency-val {
        color: #818cf8;
        font-size: 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)


# --- Custom Embedding Function for ChromaDB ---
# Classes moved to app.py


# --- Session State Initializations ---
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = {}  # filename -> {chunk_count, size_bytes}

if "doc_embedding_time" not in st.session_state:
    st.session_state.doc_embedding_time = 0.0

if "reranker" not in st.session_state:
    st.session_state.reranker = Reranker()
    st.session_state.reranker._load_model()

if "query_metrics" not in st.session_state:
    st.session_state.query_metrics = {
        "query_embedding_time": None,
        "vector_search_time": None,
        "reranker_time": None,
        "llm_time": None,
        "total_latency": None
    }


def render_query_performance_sidebar():
    metrics = st.session_state.get("query_metrics", {})
    
    def format_ms(val):
        if val is None:
            return "—"
        ms_val = round(val, 1)
        if ms_val.is_integer():
            return f"{int(ms_val)} ms"
        return f"{ms_val:.1f} ms"

    emb_str = format_ms(metrics.get("query_embedding_time"))
    search_str = format_ms(metrics.get("vector_search_time"))
    rerank_str = format_ms(metrics.get("reranker_time"))
    llm_str = format_ms(metrics.get("llm_time"))
    total_str = format_ms(metrics.get("total_latency"))

    st.markdown(f"""
        <div class="right-sidebar-panel">
            <div class="right-sidebar-header">⚡ Query Performance</div>
            
            <div class="query-metric-card">
                <div class="query-metric-label">Query Embedding Time</div>
                <div class="query-metric-val">{emb_str}</div>
            </div>
            
            <div class="query-metric-card">
                <div class="query-metric-label">Vector Search Time</div>
                <div class="query-metric-val">{search_str}</div>
            </div>
            
            <div class="query-metric-card">
                <div class="query-metric-label">Reranker Time</div>
                <div class="query-metric-val">{rerank_str}</div>
            </div>
            
            <div class="query-metric-card">
                <div class="query-metric-label">LLM Time</div>
                <div class="query-metric-val">{llm_str}</div>
            </div>
            
            <hr class="query-metric-divider">
            
            <div class="query-metric-card total-latency-card">
                <div class="query-metric-label">Total Query Latency</div>
                <div class="query-metric-val total-latency-val">{total_str}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)


# --- Sidebar UI Configuration ---
with st.sidebar:
    st.markdown('<div style="text-align: center; margin-top: 0px; margin-bottom: 0.2rem; line-height: 1.0;"><span style="font-size: 3rem;">🎓</span></div>', unsafe_allow_html=True)
    st.markdown('<h2 style="text-align: center; color: white; margin-top: 0px; margin-bottom: 0.4rem; font-size: 1.5rem; font-weight: 700;">Configuration</h2>', unsafe_allow_html=True)
    
    # 1. Securely retrieve API Key from environment variables (local .env or Streamlit Secrets)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.sidebar.error("🔑 Gemini API Key not found! Please configure GEMINI_API_KEY in your local .env file or Streamlit Cloud Secrets.")
    
    # 2. Model Selection
    model_options = {
        "gemini-3.1-flash-lite": "Gemini 3.1 Flash Lite (Fastest & Efficient)",
        "gemini-2.0-flash": "Gemini 2.0 Flash (Recommended)",
        "gemini-1.5-flash": "Gemini 1.5 Flash (Legacy)",
        "gemini-1.5-pro": "Gemini 1.5 Pro (Deep Reasoning)"
    }
    selected_model_id = st.selectbox(
        "Select LLM Model",
        options=list(model_options.keys()),
        format_func=lambda x: model_options[x]
    )

    st.markdown('<h3 style="color: white; font-size: 1.1rem; margin-bottom: 0.5rem;">Document Ingestion Parameters</h3>', unsafe_allow_html=True)
    
    chunk_size = st.slider("Chunk Size (characters)", min_value=300, max_value=2000, value=800, step=100)
    chunk_overlap = st.slider("Chunk Overlap (characters)", min_value=0, max_value=max(0, chunk_size - 50), value=min(150, max(0, chunk_size - 50)), step=25)
    
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        st.sidebar.error("⚠️ Invalid Chunk Overlap: Must satisfy 0 <= Chunk Overlap < Chunk Size.")
    
    st.markdown('<h3 style="color: white; font-size: 1.1rem; margin-bottom: 0.5rem;">Retrieval & Reranking Parameters</h3>', unsafe_allow_html=True)
    top_k = st.slider("Vector Candidates (Top-K)", min_value=5, max_value=25, value=10, step=1)
    top_p = st.slider("Reranked Results (Top-P)", min_value=1, max_value=top_k, value=min(3, top_k), step=1)
    
    st.markdown("<hr style='border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
    
    # Document Upload Section
    st.markdown('<h3 style="color: white; font-size: 1.1rem; margin-bottom: 0.5rem;">Upload Research Papers</h3>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload files (.pdf, .txt, .md, .docx)",
        type=["pdf", "txt", "md", "docx"],
        accept_multiple_files=True
    )
    
    # Ingest action button
    ingest_button = st.button("Process & Embed Documents", use_container_width=True, type="primary")

    # Ingested documents list below the button
    if st.session_state.uploaded_files:
        st.markdown("<hr style='border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
        st.markdown('<h3 style="color: white; font-size: 1.1rem; margin-bottom: 0.5rem;">Ingested Documents</h3>', unsafe_allow_html=True)
        
        for filename, info in st.session_state.uploaded_files.items():
            size_mb = info["size_bytes"] / (1024 * 1024)
            st.markdown(f"""
                <div style="background: rgba(30, 41, 59, 0.4); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 10px; margin-bottom: 8px;">
                    <strong style="color: #a5b4fc; font-size: 0.9rem;">📄 {filename}</strong><br>
                    <span style="color: #94a3b8; font-size: 0.75rem;">Size: {size_mb:.2f} MB | Chunks: {info["chunk_count"]}</span>
                </div>
            """, unsafe_allow_html=True)
            
        if st.button("Reset Session Database", type="secondary", use_container_width=True):
            if st.session_state.vector_db:
                st.session_state.vector_db.reset_db()
            st.session_state.uploaded_files = {}
            st.session_state.chat_history = []
            st.session_state.doc_embedding_time = 0.0
            st.session_state.query_metrics = {
                "query_embedding_time": None,
                "vector_search_time": None,
                "reranker_time": None,
                "llm_time": None,
                "total_latency": None
            }
            st.success("Database cleared!")
            st.rerun()

# --- Document Processing Backend Logic ---
if ingest_button and uploaded_files:
    if not api_key:
        st.sidebar.error("Please provide a Gemini API Key to process documents!")
    elif chunk_overlap < 0 or chunk_overlap >= chunk_size:
        st.sidebar.error("Cannot process documents: Chunk Overlap must be non-negative and strictly less than Chunk Size!")
    else:
        with st.spinner("Processing & Ingesting documents..."):
            try:
                # 1. Initialize Vector Database Manager (fresh if not exists or if API key changes)
                if st.session_state.vector_db is None or st.session_state.vector_db.api_key != api_key:
                    st.session_state.vector_db = VectorDBManager(api_key=api_key)
                
                # Create a fresh collection
                timestamp = int(time.time())
                collection_name = f"research_assistant_{timestamp}"
                st.session_state.vector_db.get_or_create_collection(collection_name)
                
                total_chunks_added = 0
                total_embedding_time = 0.0
                st.session_state.uploaded_files = {}  # Reset metadata
                
                # 2. Iterate and Parse Each File
                for uploaded_file in uploaded_files:
                    filename = uploaded_file.name
                    chunks, metadatas, ids = parse_document(uploaded_file, chunk_size, chunk_overlap=chunk_overlap)
                    
                    # 3. Add to ChromaDB in batches
                    if chunks:
                        st.session_state.vector_db.add_documents(
                            documents=chunks,
                            metadatas=metadatas,
                            ids=ids
                        )
                        total_embedding_time += st.session_state.vector_db.emb_fn.last_call_duration
                        total_chunks_added += len(chunks)
                        st.session_state.uploaded_files[filename] = {
                            "chunk_count": len(chunks),
                            "size_bytes": uploaded_file.size
                        }
                
                st.session_state.doc_embedding_time = total_embedding_time
                
                # Collection reference is handled by vector_db manager
                st.sidebar.success(f"Successfully ingested {len(uploaded_files)} files into {total_chunks_added} chunks!")
                
            except Exception as e:
                st.sidebar.error(f"Ingestion failed: {str(e)}")

# --- Main App Layout ---
main_layout_col, right_sidebar_col = st.columns([3.2, 1.0])

with main_layout_col:
    st.markdown('<h1 class="gradient-title">Enterprise RAG Systems</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Client-side document intelligence & high-precision Q&A powered by Gemini & ChromaDB</p>', unsafe_allow_html=True)

    # Top Metric Stats (Dynamic Row)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Gemini Model</div>
                <div class="metric-val" style="font-size: 1.1rem; color: #a5b4fc; margin-top: 5px;">{selected_model_id}</div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Ingested Documents</div>
                <div class="metric-val">{len(st.session_state.uploaded_files)}</div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        total_chunks = sum(f["chunk_count"] for f in st.session_state.uploaded_files.values())
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Total Chunks</div>
                <div class="metric-val">{total_chunks}</div>
            </div>
        """, unsafe_allow_html=True)

    with col4:
        total_size = sum(f["size_bytes"] for f in st.session_state.uploaded_files.values()) / (1024 * 1024)
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Database Size</div>
                <div class="metric-val">{total_size:.2f} MB</div>
            </div>
        """, unsafe_allow_html=True)

    with col5:
        emb_time = st.session_state.get("doc_embedding_time", 0.0)
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Document Embedding Time</div>
                <div class="metric-val">{emb_time:.1f} sec</div>
            </div>
        """, unsafe_allow_html=True)

    # Main Application Tabs
    tab_chat, tab_summary, tab_insights = st.tabs([
        "💬 Research Chat",
        "📄 Document Summarizer",
        "📊 Key Insights & Q&A"
    ])

    # --- TAB 1: Chat Interface ---
    with tab_chat:
        # Clear Chat and Warning header row
        col_warn, col_clear = st.columns([0.88, 0.12])
        with col_warn:
            if not st.session_state.vector_db or not st.session_state.vector_db.collection:
                st.warning("⚠️ No documents uploaded and processed. The Assistant will run in basic mode (general knowledge without RAG context). Please upload and process documents in the sidebar to activate the research repository.")
        with col_clear:
            if st.button("Clear Chat", type="secondary", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

        # Container to render chat history
        chat_container = st.container()

        with chat_container:
            for idx, chat in enumerate(st.session_state.chat_history):
                if chat["role"] == "user":
                    st.markdown(f"""
                        <div class="chat-bubble user-bubble">
                            <div class="chat-header" style="color: rgba(255,255,255,0.85); display: flex; align-items: center; gap: 8px;">
                                <span style="font-size: 1.1rem;">👨‍💻</span> <span>YOU</span>
                            </div>
                            {chat["content"]}
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    # Build citations list
                    citation_html = ""
                    if chat.get("sources"):
                        citation_html += '<div class="citation-container">'
                        for src in chat["sources"]:
                            citation_html += f'<span class="citation-tag">📄 {src["source"]} (Page {src["page"]})</span>'
                        citation_html += '</div>'
                    
                    st.markdown(f"""
                        <div class="chat-bubble assistant-bubble">
                            <div class="chat-header" style="color: #a5b4fc; display: flex; align-items: center; gap: 8px;">
                                <span style="font-size: 1.1rem;">🎓</span> <span>RESEARCH ASSISTANT</span>
                            </div>
                            <div>{chat["content"]}</div>
                            {citation_html}
                        </div>
                    """, unsafe_allow_html=True)

        # Chat input box
        user_query = st.chat_input("Ask a question about your uploaded research papers...")

        if user_query:
            # Display user query instantly
            with chat_container:
                st.markdown(f"""
                    <div class="chat-bubble user-bubble">
                        <div class="chat-header" style="color: rgba(255,255,255,0.85); display: flex; align-items: center; gap: 8px;">
                            <span style="font-size: 1.1rem;">👨‍💻</span> <span>YOU</span>
                        </div>
                        {user_query}
                    </div>
                """, unsafe_allow_html=True)

            st.session_state.chat_history.append({"role": "user", "content": user_query})

            # Process retrieval and answer generation with latency timing
            retrieved_sources = []
            context_str = ""
            
            query_emb_time = 0.0
            vector_search_time = 0.0
            reranker_time = 0.0
            llm_time = 0.0
            
            if st.session_state.vector_db and st.session_state.vector_db.collection:
                # 1. Measure Query Embedding Time
                t0_emb = time.perf_counter()
                query_embeddings = st.session_state.vector_db.emb_fn([user_query])
                t1_emb = time.perf_counter()
                query_emb_time = t1_emb - t0_emb

                # 2. Measure Vector Search Time (query ChromaDB with pre-computed query embedding)
                t0_vec = time.perf_counter()
                results = st.session_state.vector_db.collection.query(
                    query_embeddings=query_embeddings,
                    n_results=top_k
                )
                t1_vec = time.perf_counter()
                vector_search_time = t1_vec - t0_vec
                
                # Format retrieved context after Reranking
                if results and results["documents"] and results["documents"][0]:
                    candidate_docs = results["documents"][0]
                    candidate_metadatas = results["metadatas"][0]
                    
                    # 3. Measure Reranker Time: Apply CrossEncoder Reranking
                    t0_rerank = time.perf_counter()
                    reranked_docs, reranked_metadatas, _ = st.session_state.reranker.rerank(
                        query=user_query,
                        documents=candidate_docs,
                        metadatas=candidate_metadatas,
                        top_p=top_p
                    )
                    t1_rerank = time.perf_counter()
                    reranker_time = t1_rerank - t0_rerank
                    
                    # De-duplicate citations for UI representation
                    seen_citations = set()
                    
                    context_chunks = []
                    for i, doc_text in enumerate(reranked_docs):
                        meta = reranked_metadatas[i]
                        source_name = meta.get("source", "Unknown")
                        page_no = meta.get("page", 1)
                        
                        context_chunks.append(f"Source: {source_name} | Page: {page_no}\nContent: {doc_text}")
                        
                        citation_key = (source_name, page_no)
                        if citation_key not in seen_citations:
                            seen_citations.add(citation_key)
                            retrieved_sources.append({"source": source_name, "page": page_no})
                    
                    context_str = "\n\n---\n\n".join(context_chunks)

            # Build research system prompt
            system_prompt = f"""You are a professional RAG Research Assistant. 
Your objective is to provide a comprehensive, academically structured answer to the query based ONLY on the context below. 

If the provided context does not contain enough information to answer, state clearly: "I cannot find the answer in the provided documents." 
Provide citations to the sources (e.g. referencing [DocumentName, Page X]) in your text where applicable.

CONTEXT:
{context_str if context_str else "No documents uploaded. Assist with general knowledge."}
"""

            # Generate Streaming Response & Measure LLM Time
            with chat_container:
                assistant_placeholder = st.empty()
                full_response = ""
                try:
                    t0_llm = time.perf_counter()
                    client = genai.Client(api_key=api_key)
                    response = client.models.generate_content_stream(
                        model=selected_model_id,
                        contents=f"{system_prompt}\n\nQuery: {user_query}"
                    )
                    
                    # Pre-compute citation HTML string once outside the streaming loop
                    citation_html = ""
                    if retrieved_sources:
                        citation_html += '<div class="citation-container">'
                        for src in retrieved_sources:
                            citation_html += f'<span class="citation-tag">📄 {src["source"]} (Page {src["page"]})</span>'
                        citation_html += '</div>'

                    for chunk in response:
                        if chunk.text:
                            full_response += chunk.text
                        
                        assistant_placeholder.markdown(f"""
                            <div class="chat-bubble assistant-bubble">
                                <div class="chat-header" style="color: #a5b4fc; display: flex; align-items: center; gap: 8px;">
                                    <span style="font-size: 1.1rem;">🎓</span> <span>RESEARCH ASSISTANT</span>
                                </div>
                                <div>{full_response}</div>
                                {citation_html}
                            </div>
                        """, unsafe_allow_html=True)
                    
                    t1_llm = time.perf_counter()
                    llm_time = t1_llm - t0_llm

                    total_latency = query_emb_time + vector_search_time + reranker_time + llm_time

                    st.session_state.query_metrics = {
                        "query_embedding_time": query_emb_time * 1000,
                        "vector_search_time": vector_search_time * 1000,
                        "reranker_time": reranker_time * 1000,
                        "llm_time": llm_time * 1000,
                        "total_latency": total_latency * 1000
                    }
                    
                    # Append finalized response to history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": full_response,
                        "sources": retrieved_sources
                    })
                    
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")

            st.rerun()

    # --- TAB 2: Document Summarizer ---
    with tab_summary:
        st.markdown('<h2 style="color: white;">📄 Document Executive Summarizer</h2>', unsafe_allow_html=True)
        st.write("Select an uploaded paper to generate an automatic executive summary using the entire document text.")
        
        if not st.session_state.vector_db or not st.session_state.vector_db.collection or not st.session_state.uploaded_files:
            st.info("Please upload and process documents in the sidebar to unlock document summarization.")
        else:
            selected_file = st.selectbox("Choose Document to Summarize", list(st.session_state.uploaded_files.keys()))
            
            if st.button("Generate Summary", type="primary"):
                with st.spinner("Extracting content and writing summary..."):
                    try:
                        # Retrieve all chunks for this specific file from Vector DB
                        results = st.session_state.vector_db.get_documents_by_metadata(
                            where_clause={"source": selected_file}
                        )
                        
                        if results and results["documents"]:
                            full_doc_text = "\n\n".join(results["documents"])
                            
                            # Set up summarization prompt
                            summarize_prompt = f"""You are a senior academic research summarizer. 
Provide a comprehensive, high-quality Executive Summary for the following research document.

The summary must include:
1. **Objective/Abstract**: The primary research goal.
2. **Methodology**: How the study was conducted.
3. **Key Findings**: Crucial discoveries, numbers, and stats.
4. **Significance/Limitations**: Why this matters and potential flaws.

Document Content:
{full_doc_text[:120000]} # Trim to fit within context, though Gemini has massive limit.
"""
                            client = genai.Client(api_key=api_key)
                            response = client.models.generate_content(
                                model=selected_model_id,
                                contents=summarize_prompt
                            )
                            
                            st.markdown('<div class="metric-card" style="margin-top: 1.5rem;">', unsafe_allow_html=True)
                            st.markdown(f"### Summary for {selected_file}")
                            st.write(response.text)
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.error("No chunks found in database for the selected file.")
                    except Exception as e:
                        st.error(f"Failed to generate summary: {str(e)}")

    # --- TAB 3: Insights & Q&A Generation ---
    with tab_insights:
        st.markdown('<h2 style="color: white;">📊 Key Insights & Discussion Questions</h2>', unsafe_allow_html=True)
        st.write("Extract deep learning concepts, terminology, and generated study questions from all your ingested documents.")

        if not st.session_state.vector_db or not st.session_state.vector_db.collection or not st.session_state.uploaded_files:
            st.info("Upload and process documents in the sidebar to generate Insights.")
        else:
            if st.button("Extract Insights & Questions", type="primary"):
                with st.spinner("Analyzing cross-document repository..."):
                    try:
                        # Get a sample of top/random chunks from the vector DB to represent the dataset
                        results = st.session_state.vector_db.get_all_documents(limit=15)
                        
                        if results and results["documents"]:
                            combined_text = "\n\n---\n\n".join(results["documents"])
                            
                            insight_prompt = f"""You are a research mentor. Analyse the provided text chunks and output:
1. **Core Scientific Concepts & Definitions**: Identify the 3-5 most critical terms/theories and define them.
2. **Key Takeaways & Core Claims**: Summarize the primary assertions of the text.
3. **Advanced Discussion/Research Questions**: Formulate 3 thought-provoking questions suitable for further academic study.

Context:
{combined_text}
"""
                            client = genai.Client(api_key=api_key)
                            response = client.models.generate_content(
                                model=selected_model_id,
                                contents=insight_prompt
                            )
                            
                            st.markdown('<div class="metric-card" style="margin-top: 1.5rem;">', unsafe_allow_html=True)
                            st.markdown("### Generated Insights")
                            st.write(response.text)
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.error("Vector database is empty. No concepts can be extracted.")
                    except Exception as e:
                        st.error(f"Failed to extract insights: {str(e)}")

with right_sidebar_col:
    render_query_performance_sidebar()


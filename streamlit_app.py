import streamlit as st
import os
import sys
import time
import chromadb
from google import genai
from dotenv import load_dotenv

# Ensure project directory is in sys.path for robust relative module imports
project_dir = os.path.dirname(os.path.abspath(__file__))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

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

    /* Custom Webkit Scrollbars */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: #0d0f14;
    }
    ::-webkit-scrollbar-thumb {
        background: #1e293b;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #334155;
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
        padding-top: 1.2rem !important;
        padding-bottom: 2rem !important;
        max-width: 1400px !important;
    }

    /* Gradient Header Text */
    .gradient-title {
        background: linear-gradient(135deg, #c084fc 0%, #818cf8 50%, #60a5fa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: -0.04rem;
        margin-bottom: 0.2rem;
        text-align: center;
    }

    .subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        font-weight: 400;
        margin-bottom: 1.75rem;
        text-align: center;
    }

    /* Sidebar Custom Styling */
    section[data-testid="stSidebar"] {
        background-color: #0f121d !important;
        border-right: 1px solid rgba(255, 255, 255, 0.07);
    }

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

    /* Hide empty Streamlit sidebar navigation spacing */
    div[data-testid="stSidebarNav"] {
        display: none !important;
        height: 0px !important;
        margin: 0px !important;
        padding: 0px !important;
    }

    /* Vertical widget spacing inside sidebar */
    section[data-testid="stSidebar"] div.stVerticalBlock {
        gap: 0.5rem !important;
    }

    /* Sidebar Section Headers */
    .sidebar-section-header {
        color: #f1f5f9;
        font-size: 0.92rem;
        font-weight: 700;
        margin-top: 0.9rem;
        margin-bottom: 0.35rem;
        display: flex;
        align-items: center;
        gap: 6px;
        letter-spacing: 0.01rem;
    }

    /* Ingested Document List Items */
    .doc-item-card {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 10px 12px;
        margin-bottom: 8px;
        transition: border-color 0.2s ease;
    }
    .doc-item-card:hover {
        border-color: rgba(99, 102, 241, 0.3);
    }

    /* Glassmorphism Metric Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 0.9rem 1.1rem;
        margin: 0.3rem 0;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        height: 125px;
        box-sizing: border-box;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .metric-card:hover {
        border-color: rgba(99, 102, 241, 0.35);
        transform: translateY(-2px);
        box-shadow: 0 8px 20px -6px rgba(99, 102, 241, 0.25);
    }

    .metric-title {
        font-size: 0.78rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.04rem;
        margin-bottom: 0.3rem;
    }

    .metric-val {
        font-size: 1.65rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.02rem;
    }

    /* Output Content Cards for Summarizer & Insights */
    .output-card {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.5rem;
        margin-top: 1.25rem;
        color: #e2e8f0;
        line-height: 1.6;
    }

    .output-card-header {
        color: #a5b4fc;
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 1rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 0.5rem;
    }

    /* Custom Chat Styling */
    .chat-bubble {
        padding: 1.1rem 1.3rem;
        border-radius: 16px;
        margin-bottom: 1rem;
        max-width: 85%;
        line-height: 1.55;
        font-size: 0.94rem;
        animation: fadeIn 0.3s ease-in-out;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(6px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .user-bubble {
        background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%);
        color: #ffffff;
        margin-left: auto;
        border-bottom-right-radius: 3px;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.25);
    }

    .assistant-bubble {
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        color: #f1f5f9;
        margin-right: auto;
        border-bottom-left-radius: 3px;
    }

    .chat-header {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04rem;
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
        background: rgba(99, 102, 241, 0.12);
        color: #818cf8;
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 20px;
        padding: 0.25rem 0.7rem;
        font-size: 0.72rem;
        font-weight: 600;
        margin-right: 0.4rem;
        margin-top: 0.25rem;
        transition: all 0.2s ease;
    }
    .citation-tag:hover {
        background: rgba(99, 102, 241, 0.22);
        border-color: rgba(99, 102, 241, 0.4);
    }

    /* Center Tabs */
    .stTabs {
        display: flex;
        justify-content: center;
        margin-top: 0.5rem;
    }

    .stTabs > div {
        width: 100%;
    }

    .stTabs [role="tablist"] {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        width: fit-content !important;
        margin: 0 auto !important;
        gap: 10px !important;
    }

    .stTabs [role="tab"] {
        height: 44px;
        padding: 0 22px;
        border-radius: 10px 10px 0 0;
        background: rgba(30, 41, 59, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.06);
        color: #94a3b8;
        font-weight: 500;
        font-size: 0.9rem;
        transition: all 0.2s ease;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(99, 102, 241, 0.16) !important;
        border-color: rgba(99, 102, 241, 0.35) !important;
        color: #c7d2fe !important;
        font-weight: 600;
    }

    /* Chat Input Styling */
    div[data-testid="stChatInput"] {
        background: #000000 !important;
        border: 1px solid #3d3d3d !important;
        border-radius: 14px !important;
        padding: 8px !important;
        transition: border-color 0.2s ease;
    }

    div[data-testid="stChatInput"] > div {
        background: #000000 !important;
    }

    div[data-testid="stChatInput"] textarea {
        background: #000000 !important;
        color: #ffffff !important;
        border: none !important;
        box-shadow: none !important;
        caret-color: #ffffff !important;
    }

    div[data-testid="stChatInput"] textarea::placeholder {
        color: #9ca3af !important;
    }

    div[data-testid="stChatInput"]:focus-within {
        border: 1px solid #6366f1 !important;
    }

    button[data-testid="stChatInputSubmitButton"] {
        background-color: #10b981 !important;
        color: white !important;
        border-radius: 10px !important;
    }

    /* Button Styling */
    button[kind="primary"] {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.25) !important;
        transition: all 0.2s ease !important;
    }
    button[kind="primary"]:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 18px rgba(99, 102, 241, 0.4) !important;
    }

    button[kind="secondary"] {
        background-color: #000000 !important;
        color: #ffffff !important;
        border: 1px solid #404040 !important;
        border-radius: 10px !important;
        transition: all 0.2s ease !important;
    }

    button[kind="secondary"]:hover {
        background-color: #1a1a1a !important;
        border-color: #666666 !important;
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

if "query_perf" not in st.session_state:
    st.session_state.query_perf = None

if "reranker" not in st.session_state:
    st.session_state.reranker = Reranker()
    st.session_state.reranker._load_model()

# --- Sidebar UI Configuration ---
with st.sidebar:
    st.markdown('<div style="text-align: center; margin-top: 0px; margin-bottom: 0.2rem; line-height: 1.0;"><span style="font-size: 3rem;">🎓</span></div>', unsafe_allow_html=True)
    st.markdown('<h2 style="text-align: center; color: white; margin-top: 0px; margin-bottom: 0.4rem; font-size: 1.5rem; font-weight: 700;">Configuration</h2>', unsafe_allow_html=True)
    
    # 1. Securely retrieve API Key from environment variables (local .env) or Streamlit Secrets (.streamlit/secrets.toml)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets.get("GEMINI_API_KEY") or (st.secrets.get("gemini", {}).get("api_key") if isinstance(st.secrets.get("gemini"), dict) else None)
        except Exception:
            api_key = None

    if not api_key:
        st.sidebar.error("🔑 Gemini API Key not found! Please configure GEMINI_API_KEY in your local .env file or Streamlit Secrets (.streamlit/secrets.toml).")
    
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

    st.markdown('<div class="sidebar-section-header">⚙️ Document Ingestion Parameters</div>', unsafe_allow_html=True)
    
    chunk_size = st.slider("Chunk Size (characters)", min_value=300, max_value=2000, value=800, step=100)
    chunk_overlap = st.slider("Chunk Overlap (characters)", min_value=0, max_value=max(0, chunk_size - 50), value=min(150, max(0, chunk_size - 50)), step=25)
    
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        st.sidebar.error("⚠️ Invalid Chunk Overlap: Must satisfy 0 <= Chunk Overlap < Chunk Size.")
    
    st.markdown('<div class="sidebar-section-header">🔍 Retrieval & Reranking Parameters</div>', unsafe_allow_html=True)
    top_k = st.slider("Vector Candidates (Top-K)", min_value=5, max_value=25, value=10, step=1)
    top_p = st.slider("Reranked Results (Top-P)", min_value=1, max_value=top_k, value=min(3, top_k), step=1)
    
    st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 0.8rem 0;'>", unsafe_allow_html=True)
    
    # Document Upload Section
    st.markdown('<div class="sidebar-section-header">📄 Upload Research Papers</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload files (.pdf, .txt, .md, .docx)",
        type=["pdf", "txt", "md", "docx"],
        accept_multiple_files=True
    )
    
    # Ingest action button
    ingest_button = st.button("Process & Embed Documents", use_container_width=True, type="primary")

    # Ingested documents list below the button
    if st.session_state.uploaded_files:
        st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 0.8rem 0;'>", unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section-header">📚 Ingested Documents</div>', unsafe_allow_html=True)
        
        for filename, info in st.session_state.uploaded_files.items():
            size_mb = info["size_bytes"] / (1024 * 1024)
            st.markdown(f"""
                <div class="doc-item-card">
                    <div style="display: flex; align-items: center; gap: 6px;">
                        <span style="font-size: 0.9rem;">📄</span>
                        <strong style="color: #c7d2fe; font-size: 0.85rem; word-break: break-all;">{filename}</strong>
                    </div>
                    <div style="color: #94a3b8; font-size: 0.72rem; margin-top: 4px; display: flex; justify-content: space-between;">
                        <span>Size: {size_mb:.2f} MB</span>
                        <span>Chunks: {info["chunk_count"]}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
        if st.button("Reset Session Database", type="secondary", use_container_width=True):
            if st.session_state.vector_db:
                st.session_state.vector_db.reset_db()
            st.session_state.uploaded_files = {}
            st.session_state.chat_history = []
            st.session_state.doc_embedding_time = 0.0
            st.session_state.query_perf = None
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
st.markdown('<h1 class="gradient-title">Enterprise RAG Systems</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Client-side document intelligence & high-precision Q&A powered by Gemini & ChromaDB</p>', unsafe_allow_html=True)

# Top Metric Stats (Dynamic Row)
col1, col2, col3, col4, col5, col6 = st.columns(6)
with col1:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Gemini Model</div>
            <div class="metric-val" style="font-size: 0.92rem; color: #a5b4fc; margin-top: 4px; line-height: 1.3;">{selected_model_id}</div>
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

with col6:
    perf = st.session_state.get("query_perf")
    if perf and perf.get("total_ms") is not None:
        emb_str = f"{perf['embedding_ms']} ms"
        vec_str = f"{perf['vector_search_ms']} ms"
        rerank_str = f"{perf['reranker_ms']} ms"
        llm_str = f"{perf['llm_ms']} ms"
        total_str = f"{perf['total_ms']} ms"
    else:
        emb_str = "-- ms"
        vec_str = "-- ms"
        rerank_str = "-- ms"
        llm_str = "-- ms"
        total_str = "-- ms"

    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">⚡ Query Performance</div>
            <div style="display: flex; flex-direction: column; gap: 2px; font-size: 0.75rem; color: #94a3b8; margin-top: 2px;">
                <div style="display: flex; justify-content: space-between; align-items: center;"><span>Query Embedding Time</span><span style="font-weight: 600; color: #ffffff;">{emb_str}</span></div>
                <div style="display: flex; justify-content: space-between; align-items: center;"><span>Vector Search Time</span><span style="font-weight: 600; color: #ffffff;">{vec_str}</span></div>
                <div style="display: flex; justify-content: space-between; align-items: center;"><span>Reranker Time</span><span style="font-weight: 600; color: #ffffff;">{rerank_str}</span></div>
                <div style="display: flex; justify-content: space-between; align-items: center;"><span>LLM Time</span><span style="font-weight: 600; color: #ffffff;">{llm_str}</span></div>
                <div style="display: flex; justify-content: space-between; align-items: center; border-top: 1px solid rgba(255, 255, 255, 0.1); padding-top: 2px; margin-top: 1px;"><span>Total Query Latency</span><span style="font-weight: 700; color: #a5b4fc;">{total_str}</span></div>
            </div>
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

        # Process retrieval and answer generation
        retrieved_sources = []
        context_str = ""
        
        query_embedding_time_ms = 0.0
        vector_search_time_ms = 0.0
        reranker_time_ms = 0.0
        llm_time_ms = 0.0

        if st.session_state.vector_db and st.session_state.vector_db.collection:
            # 1. Query Vector DB for top-K matching candidate chunks
            t_vec_start = time.perf_counter()
            results = st.session_state.vector_db.query_similarity(user_query, k=top_k)
            t_vec_end = time.perf_counter()

            total_vec_db_ms = (t_vec_end - t_vec_start) * 1000.0
            query_embedding_time_ms = getattr(st.session_state.vector_db.emb_fn, "last_call_duration", 0.0) * 1000.0
            vector_search_time_ms = max(0.0, total_vec_db_ms - query_embedding_time_ms)

            # Format retrieved context after Reranking
            if results and results["documents"] and results["documents"][0]:
                candidate_docs = results["documents"][0]
                candidate_metadatas = results["metadatas"][0]
                
                # 2. Apply CrossEncoder Reranking: Top-K candidates -> Top-P relevant chunks
                t_rerank_start = time.perf_counter()
                reranked_docs, reranked_metadatas, _ = st.session_state.reranker.rerank(
                    query=user_query,
                    documents=candidate_docs,
                    metadatas=candidate_metadatas,
                    top_p=top_p
                )
                t_rerank_end = time.perf_counter()
                reranker_time_ms = (t_rerank_end - t_rerank_start) * 1000.0
                
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

        # Generate Streaming Response
        with chat_container:
            # We construct assistant bubble placeholder
            assistant_placeholder = st.empty()
            
            # Run stream
            full_response = ""
            try:
                # Instantiate Client
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

                t_llm_start = time.perf_counter()
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
                t_llm_end = time.perf_counter()
                llm_time_ms = (t_llm_end - t_llm_start) * 1000.0

                emb_ms = round(query_embedding_time_ms)
                vec_ms = round(vector_search_time_ms)
                rerank_ms = round(reranker_time_ms)
                llm_ms = round(llm_time_ms)
                total_ms = emb_ms + vec_ms + rerank_ms + llm_ms

                st.session_state.query_perf = {
                    "embedding_ms": emb_ms,
                    "vector_search_ms": vec_ms,
                    "reranker_ms": rerank_ms,
                    "llm_ms": llm_ms,
                    "total_ms": total_ms
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
                        
                        st.markdown(f"""
                            <div class="output-card">
                                <div class="output-card-header">📄 Executive Summary for {selected_file}</div>
                        """, unsafe_allow_html=True)
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
                        
                        st.markdown("""
                            <div class="output-card">
                                <div class="output-card-header">📊 Repository Key Insights & Discussion Questions</div>
                        """, unsafe_allow_html=True)
                        st.write(response.text)
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.error("Vector database is empty. No concepts can be extracted.")
                except Exception as e:
                    st.error(f"Failed to extract insights: {str(e)}")

# Document manager was moved to the sidebar

"""
Streamlit application for Retrieval-Augmented Generation (RAG) system.
Combines document processing, vector retrieval, and LLM-based answer generation.
"""

import streamlit as st
import os
from dotenv import load_dotenv
from vectorstore import VectorStore
from chatbot import RAGChatbot, ConversationManager
import tempfile

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG Document QA",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chatbot" not in st.session_state:
    st.session_state.chatbot = None
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "documents_indexed" not in st.session_state:
    st.session_state.documents_indexed = False

# Title
st.title("🔍 RAG Document Question Answering System")
st.markdown("""
Ask questions about your documents using AI-powered Retrieval-Augmented Generation (RAG).
The system retrieves relevant content and generates accurate, context-grounded answers.
""")

# ============================================================================
# SIDEBAR - Configuration and Document Management
# ============================================================================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Keys Section
    with st.expander("🔑 API Keys", expanded=False):
        openai_api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
            key="openai_key"
        )
       
    
    # RAG Settings Section
    st.subheader("📚 RAG Settings")
    col1, col2 = st.columns(2)
    with col1:
        chunk_size = st.number_input("Chunk Size", 256, 2048, 1024, step=256)
    with col2:
        chunk_overlap = st.number_input("Chunk Overlap", 0, 512, 128, step=64)
    
    top_k = st.slider("Retrieved Documents (K)", 1, 10, 4)
    
    st.divider()
    
    # LLM Settings Section
    st.subheader("🤖 LLM Settings")
    model = st.selectbox(
        "Model",
        ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
        index=0
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, step=0.1)
    max_tokens = st.slider("Max Response Tokens", 100, 2000, 500, step=100)
    
    st.divider()
    
    # Document Upload Section
    st.subheader("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF or Text Files",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("🚀 Process & Index Documents", key="process_btn", use_container_width=True):
            # Validate API keys
            if not openai_api_key:
                st.error("❌ Please provide both OpenAI and Pinecone API keys")
                st.stop()
            
            with st.spinner("Processing documents..."):
                try:
                    # Initialize VectorStore
                    vectorstore = VectorStore(
                          openai_api_key=openai_api_key,
                          chunk_size=chunk_size,
                          chunk_overlap=chunk_overlap,)
                    
                    # Process each uploaded file
                    progress_bar = st.progress(0)
                    total_chunks = 0
                    
                    for idx, uploaded_file in enumerate(uploaded_files):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                            tmp_file.write(uploaded_file.getbuffer())
                            tmp_path = tmp_file.name
                        
                        try:
                            if uploaded_file.name.endswith(".pdf"):
                                success = vectorstore.process_and_index_pdf(tmp_path)
                            else:
                                success = vectorstore.process_and_index_text(tmp_path)
                            
                            if success:
                                st.write(f"✅ Processed: {uploaded_file.name}")
                        finally:
                            os.unlink(tmp_path)
                        
                        progress_bar.progress((idx + 1) / len(uploaded_files))
                    
                    # Store in session state
                    st.session_state.vectorstore = vectorstore
                    st.session_state.chatbot = RAGChatbot(
                        openai_api_key=openai_api_key,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    st.session_state.conversation = ConversationManager()
                    st.session_state.documents_indexed = True
                    
                    st.success("✅ All documents indexed and ready for queries!")
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    # Status and Clear Section
    st.divider()
    if st.session_state.documents_indexed:
        st.success("✅ Documents Ready")
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.vectorstore = None
            st.session_state.chatbot = None
            st.session_state.conversation = None
            st.session_state.documents_indexed = False
            st.rerun()
    else:
        st.info("👈 Upload documents to get started")

# ============================================================================
# MAIN CONTENT - Question Answering
# ============================================================================
if not st.session_state.documents_indexed:
    st.warning("⚠️ Please upload and process documents first (see sidebar)")
    st.info("Steps:\n1. Enter API keys\n2. Upload PDF or text files\n3. Click 'Process & Index Documents'\n4. Ask questions!")
else:
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["💬 Ask Questions", "📜 History", "⚙️ Settings"])
    
    # ========================================================================
    # TAB 1: Question Answering
    # ========================================================================
    with tab1:
        st.subheader("Ask a Question About Your Documents")
        
        # Question input
        question = st.text_area(
            "Enter your question:",
            placeholder="What is the main topic of the document?",
            height=100
        )
        
        # Generate answer button
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            generate_btn = st.button("Generate Answer", type="primary", use_container_width=True)
        with col2:
            show_context = st.checkbox("Show Context", value=True)
        with col3:
            show_source = st.checkbox("Show Source", value=True)
        
        if generate_btn:
            if not question.strip():
                st.warning("⚠️ Please enter a question")
            else:
                with st.spinner("Retrieving relevant content and generating answer..."):
                    try:
                        # Validate API key
                        openai_key = openai_api_key if 'openai_api_key' in locals() else os.getenv("OPENAI_API_KEY")
                        if not openai_key:
                            st.error("❌ OpenAI API key not configured")
                            st.stop()
                        
                        # Retrieve relevant documents
                        retrieved = st.session_state.vectorstore.retrieve(question, top_k=top_k)
                        
                        if not retrieved:
                            st.error("❌ No relevant documents found")
                        else:
                            # Prepare context
                            context = "\n\n".join([f"[Relevance: {score:.2f}]\n{content}" for content, score in retrieved])
                            
                            # Generate answer
                            answer = st.session_state.chatbot.generate_answer(question, context)
                            
                            # Add to conversation history
                            st.session_state.conversation.add_exchange(question, answer, context)
                            
                            # Display answer
                            st.success("✅ Answer Generated")
                            st.markdown("### Answer")
                            st.write(answer)
                            
                            # Display context if requested
                            if show_context:
                                with st.expander("📚 Retrieved Context", expanded=False):
                                    for i, (content, score) in enumerate(retrieved, 1):
                                        st.markdown(f"**Document {i}** (Relevance: {score:.2f})")
                                        st.text(content[:500] + "..." if len(content) > 500 else content)
                                        st.divider()
                    
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
    
    # ========================================================================
    # TAB 2: Conversation History
    # ========================================================================
    with tab2:
        st.subheader("Conversation History")
        
        if st.session_state.conversation and st.session_state.conversation.get_history():
            history = st.session_state.conversation.get_history()
            
            # Display statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Exchanges", len(history))
            
            # Display exchanges
            for i, exchange in enumerate(history, 1):
                with st.expander(f"Exchange {i}: {exchange['question'][:50]}..."):
                    st.markdown("**Question:**")
                    st.write(exchange['question'])
                    st.markdown("**Answer:**")
                    st.write(exchange['answer'])
            
            # Clear history button
            if st.button("Clear History", use_container_width=True):
                st.session_state.conversation.clear_history()
                st.rerun()
        else:
            st.info("No conversation history yet. Ask a question to get started!")
    
    # ========================================================================
    # TAB 3: Settings
    # ========================================================================
    with tab3:
        st.subheader("System Settings")
        
        # Current configuration
        if st.session_state.chatbot:
            config = st.session_state.chatbot.get_current_config()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("LLM Model", config['model'])
            with col2:
                st.metric("Temperature", f"{config['temperature']:.1f}")
            with col3:
                st.metric("Max Tokens", config['max_tokens'])
        
        st.divider()
        
        # RAG Configuration
        st.subheader("RAG Configuration")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Chunk Size:** {chunk_size}")
            st.write(f"**Chunk Overlap:** {chunk_overlap}")
        with col2:
            st.write(f"**Top-K Retrieval:** {top_k}")
            st.write(f"**Vector DB:** Pinecone")
        
        st.divider()
        
        # Model Information
        st.subheader("Model Information")
        model_info = {
            "gpt-4": "Most capable model, best quality",
            "gpt-4-turbo": "Fast GPT-4 with extended context",
            "gpt-3.5-turbo": "Fastest and most cost-effective"
        }
        
        for model_name, description in model_info.items():
            st.write(f"**{model_name}**: {description}")

# ============================================================================
# FOOTER
# ============================================================================
st.divider()
st.markdown("""
---
### How This RAG System Works:

1. **Document Upload**: PDF and text files are uploaded and processed
2. **Text Chunking**: Documents are split into manageable chunks
3. **Embedding**: Each chunk is converted to a vector using OpenAI embeddings
4. **Indexing**: Vectors are stored in Pinecone for fast retrieval
5. **Query Processing**: Your question is converted to an embedding
6. **Retrieval**: The system finds the most relevant chunks
7. **Answer Generation**: GPT-4 generates an answer using the retrieved context

This ensures factually grounded answers based on your specific documents!
""")

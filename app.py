import streamlit as st
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os

# ==========================================
# 1. CONFIGURATION & CACHING ⚡
# ==========================================
st.set_page_config(page_title="MIT AI News Chatbot", page_icon="🦉", layout="wide")

@st.cache_resource
def load_knowledge_base():
    """Load the Vector DB and Embedding Model once and cache it."""
    chroma_client = chromadb.PersistentClient(path="./data_chroma_db")
    
    emb_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    collection = chroma_client.get_collection(
        name="data_knowledge",
        embedding_function=emb_func
    )
    return collection

@st.cache_resource
def load_dataset_info():
    """Load the CSV to show dataset statistics."""
    # Using relative path for cloud deployment
    df = pd.read_csv('MIT_AI_ARTICLES.csv')
    return df

@st.cache_resource
def load_llm():
    """Load the Groq LLM (cloud-compatible)."""
    # Get API key from Streamlit secrets
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        st.error("⚠️ GROQ_API_KEY not found in Streamlit Secrets!")
        st.stop()
    
    return ChatGroq(
        model="qwen/qwen3.8-27b",  # Verified available model, excellent for RAG
        temperature=0,
        api_key=api_key
    )

# ==========================================
# 2. CORE RAG LOGIC 🧠
# ==========================================
def get_answer(query, collection, llm):
    results = collection.query(query_texts=[query], n_results=3)
    docs = results['documents'][0]
    metadatas = results['metadatas'][0]
    
    # Truncate to save context window and speed up response
    truncated_docs = [doc[:500] + "..." if len(doc) > 500 else doc for doc in docs]
    
    prompt = ChatPromptTemplate.from_template(
        "You are an expert AI research assistant from MIT. "
        "Answer the user's question based ONLY on the provided context. "
        "If the answer is not in the context, say you don't know.\n\n"
        "Context: {context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )
    chain = prompt | llm
    response = chain.invoke({
        "context": "\n\n".join(truncated_docs), 
        "question": query
    })
    
    return response.content, metadatas

# ==========================================
# 3. STREAMLIT UI 🎨
# ==========================================
st.title("🦉 MIT AI News Chatbot")
st.caption("Ask questions about the latest research in Protein Models, Drug Discovery, and AI.")

# Load Resources
try:
    collection = load_knowledge_base()
    df = load_dataset_info()
    llm = load_llm()
    st.success("✅ Knowledge Base & LLM Loaded!")
except Exception as e:
    st.error(f"Error loading resources: \n\n{e}")
    st.stop()

# ==========================================
# 4. SIDEBAR: WHAT'S IN THE KNOWLEDGE BASE? 📚
# ==========================================
with st.sidebar:
    st.header("📊 Knowledge Base Info")
    
    st.metric("Total Articles", len(df))
    
    if 'publication_date' in df.columns:
        df['publication_date'] = pd.to_datetime(df['publication_date'], errors='coerce')
        min_date = df['publication_date'].min().strftime('%Y-%m-%d')
        max_date = df['publication_date'].max().strftime('%Y-%m-%d')
        st.caption(f"📅 Date Range: {min_date} to {max_date}")
    
    st.subheader("👥 Top Authors")
    top_authors = df['author'].value_counts().head(5)
    for author, count in top_authors.items():
        st.write(f"**{author}**: {count} articles")
    
    st.divider()
    
    st.subheader("💡 Try Asking:")
    sample_questions = [
        "What is a protein language model?",
        "How is AI being used for drug discovery?",
        "What are the latest breakthroughs in robotics?",
        "How does AI help with climate change?",
        "Tell me about autonomous drones research"
    ]
    
    for q in sample_questions:
        if st.button(q, key=f"sample_{q}", use_container_width=True):
            st.session_state.selected_question = q
    
    st.divider()
    
    st.subheader("📰 Recent Articles")
    recent_articles = df.head(5)
    for idx, row in recent_articles.iterrows():
        with st.expander(f"📄 {str(row['title'])[:50]}..."):
            st.write(f"**Author:** {row['author']}")
            st.write(f"**Date:** {row['publication_date']}")

# ==========================================
# 5. MAIN CHAT INTERFACE 💬
# ==========================================
default_query = st.session_state.get('selected_question', '')

query = st.text_input(
    "What would you like to know?", 
    placeholder="e.g., How is AI used for drug discovery?",
    value=default_query
)

if st.button("Ask the AI") or default_query:
    if query:
        with st.spinner("🤔 Thinking..."):
            answer, sources = get_answer(query, collection, llm)
            
            st.markdown("### 🤖 Response")
            st.write(answer)
            
            st.markdown("### 📚 Sources")
            for i, meta in enumerate(sources):
                with st.expander(f"Source {i+1}: {meta.get('title', 'Unknown')}"):
                    st.write(f"**Author:** {meta.get('author', 'Unknown')}")
        
        if 'selected_question' in st.session_state:
            del st.session_state.selected_question

st.divider()
st.caption("Built with ❤️ using Streamlit, ChromaDB, and Groq | Dataset: MIT AI News")
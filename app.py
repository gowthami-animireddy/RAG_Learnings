import streamlit as st
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime

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
    df = pd.read_csv('/home/gowthamireddy/chat_with_your_data/MIT_AI_ARTICLES.csv')
    return df

@st.cache_resource
def load_llm():
    """Load the Ollama LLM once."""
    return ChatOllama(model="llama3.2", temperature=0)

# ==========================================
# 2. CORE RAG LOGIC 🧠
# ==========================================
def get_answer(query, collection, llm):
    results = collection.query(query_texts=[query], n_results=3)
    docs = results['documents'][0]
    metadatas = results['metadatas'][0]
    
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
except Exception as e:
    st.error(f"Error loading database. Did you run the ingestion script first? \n\n{e}")
    st.stop()

# ==========================================
# 4. SIDEBAR: WHAT'S IN THE KNOWLEDGE BASE? 📚
# ==========================================
with st.sidebar:
    st.header("📊 Knowledge Base Info")
    
    # Dataset Statistics
    st.metric("Total Articles", len(df))
    
    # Date Range
    if 'publication_date' in df.columns:
        df['publication_date'] = pd.to_datetime(df['publication_date'], errors='coerce')
        min_date = df['publication_date'].min().strftime('%Y-%m-%d')
        max_date = df['publication_date'].max().strftime('%Y-%m-%d')
        st.caption(f"📅 Date Range: {min_date} to {max_date}")
    
    # Top Authors
    st.subheader("👥 Top Authors")
    top_authors = df['author'].value_counts().head(5)
    for author, count in top_authors.items():
        st.write(f"**{author}**: {count} articles")
    
    st.divider()
    
    # Sample Questions
    st.subheader("💡 Try Asking:")
    sample_questions = [
        "What is a protein language model?",
        "How is AI being used for drug discovery?",
        "What are the latest breakthroughs in robotics?",
        "How does AI help with climate change?",
        "What is generative AI used for in healthcare?",
        "Tell me about autonomous drones research"
    ]
    
    for q in sample_questions:
        if st.button(q, key=f"sample_{q}", use_container_width=True):
            st.session_state.selected_question = q
    
    st.divider()
    
    # Recent Articles
    st.subheader("📰 Recent Articles")
    recent_articles = df.head(5)
    for idx, row in recent_articles.iterrows():
        with st.expander(f"📄 {row['title'][:50]}..."):
            st.write(f"**Author:** {row['author']}")
            st.write(f"**Date:** {row['publication_date']}")
            st.caption(row['summary'][:150] + "...")

# ==========================================
# 5. MAIN CHAT INTERFACE 💬
# ==========================================
# Check if a sample question was clicked
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
            
            # Display Answer
            st.markdown("### 🤖 Response")
            st.write(answer)
            
            # Display Sources (Citations)
            st.markdown("### 📚 Sources")
            for i, meta in enumerate(sources):
                with st.expander(f"Source {i+1}: {meta.get('title', 'Unknown')}"):
                    st.write(f"**Author:** {meta.get('author', 'Unknown')}")
                    st.caption(f"Retrieved from MIT AI News dataset")
        
        # Clear the selected question after use
        if 'selected_question' in st.session_state:
            del st.session_state.selected_question

# Footer
st.divider()
st.caption("Built with ❤️ using Streamlit, ChromaDB, and Llama 3.2 | Dataset: MIT AI News (2023-2025)")
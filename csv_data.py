import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

# ==========================================
# 1. DATA PREP: Your Pandas Superpower 🐼
# ==========================================
print("📂 Loading MIT AI Articles dataset...")
data = pd.read_csv('/home/gowthamireddy/chat_with_your_data/csv_data.py')
df = pd.DataFrame(data)
print(f"✅ Dataset loaded successfully! Shape: {df.shape}\n")

# ==========================================
# 2. VECTOR DATABASE SETUP 🗄️
# ==========================================
print("⚙️ Initializing ChromaDB...")
chroma_client = chromadb.PersistentClient(path="./data_chroma_db")

# Clear old data to prevent duplicates on re-runs
try:
    chroma_client.delete_collection("data_knowledge")
    print("🗑️ Cleared old data from ChromaDB")
except Exception:
    pass  # Collection doesn't exist yet, that's fine

# Use a free, local embedding model to convert text to numbers
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2" # Fast, lightweight, and runs on CPU
)

# Create a collection
collection = chroma_client.get_or_create_collection(
    name="data_knowledge",
    embedding_function=embedding_function
)

# ==========================================
# 3. INGEST DATA INTO THE VECTOR DB 🚀
# ==========================================
print("🔄 Embedding and storing articles (this may take 1-2 minutes)...")
documents = df['body'].fillna('').tolist()
ids = [str(i) for i in range(len(documents))]

# PRO TIP: Fill NA values in metadata to prevent ChromaDB type errors
metadatas = [
    {
        "title": str(row['title']) if pd.notna(row['title']) else "Unknown Title",
        "author": str(row['author']) if pd.notna(row['author']) else "Unknown Author"
    } 
    for _, row in df.iterrows()
]

# Add to ChromaDB
collection.add(
    documents=documents,
    ids=ids,
    metadatas=metadatas
)
print("✅ Data successfully embedded and stored in ChromaDB!\n")

# ==========================================
# 4. THE RAG QUERY FUNCTION 🔍
# ==========================================
def ask_data(query: str, top_k: int = 2):
    print(f"🔥 You asked: '{query}'")
    
    # Step A: RETRIEVAL
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    
    retrieved_docs = results['documents'][0]
    retrieved_titles = [meta['title'] for meta in results['metadatas'][0]]
    
    # TRUNCATE LONG DOCUMENTS - Keep first 500 characters to save LLM context window
    truncated_docs = [doc[:500] + "..." if len(doc) > 500 else doc for doc in retrieved_docs]
    
    print(f"📚 Retrieved {len(retrieved_docs)} articles:")
    for i, title in enumerate(retrieved_titles):
        print(f"   {i+1}. {title}")
    print()
    
    # Step B: GENERATION
    from langchain_ollama import ChatOllama
    from langchain_core.prompts import ChatPromptTemplate

    llm = ChatOllama(model="llama3.2", temperature=0)
    
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful AI assistant. Answer the question based ONLY on the provided context.\n"
        "Context: {context}\n"
        "Question: {question}\n"
        "Answer:"
    )
    
    chain = prompt | llm
    response = chain.invoke({"context": "\n\n".join(truncated_docs), "question": query})
    print(f"🤖 AI Response:\n{response.content}")
    print("-" * 60)

# ==========================================
# 5. TEST IT OUT! 🧪
# ==========================================
ask_data("What is a protein language model?")
ask_data("What kind of AI is being created for drug discovery?")
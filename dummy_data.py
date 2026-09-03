import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

# ==========================================
# 1. DATA PREP: Your Pandas Superpower 🐼
# ==========================================
# Let's create a "Faya" dataset about GenAI concepts
data = {
    "id": [1, 2, 3, 4],
    "topic": ["Generative AI", "Vector Databases", "Prompt Engineering", "Fine-Tuning"],
    "description": [
        "Generative AI creates new content like text, images, or code using patterns learned from massive datasets.",
        "Vector databases store data as mathematical embeddings, enabling lightning-fast similarity searches for RAG applications.",
        "Prompt engineering is the art of crafting precise inputs to guide Large Language Models to produce accurate outputs.",
        "Fine-tuning adapts a pre-trained model to a specific task or domain using a smaller, specialized dataset."
    ]
}
df = pd.DataFrame(data)
print("✅ Dataset loaded:\n", df[["topic", "description"]], "\n")

# ==========================================
# 2. VECTOR DATABASE SETUP 🗄️
# ==========================================
# Initialize a local ChromaDB client (saves to a folder on your machine)
chroma_client = chromadb.PersistentClient(path="./faya_chroma_db")

# Use a free, local embedding model to convert text to numbers
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2" # Fast, lightweight, and runs on CPU
)

# Create a collection (like a "table" in SQL, but for vectors)
collection = chroma_client.get_or_create_collection(
    name="faya_knowledge",
    embedding_function=embedding_function
)

# ==========================================
# 3. INGEST DATA INTO THE VECTOR DB 🚀
# ==========================================
# Combine topic and description for richer context
documents = [f"Topic: {row['topic']}. Details: {row['description']}" for _, row in df.iterrows()]
ids = [str(row['id']) for _, row in df.iterrows()]
metadatas = [{"topic": row['topic']} for _, row in df.iterrows()]

# Add to ChromaDB (this automatically triggers the embedding function!)
collection.add(
    documents=documents,
    ids=ids,
    metadatas=metadatas
)
print("✅ Data successfully embedded and stored in ChromaDB!\n")

# ==========================================
# 4. THE RAG QUERY FUNCTION 🔍
# ==========================================
def ask_faya(query: str, top_k: int = 2):
    print(f"🔥 You asked: '{query}'")
    
    # Step A: RETRIEVAL - Search the vector DB for similar text
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    
    # Extract the retrieved text and metadata
    retrieved_docs = results['documents'][0]
    retrieved_topics = [meta['topic'] for meta in results['metadatas'][0]]
    
    print(f"📚 Retrieved Context from DB:\n - " + "\n - ".join(retrieved_docs) + "\n")
    
    # Step B: GENERATION - (Simulated for now, see Step 3 below to make it real!)
    print(f"🤖 AI Response: Based on my database, this relates to {', '.join(retrieved_topics)}.")
    print("-" * 50)

# ==========================================
# 5. TEST IT OUT! 🧪
# ==========================================
ask_faya("How do we store data for fast similarity search?")
ask_faya("What is the art of crafting precise inputs for models?")
ask_faya("Tell me about adapting pre-trained models.")
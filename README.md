# 🔥 Chat with your Faya: A Mini RAG Pipeline

Welcome to **Chat with your Faya**, a foundational Retrieval-Augmented Generation (RAG) application. This project demonstrates the core mechanics of how modern Generative AI applications retrieve private, domain-specific data and use Large Language Models (LLMs) to generate accurate, context-aware answers.

## 🧠 What is RAG?
Retrieval-Augmented Generation (RAG) solves a major limitation of standard LLMs: hallucination and lack of up-to-date/private knowledge. Instead of relying solely on an LLM's pre-trained memory, RAG:
1. **Retrieves** relevant documents from a custom knowledge base.
2. **Augments** the LLM's prompt with this retrieved context.
3. **Generates** a factual answer based *only* on the provided data.

## 🛠️ Tech Stack
- **Python 3.11+**: Core programming language.
- **Pandas**: Data manipulation and structuring.
- **Sentence Transformers**: Converting text into high-dimensional mathematical embeddings.
- **ChromaDB**: A lightweight, lightning-fast local Vector Database.
- **LangChain & Ollama**: Orchestrating the prompt and connecting to local LLMs (Llama 3.2).

## 🏗️ Architecture Flow
1. **Data Ingestion**: Raw text data is loaded and structured using Pandas.
2. **Embedding**: Text chunks are passed through a local embedding model (`all-MiniLM-L6-v2`) to create vector representations.
3. **Vector Storage**: Vectors and metadata are stored in a persistent ChromaDB collection.
4. **Semantic Retrieval**: A user query is embedded and compared against the database using cosine similarity to find the most relevant context.
5. **Generation**: The retrieved context is injected into a strict prompt and sent to a local LLM to generate the final answer.

## 🚀 Getting Started

### 1. Prerequisites
Make sure you have Python 3.9+ installed.

### 2. Installation
Clone the repository and install the required dependencies:
```bash
pip install pandas chromadb sentence-transformers

3. Connect a Real Local LLM
If you want to connect a real local LLM instead of the simulated response:
Install Ollama and pull a model:

# Download Ollama from https://ollama.com, then run:

```bash 
ollama pull llama3.2

Install LangChain wrappers:

pip install langchain langchain-ollama

Expected Output:
✅ Dataset loaded:
                 topic                                        description
0       Generative AI  Generative AI creates new content like text, i...
1    Vector Databases  Vector databases store data as mathematical em...
2  Prompt Engineering  Prompt engineering is the art of crafting prec...
3         Fine-Tuning  Fine-tuning adapts a pre-trained model to a sp...

✅ Data successfully embedded and stored in ChromaDB!

🔥 You asked: 'How do we store data for fast similarity search?'
📚 Retrieved Context from DB:
 - Topic: Vector Databases. Details: Vector databases store data as mathematical embeddings...
 - Topic: Generative AI. Details: Generative AI creates new content like text, images...

🤖 AI Response: Based on my database, this relates to Vector Databases, Generative AI.
--------------------------------------------------

📂 Project Structure

chat-with-your-faya-rag/
├── faya_rag.py              # Main RAG pipeline script
├── faya_chroma_db/          # Local persistent vector database (auto-generated)
├── .gitignore               # Git ignore file
└── README.md                # Project documentation
# 🔥 Chat with your Data: A Mini RAG Pipeline

Welcome to **Chat with your Data**, a foundational **Retrieval-Augmented Generation (RAG)** application.

This project demonstrates the core mechanics behind modern Generative AI applications: retrieving relevant information from a private knowledge base and providing it to an LLM to generate accurate, context-aware responses.

---

## 🧠 What is RAG?

**Retrieval-Augmented Generation (RAG)** helps address two major limitations of standard LLMs:

* Hallucinations
* Lack of access to private or up-to-date information

Instead of relying solely on an LLM's pre-trained knowledge, a RAG system follows three main steps:

1. **Retrieval** — Find relevant information from a custom knowledge base.
2. **Augmentation** — Add the retrieved information to the LLM's prompt as context.
3. **Generation** — Generate an answer using the retrieved context.

### 🔄 RAG Pipeline

```text
User Query
    ↓
Query Embedding
    ↓
Vector Database
    ↓
Similarity Search
    ↓
Relevant Context
    ↓
Prompt + Context
    ↓
LLM
    ↓
Final Answer
```

---

## 🛠️ Tech Stack

| Technology                | Purpose                              |
| ------------------------- | ------------------------------------ |
| **Python 3.11+**          | Core programming language            |
| **Pandas**                | Data loading and manipulation        |
| **Sentence Transformers** | Converts text into vector embeddings |
| **ChromaDB**              | Local persistent vector database     |
| **LangChain**             | LLM/RAG orchestration                |
| **Ollama**                | Runs local LLMs                      |
| **Llama 3.2**             | Local language model                 |

---

## 🏗️ Architecture Flow

### 1. 📥 Data Ingestion

Raw text data is loaded and structured using **Pandas**.

Example:

| Topic              | Description                                                            |
| ------------------ | ---------------------------------------------------------------------- |
| Generative AI      | Generative AI creates new content such as text and images.             |
| Vector Databases   | Vector databases store data as mathematical embeddings.                |
| Prompt Engineering | Prompt engineering involves designing effective instructions for LLMs. |
| Fine-Tuning        | Fine-tuning adapts a pre-trained model to a specific task or domain.   |

### 2. 🧮 Embedding

Each piece of text is converted into a numerical vector using the:

```text
all-MiniLM-L6-v2
```

embedding model.

These vectors represent the **semantic meaning** of the text.

### 3. 🗄️ Vector Storage

The generated embeddings, along with their original text and metadata, are stored in a persistent **ChromaDB** collection.

```text
Text
 ↓
Embedding Model
 ↓
Vector
 ↓
ChromaDB
```

### 4. 🔍 Semantic Retrieval

When a user asks a question:

1. The query is converted into an embedding.
2. The query embedding is compared with stored embeddings.
3. ChromaDB performs a similarity search.
4. The most relevant documents are retrieved.

This allows the system to find information based on **meaning**, rather than only exact keyword matches.

### 5. 🤖 Generation

The retrieved context is inserted into a prompt and sent to the local LLM.

The LLM then generates a response based on the retrieved information.

---

# 🚀 Getting Started

## 1. Prerequisites

Make sure you have:

* Python **3.9+**
* `pip`
* Git

For local LLM support, you will also need **Ollama**.

---

## 2. Install Dependencies

Clone the repository:

```bash
git clone <your-repository-url>
cd chat-with-your-data-rag
```

Install the core dependencies:

```bash
pip install pandas chromadb sentence-transformers
```

---

## 3. Run the Basic RAG Pipeline

Run the main Python script:

```bash
python data_rag.py
```

The application will:

1. Load the dataset.
2. Generate embeddings.
3. Store them in ChromaDB.
4. Convert the user query into an embedding.
5. Retrieve relevant documents.
6. Generate a response.

---

# 🦙 Connect a Real Local LLM

The project can also be connected to a real local LLM using **Ollama**.

## Install Ollama

Download and install Ollama from:

https://ollama.com

Then pull the Llama 3.2 model:

```bash
ollama pull llama3.2
```

Verify that the model is available:

```bash
ollama list
```

---

## Install LangChain Integration

Install the required LangChain packages:

```bash
pip install langchain langchain-ollama
```

The architecture then becomes:

```text
User Query
     ↓
Sentence Transformer
     ↓
ChromaDB
     ↓
Retrieved Context
     ↓
LangChain Prompt
     ↓
Ollama
     ↓
Llama 3.2
     ↓
Final Answer
```

---

# 📊 Example Output

```text
✅ Dataset loaded:

                 topic                                        description
0       Generative AI  Generative AI creates new content like text, i...
1    Vector Databases  Vector databases store data as mathematical em...
2  Prompt Engineering  Prompt engineering is the art of crafting prec...
3         Fine-Tuning  Fine-tuning adapts a pre-trained model to a sp...

✅ Data successfully embedded and stored in ChromaDB!

🔥 You asked:
'How do we store data for fast similarity search?'

📚 Retrieved Context from DB:

- Topic: Vector Databases
  Details: Vector databases store data as mathematical embeddings...

- Topic: Generative AI
  Details: Generative AI creates new content like text, images...

🤖 AI Response:
Based on my database, this relates to Vector Databases and Generative AI.
```

---

# 📂 Project Structure

```text
chat-with-your-data-rag/
│
├── data_rag.py              # Main RAG pipeline
│
├── data_chroma_db/          # Persistent ChromaDB storage
│
├── .gitignore               # Git ignore rules
│
└── README.md                # Project documentation
```

> `data_chroma_db/` is automatically generated when the vector database is created.

---

# 🎯 Key Concepts Demonstrated

This project provides hands-on experience with:

* Retrieval-Augmented Generation (RAG)
* Text embeddings
* Semantic search
* Vector databases
* Similarity search
* ChromaDB
* Sentence Transformers
* Prompt augmentation
* Local LLMs
* Ollama
* LangChain


---

## ⭐ Project Goal

**Chat with your Data** is designed as a simple starting point for understanding how RAG systems work internally — from **raw data → embeddings → vector storage → retrieval → LLM generation**.

It serves as a foundation that can later be extended into a production-style RAG system.



deployed - https://mit-ai-news-chatbot.streamlit.app/

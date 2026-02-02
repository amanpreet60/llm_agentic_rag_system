# RAG Pipeline — Retrieval-Augmented Generation Module

# RAG Pipeline

## Overview
The **RAG Pipeline** (Retrieval-Augmented Generation) combines **document search** with **AI text generation**.  
It allows applications to **find relevant information** from documents or APIs and **generate accurate, context-aware answers**.  

The pipeline is fast, accurate, and avoids duplicate results, making it ideal for:
- Q&A systems  
- AI assistants  
- Knowledge search tools  

## Features
- Retrieve relevant information from documents or APIs  
- Embed text into vector databases for fast search  
- Generate AI responses using large language models  
- Keep knowledge up-to-date with real-time sources like Wikipedia  

## Technologies Used
- **Hugging Face Inference API** – AI text generation  
- **LangChain** – Document embeddings and vectorstore management  
- **Chroma** – Persistent vector database  
- **Wikipedia REST API** – Real-time information retrieval  
- **Sentence Transformers** – Semantic text embeddings  

---

## Features

- **Dynamic Knowledge Retrieval**: Automatically decides whether a query requires external knowledge.  
- **Wikipedia Integration**: Fetches and parses high-quality Wikipedia articles.  
- **Text Chunking**: Splits documents into semantically meaningful chunks using `RecursiveCharacterTextSplitter`.  
- **Semantic Embeddings**: Generates embeddings with `all-MiniLM-L6-v2` for high-quality similarity search.  
- **Near-Duplicate Filtering**: Removes redundant document chunks to optimize vector storage and retrieval.  
- **Vector Database Persistence**: Stores embeddings in Chroma for scalable, repeatable queries.  
- **LLM Answer Generation**: Generates precise answers using **Meta-LLaMA 3.1 Instruct**, conditioned on retrieved context.  
- **Modular Design**: Easy to integrate in **Python projects**, **chatbots**, or **knowledge-driven AI systems**.

---

## Tech Stack

| Component | Purpose |
|-----------|---------|
| Python 3.11 | Core language |
| Hugging Face InferenceClient | LLM inference |
| LangChain Core & Community Modules | Document handling & embeddings |
| Chroma VectorStore | Persistent semantic search |
| Sentence Transformers | High-quality embeddings (`all-MiniLM-L6-v2`) |
| Wikipedia REST & Action APIs | Real-time knowledge source |
| Transformers | Optional LLM alternatives |

---


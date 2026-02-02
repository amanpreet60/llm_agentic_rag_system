# RAG Pipeline — Retrieval-Augmented Generation Module

## Overview

**RAG Pipeline** is a **state-of-the-art Retrieval-Augmented Generation module** designed to seamlessly integrate **document retrieval** and **large language model (LLM) generation**.  
It enables applications to **fetch contextually relevant knowledge** from external sources, **embed it into vector stores**, and generate **highly accurate, context-aware responses**.

This project leverages:
- **Hugging Face Inference API** for LLM generation  
- **LangChain** for document embeddings and vectorstore management  
- **Chroma** for persistent vector databases  
- **Wikipedia REST API** for real-time knowledge retrieval  
- **Sentence Transformers** for semantic embeddings  

The RAG module is optimized for **speed, accuracy, and deduplication**, making it ideal for **question answering systems**, **AI assistants**, and **knowledge retrieval pipelines**.

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


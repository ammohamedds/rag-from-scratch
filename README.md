# rag-from-scratch
Build a lightweight Retrieval-Augmented Generation (RAG) system from scratch using Python, Ollama, Nomic Embed Text for embeddings, and Qwen 4B Instruct for response generation.


# RAG from Scratch

## Overview

This project aims to build a lightweight Retrieval-Augmented Generation (RAG) system from scratch using Python and locally running Large Language Models (LLMs).

The goal is to understand the fundamental components of a RAG pipeline, including document ingestion, text preprocessing, embeddings, vector storage, semantic retrieval, and response generation.

The project uses Ollama to run AI models locally.

## Technologies

| Technology | Purpose |
|---|---|
| Python | Main development language |
| Ollama | Local model management and execution |
| Nomic Embed Text | Generate text embeddings |
| Qwen3 4B | Generate natural-language responses |
| Git & GitHub | Version control and project management |

## Models

Two models are used:

- **nomic-embed-text**: Converts text into numerical vector representations (embeddings) for semantic search.
- **qwen3:4b**: An instruction-following language model used to generate responses based on retrieved information.

## Environment Setup

### 1. Install Ollama

Download and install Ollama for Windows:

https://ollama.com/download

Verify the installation:

```bash
ollama --version
```

### 2. Download the models

Download the embedding model:

```bash
ollama pull nomic-embed-text
```

Download the language model:

```bash
ollama pull qwen3:4b
```

Verify the downloaded models:

```bash
ollama list
```

### 3. Test the models

Test the language model:

```bash
ollama run qwen3:4b
```

Test the embedding model with sample text:

```bash
ollama run nomic-embed-text "Artificial intelligence is transforming data science."
```

**Note:** The embedding model requires input text, whereas Qwen supports interactive conversations.

## Current Progress

- [x] Created GitHub repository.
- [x] Cloned the repository locally.
- [x] Installed Ollama on Windows.
- [x] Downloaded Nomic Embed Text.
- [x] Downloaded Qwen3 4B.
- [x] Verified that both models are available locally.
- [x] Successfully tested Qwen through the CMD terminal.
- [ ] Set up Python virtual environment.
- [ ] Integrate Python with Ollama.
- [ ] Generate embeddings from Python.
- [ ] Implement document ingestion and preprocessing.
- [ ] Implement text chunking.
- [ ] Set up vector storage.
- [ ] Implement semantic retrieval.
- [ ] Generate answers using retrieved context.
- [ ] Evaluate the RAG pipeline.

## Planned RAG Architecture

The intended workflow is:

1. Document ingestion
2. Text extraction and preprocessing
3. Document chunking
4. Embedding generation
5. Vector storage
6. Similarity-based retrieval
7. Context construction
8. Answer generation using Qwen3

## Project Status

This project is currently in the environment setup and model verification phase. The Python-based RAG pipeline will be implemented incrementally.
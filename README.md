# LangChain Basics

A clean, structured learning repository for LangChain. It walks you from simple chat model usage through prompting, chains, RAG (Retrieval-Augmented Generation), and agents.

## Quick start

1. Create a virtual environment and install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file from `.env.example` and set your API key(s):

   ```bash
   cp .env.example .env
   # Edit .env and set OPENAI_API_KEY
   ```

## Project layout

- `01_chat_models/`
  - `01_basic_invoke.py` – call a chat model
  - `02_conversation_memory.py` – simple conversation/memory
  - `03_alternative_models.py` – switch models/providers
  - `04_user_conversation.py` – interactive prompts
  - `05_message_history_firebase.py` – persist history example
- `02_prompting/`
  - `01_prompt_templates_basics.py` – prompt templates
- `03_chains/`
  - `01_chain_basics.py` – basic chains
  - `02_chain_internals.py` – under the hood
  - `03_chain_sequential.py` – sequential chains
  - `04_chain_parallel.py` – parallel chains
  - `05_chain_conditional.py` – conditional chains
- `04_rag/`
  - `01_build_vectorstore_basic.py` – index one file
  - `02_query_vectorstore_basic.py` – query the index
  - `03_build_with_metadata.py` – index multiple files with metadata
  - `04_query_with_metadata.py` – query with metadata filtering
  - `05_one_off_qa.py` – compose retrieved chunks into an answer
- `05_agents/`
  - `01_agents_basics.py` – agent basics
- `data/`
  - `documents/` – sample text documents
  - `vectorstores/` – generated Chroma indexes (gitignored)

## RAG workflow

- Build an index from documents:

  ```bash
  python 04_rag/01_build_vectorstore_basic.py
  python 04_rag/03_build_with_metadata.py
  ```

- Query the index:

  ```bash
  python 04_rag/02_query_vectorstore_basic.py
  python 04_rag/04_query_with_metadata.py
  ```

- Try end-to-end QA:

  ```bash
  python 04_rag/05_one_off_qa.py
  ```

## Notes

- Vector stores are generated under `data/vectorstores/` and excluded from Git.
- Documents live under `data/documents/`. You can add or remove files freely.
- Set `OPENAI_API_KEY` in your `.env` before running scripts.

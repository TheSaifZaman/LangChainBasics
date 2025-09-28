"""
MMR (Maximal Marginal Relevance) retrieval demo on the basic vectorstore.

What this shows
- How to use `search_type="mmr"` with a Chroma-backed retriever.
- Tunable `lambda_mult` to balance relevance vs. diversity in retrieved chunks.

Prerequisite
- Build the basic index first:
    python 04_rag/01_build_vectorstore_basic.py

Run
    python 04_rag/06_query_mmr.py
"""

from pathlib import Path
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

project_root = Path(__file__).resolve().parents[1]
vs_dir = project_root / "data" / "vectorstores" / "chroma_basic"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Load the existing vectorstore and create an MMR retriever
vectorstore = Chroma(persist_directory=str(vs_dir), embedding_function=embeddings)
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5},
)

query = "Key events involving Frodo and the Ring"
results = retriever.invoke(query)

print("\n--- MMR Retrieval Results ---")
for i, doc in enumerate(results, 1):
    print(f"Doc {i}:\n{doc.page_content[:300]}\n")

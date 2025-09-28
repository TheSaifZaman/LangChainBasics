import os
from pathlib import Path
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Define the directory containing the text file and the persistent directory
project_root = Path(__file__).resolve().parents[1]
data_dir = project_root / "data"
file_path = data_dir / "documents" / "lord_of_the_rings.txt"
persistent_directory = data_dir / "vectorstores" / "chroma_basic"

# Check if the Chroma vector store already exists
if not persistent_directory.exists():
    print("Persistent directory does not exist. Initializing vector store...")

    # Ensure the text file exists
    if not file_path.exists():
        raise FileNotFoundError(
            f"The file {file_path} does not exist. Please check the path."
        )

    # Read the text content from the file
    loader = TextLoader(str(file_path))
    documents = loader.load()

    # Split the document into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50) 
    docs = text_splitter.split_documents(documents)

    # Display information about the split documents
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")
    print(f"Sample chunk:\n{docs[0].page_content}\n")

    # Create embeddings
    print("\n--- Creating embeddings ---")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )  # Update to a valid embedding model if needed
    print("\n--- Finished creating embeddings ---")

    # Create the vector store and persist it automatically
    print("\n--- Creating vector store ---")
    db = Chroma.from_documents(
        docs, embeddings, persist_directory=str(persistent_directory))
    print("\n--- Finished creating vector store ---")

else:
    print("Vector store already exists at:", persistent_directory)
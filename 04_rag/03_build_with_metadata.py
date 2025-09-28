import os
from pathlib import Path

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma

# Define the directory containing the text files and the persistent directory
project_root = Path(__file__).resolve().parents[1]
data_dir = project_root / "data"
books_dir = data_dir / "documents"
persistent_directory = data_dir / "vectorstores" / "chroma_with_metadata"

print(f"Books directory: {books_dir}")
print(f"Persistent directory: {persistent_directory}")

# Check if the Chroma vector store already exists
if not persistent_directory.exists():
    print("Persistent directory does not exist. Initializing vector store...")

    # Ensure the books directory exists
    if not books_dir.exists():
        raise FileNotFoundError(
            f"The directory {books_dir} does not exist. Please check the path."
        )

    # List all text files in the directory
    book_files = [f for f in os.listdir(books_dir) if f.endswith(".txt")]

    # Read the text content from each file and store it with metadata
    documents = []
    for book_file in book_files:
        file_path = books_dir / book_file
        loader = TextLoader(str(file_path))
        book_docs = loader.load()
        for doc in book_docs:
            # Add metadata to each document indicating its source
            doc.metadata = {"source": book_file}
            documents.append(doc)

    # Split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    # Display information about the split documents
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")

    # Create embeddings
    print("\n--- Creating embeddings ---")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )  # Update to a valid embedding model if needed
    print("\n--- Finished creating embeddings ---")

    # Create the vector store and persist it
    print("\n--- Creating and persisting vector store ---")
    db = Chroma.from_documents(
        docs, embeddings, persist_directory=str(persistent_directory))
    print("\n--- Finished creating and persisting vector store ---")

else:
    print("Vector store already exists at:", persistent_directory)

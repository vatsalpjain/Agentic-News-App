import numpy as np
import chromadb 
import uuid
import os
from typing import List
from langchain_core.documents import Document

# Vector Store
class VectorStore:
    """Manages documents embedding in a chromaDB vector store """

    def __init__(self, collection_name: str = "pdf_documents", persist_directory: str = "data/vector_store"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        """Initialize the ChromaDB client and collection"""
        try:
            # Create Persistent Directory
            os.makedirs(self.persist_directory,exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            # Get or create collection
            print("Initializing ChromaDB Client...")
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description":"PDF Document Collection"}
            )
            print(f"Collection '{self.collection_name}' is ready.")
        except Exception as e:
            print(f"Error initializing ChromaDB: {e}")
            raise

    def add_documents(self, documents: List[Document], embeddings: np.ndarray):
        """Add documents to the vector store after generating embeddings"""
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match the number of embeddings")

        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            # Generate unique ID
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            # Prepare metadata
            metadata = dict(doc.metadata)
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)

            # Document content
            documents_text.append(doc.page_content)

            # Embedding
            embeddings_list.append(embedding.tolist())

        # Add to collection
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents_text
            )
            print(f"Successfully added {len(documents)} documents to vector store")
            print(f"Total documents in collection: {self.collection.count()}")

        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise
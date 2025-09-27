# To make imports cleaner, you can selectively expose core classes/functions here
from .data_ingestion import process_all_docs
from .chunking import split_documents
from .embedding import EmbeddingManager
from .vectorstore import VectorStore
from .retriever import RAGRetriever
from .llm_interface import llm, rag_simple
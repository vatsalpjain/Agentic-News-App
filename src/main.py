from .data_ingestion import process_all_docs, scrape_bbc_headlines, search_documents
from .chunking import split_documents
from .embedding import EmbeddingManager
from .vectorstore import VectorStore
from .retriever import RAGRetriever
from .llm_interface import llm, rag_simple


def main():
    query = "who is Donald Trump?"
    
    # Load local + BBC news documents
    docs = process_all_docs('C:/dev/work/NSDC/RAGapp/data/all_files')
    
    # Search loaded documents for the query string
    search_results = search_documents(docs, query, max_results=5)
    print(f"\nSearch Results for '{query}':")
    for doc in search_results:
        print("-", doc['page_content'] if isinstance(doc, dict) else doc.page_content)
    
    # Split documents into chunks for embedding
    chunks = split_documents(docs)
    
    # Generate embeddings for chunks
    embedder = EmbeddingManager()
    embeddings = embedder.generate_embeddings([chunk.page_content for chunk in chunks])
    
    # Initialize vectorstore and add documents+embeddings
    store = VectorStore()
    store.add_documents(chunks, embeddings)
    
    # Setup retriever
    retriever = RAGRetriever(store, embedder)
    
    # Run RAG query against vectorstore
    answer = rag_simple(query, retriever, llm)
    print("\nRAG answer:", answer)
    
    # Optionally display BBC headlines fetched separately
    bbc_headlines = scrape_bbc_headlines()
    print("\nBBC News Headlines:")
    for doc in bbc_headlines:
        print("-", doc['page_content'])

if __name__ == '__main__':
    main()


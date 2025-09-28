from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor

from .data_ingestion import process_all_docs, scrape_bbc_headlines
from .chunking import split_documents
from .embedding import EmbeddingManager
from .vectorstore import VectorStore
from .retriever import RAGRetriever
from .llm_interface import llm, rag_simple

@dataclass
class NewsArticle:
    title: str
    content: str
    source: str
    url: str
    timestamp: datetime
    embedding: List[float] = None

class NewsPipeline:
    def __init__(self, 
                 vector_store: VectorStore,
                 embedding_manager: EmbeddingManager,
                 max_workers: int = 3):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        self.retriever = RAGRetriever(vector_store, embedding_manager)
        self.max_workers = max_workers
        self.logger = logging.getLogger(__name__)

    async def process_news_batch(self, articles: List[NewsArticle]):
        """Process a batch of news articles in parallel"""
        chunks = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Chunk articles
            chunked_articles = list(executor.map(
                lambda x: split_documents([x]), articles
            ))
            chunks.extend([chunk for sublist in chunked_articles for chunk in sublist])
            
            # Generate embeddings
            embeddings = self.embedding_manager.generate_embeddings(
                [chunk.page_content for chunk in chunks]
            )
            
            # Store in vector DB
            self.vector_store.add_documents(chunks, embeddings)
        
        return len(chunks)

    async def query_news(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Query the news database"""
        # Get relevant articles
        results = self.retriever.retrieve(query, top_k=top_k)
        
        # Generate summary with LLM
        summary = rag_simple(query, self.retriever, llm, top_k=top_k)
        
        return {
            "query": query,
            "summary": summary,
            "articles": results,
            "timestamp": datetime.now().isoformat()
        }

    async def refresh_news(self):  # Make refresh_news async
        """Refresh news from all sources"""
        try:
            # Process local documents
            local_docs = process_all_docs("data/all_files")
            
            # Get live news
            bbc_news = scrape_bbc_headlines()
            
            if not local_docs and not bbc_news:
                self.logger.warning("No documents found to process")
                return
            
            # Combine and process
            all_docs = local_docs + bbc_news
            chunks = split_documents(all_docs)
            
            if not chunks:
                self.logger.warning("No chunks generated from documents")
                return
                
            embeddings = self.embedding_manager.generate_embeddings(
                [chunk.page_content for chunk in chunks]
            )
            self.vector_store.add_documents(chunks, embeddings)
            
        except Exception as e:
            self.logger.error(f"Error in refresh_news: {e}")
            raise
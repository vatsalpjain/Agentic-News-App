import asyncio
import logging
from .pipeline import NewsPipeline
from .embedding import EmbeddingManager
from .vectorstore import VectorStore
from .mcp.server import NewsServer
from .utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

async def periodic_refresh(pipeline):
    while True:
        try:
            await asyncio.sleep(3600)  # Refresh every hour
            await pipeline.refresh_news()
        except Exception as e:
            logger.error(f"Error in periodic refresh: {e}")

async def main():
    try:
        setup_logging()
        logger.info("Starting NewsRag application")
        
        # Initialize components
        embedding_manager = EmbeddingManager()
        vector_store = VectorStore()
        pipeline = NewsPipeline(vector_store, embedding_manager)
        
        # Create server with pipeline
        server = NewsServer(pipeline=pipeline)
        
        # Initial news refresh
        await pipeline.refresh_news()
        
        # Start everything
        await asyncio.gather(
            server.start(),
            periodic_refresh(pipeline)
        )
    except Exception as e:
        logger.error(f"Application failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())


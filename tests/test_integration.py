import asyncio
import websockets
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def query_news(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Query the news database"""
    logger.info(f"Processing query: {query}")
    
    # Get relevant articles
    results = retriever.retrieve(query, top_k=top_k)
    logger.info(f"Found {len(results)} relevant articles")
    
    # Generate summary with LLM
    summary = rag_simple(query, retriever, llm, top_k=top_k)
    logger.info("Generated summary")
    
    return {
        "query": query,
        "summary": summary,
        "articles": results,
        "timestamp": datetime.now().isoformat()
    }

async def test_news_query():
    """Test the news query functionality"""
    uri = "ws://localhost:8765"
    try:
        async with websockets.connect(uri) as websocket:
            # Test queries
            test_queries = [
                "AI developments in 2024",
                "Climate change impact",
                "Space exploration news"
            ]
            
            for query in test_queries:
                logger.info(f"\nTesting query: {query}")
                try:
                    # Send query
                    message = {
                        "type": "query",
                        "query": query
                    }
                    await websocket.send(json.dumps(message))
                    
                    # Get response
                    response = await websocket.recv()
                    data = json.loads(response)
                    
                    # Print results
                    if data["type"] == "query_result":
                        results = data["data"]
                        logger.info(f"\nQuery: {query}")
                        logger.info(f"Summary: {results.get('summary', 'No summary available')}")
                        
                        if "articles" in results and results["articles"]:
                            logger.info("\nRetrieved Articles:")
                            for i, article in enumerate(results["articles"], 1):
                                logger.info(f"\n{i}. Score: {article['similarity_score']:.2f}")
                                logger.info(f"Content: {article['content'][:200]}...")
                        else:
                            logger.warning("No articles found for this query")
                    else:
                        logger.error(f"Error response: {data.get('data', 'Unknown error')}")
                        
                except Exception as e:
                    logger.error(f"Error processing query '{query}': {e}")
                
                await asyncio.sleep(1)  # Small delay between queries
                
    except ConnectionRefusedError:
        logger.error("Could not connect to the news server. Is it running?")
    except Exception as e:
        logger.error(f"Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_news_query())
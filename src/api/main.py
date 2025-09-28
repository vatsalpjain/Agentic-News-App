from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
import logging
from ..pipeline import NewsPipeline
from ..embedding import EmbeddingManager
from ..vectorstore import VectorStore

# Initialize logging
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="NewsRAG API",
    description="AI-powered news aggregation and analysis API",
    version="1.0.0"
)

# CORS middleware setup for React frontend
# Replace localhost:3000 with your React app's URL in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response validation
class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

class NewsResponse(BaseModel):
    query: str
    summary: str
    articles: List[Dict[str, Any]]
    timestamp: str

# Global state
pipeline: Optional[NewsPipeline] = None

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    global pipeline
    try:
        embedding_manager = EmbeddingManager()
        vector_store = VectorStore()
        pipeline = NewsPipeline(vector_store, embedding_manager)
        # Initial refresh of news
        await pipeline.refresh_news()
        # Start background task for periodic refresh
        asyncio.create_task(periodic_refresh())
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

async def periodic_refresh():
    """Background task to refresh news periodically"""
    while True:
        try:
            await asyncio.sleep(3600)  # 1 hour
            await pipeline.refresh_news()
        except Exception as e:
            logger.error(f"Refresh failed: {e}")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "healthy", "service": "NewsRAG API"}

@app.post("/api/query", response_model=NewsResponse)
async def query_news(request: QueryRequest):
    """Query the news database"""
    try:
        results = await pipeline.query_news(request.query, request.top_k)
        return results
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            if data["type"] == "query":
                results = await pipeline.query_news(data["query"])
                await websocket.send_json({
                    "type": "query_result",
                    "data": results
                })
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()
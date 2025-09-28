import asyncio
import json
from typing import Dict, Any
import websockets
import logging
from ..pipeline import NewsPipeline

logger = logging.getLogger(__name__)

class NewsServer:
    def __init__(self, pipeline: NewsPipeline, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.pipeline = pipeline
        self.clients = set()
        self.news_cache: Dict[str, Any] = {}
        logger.info(f"Initialized NewsServer on {host}:{port}")

    async def register(self, websocket):
        self.clients.add(websocket)
        try:
            await self.send_cache(websocket)
            async for message in websocket:
                await self.process_message(websocket, message)
        finally:
            self.clients.remove(websocket)

    async def send_cache(self, websocket):
        if self.news_cache:
            await websocket.send(json.dumps({
                "type": "cache_update",
                "data": self.news_cache
            }))

    async def broadcast(self, message: Dict[str, Any]):
        if self.clients:
            await asyncio.gather(
                *[client.send(json.dumps(message)) for client in self.clients]
            )

    async def process_message(self, websocket, message):
        try:
            data = json.loads(message)
            if data["type"] == "query":
                query = data["query"]
                logger.info(f"Processing query: {query}")
                
                # Use the pipeline to process the query
                results = await self.pipeline.query_news(query)
                
                # Cache the results
                self.news_cache[query] = results
                
                # Send response
                response = {
                    "type": "query_result",
                    "data": results
                }
                await websocket.send(json.dumps(response))
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            error_response = {
                "type": "error",
                "data": str(e)
            }
            await websocket.send(json.dumps(error_response))

    async def start(self):
        logger.info(f"Starting NewsServer on {self.host}:{self.port}")
        async with websockets.serve(self.register, self.host, self.port):
            await asyncio.Future()  # run forever
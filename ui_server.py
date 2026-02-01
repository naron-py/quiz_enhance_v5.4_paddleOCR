"""
WebSocket server for Electron UI communication
"""
import asyncio
import websockets
import json
import logging

logger = logging.getLogger(__name__)

# Global singleton instance
current_server = None

class UIServer:
    def __init__(self, port=8765):
        self.port = port
        self.clients = set()
        self.server = None
        
    async def register(self, websocket):
        self.clients.add(websocket)
        logger.info(f"UI client connected. Total clients: {len(self.clients)}")
        
    async def unregister(self, websocket):
        self.clients.remove(websocket)
        logger.info(f"UI client disconnected. Total clients: {len(self.clients)}")
        
    async def broadcast(self, message):
        """Send message to all connected clients"""
        if self.clients:
            await asyncio.gather(
                *[client.send(message) for client in self.clients],
                return_exceptions=True
            )
    
    async def handler(self, websocket):
        await self.register(websocket)
        try:
            async for message in websocket:
                # Handle incoming messages from UI if needed
                pass
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self.unregister(websocket)
    
    async def start_server(self):
        # Store the loop running this server
        self.loop = asyncio.get_running_loop()
        self.server = await websockets.serve(
            self.handler,
            "localhost",
            self.port
        )
        logger.info(f"WebSocket server started on ws://localhost:{self.port}")
        
    def send_match_result(self, matched_choice, matched_answer, question, answers, score):
        """Send OCR match result to UI safely from any thread"""
        print("DEBUG: send_match_result called")
        data = {
            "matched_choice": matched_choice,
            "matched_answer": matched_answer,
            "question": question,
            "answers": answers,
            "score": score
        }
        message = json.dumps(data)
        
        # Schedule broadcast in event loop correctly from other threads
        if hasattr(self, 'loop') and self.loop and self.loop.is_running():
            print(f"DEBUG: Scheduling broadcast to {len(self.clients)} clients on thread loop")
            asyncio.run_coroutine_threadsafe(self.broadcast(message), self.loop)
        else:
            print("DEBUG: Event loop issue - cannot broadcast")
            logger.warning("Event loop not running, cannot broadcast message")

import os
import sys
import json
import base64
import argparse
import asyncio
from io import BytesIO
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from dotenv import load_dotenv
from loguru import logger
from gtts import gTTS

from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.openai import OpenAILLMService
from pipecat.frames.frames import LLMMessagesFrame, TextFrame

# Load environment
load_dotenv(override=True)

# Setup logging
logger.add(sys.stderr, level="DEBUG")

# FastAPI instance
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI API Key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is required in .env")

# WebSocket connections
connected_clients: set[WebSocket] = set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Server starting...")
    yield
    logger.info("Server shutting down...")


@app.get("/", response_class=HTMLResponse)
async def get_root():
    file_path = os.path.join(os.path.dirname(__file__), "index.html")
    return FileResponse(file_path)


async def synthesize_audio(text: str) -> str:
    """Generate audio from text using gTTS"""
    try:
        tts = gTTS(text=text, lang='en')
        audio_buffer = BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        audio_b64 = base64.b64encode(audio_buffer.read()).decode()
        return audio_b64
    except Exception as e:
        logger.error(f"Audio synthesis error: {e}")
        return ""


async def process_with_llm(user_message: str, conversation_history: list) -> str:
    """Process message using pipecat OpenAI service correctly."""
    try:
        logger.info(f"Processing with pipecat: {user_message}")
        
        # Create conversation messages including history
        all_messages = []
        
        # Add conversation history
        for msg in conversation_history:
            all_messages.append(msg)
        
        # Add current user message
        all_messages.append({"role": "user", "content": user_message})
        
        # Create LLM service with proper initialization
        llm_service = OpenAILLMService(
            api_key=api_key,
            model="gpt-3.5-turbo"
        )
        
        # Access the underlying OpenAI client from the pipecat service
        # The pipecat OpenAILLMService creates an AsyncOpenAI client internally
        if hasattr(llm_service, '_client'):
            client = llm_service._client
        else:
            # Fallback - create our own client if the internal one isn't accessible
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=api_key)
        
        # Make the chat completion call
        chat_completion = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=all_messages,
            max_tokens=150,
            temperature=0.7
        )
        
        response_text = chat_completion.choices[0].message.content
        
        logger.info(f"LLM Response: {response_text}")
        return response_text
        
    except Exception as e:
        logger.error(f"LLM processing error: {e}")
        return "I'm having trouble processing your request right now. Please try again."


@app.websocket("/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    
    # Conversation history for this session
    conversation_history = []
    
    try:
        # Send keepalive ping every 30 seconds
        async def keepalive():
            while True:
                try:
                    await asyncio.sleep(30)
                    await websocket.ping()
                except:
                    break
        
        keepalive_task = asyncio.create_task(keepalive())
        
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "voice_chunk":
                # Display voice chunks in real-time
                chunk_text = message.get("content", "")
                if chunk_text:
                    await websocket.send_json({"type": "voice_chunk", "content": chunk_text})

            elif message["type"] == "stop_recording":
                logger.info("Voice recording stopped")
                transcribed_text = message.get("content", "").strip()
                
                if not transcribed_text:
                    await websocket.send_json({"type": "error", "content": "No speech detected. Please try again."})
                    continue

                logger.info(f"Processing transcribed text: {transcribed_text}")
                
                # Process with OpenAI LLM using pipecat
                try:
                    response_text = await process_with_llm(transcribed_text, conversation_history)
                    
                    # Add to conversation history
                    conversation_history.append({"role": "user", "content": transcribed_text})
                    conversation_history.append({"role": "assistant", "content": response_text})

                    # Send text response
                    await websocket.send_json({"type": "message", "content": response_text})

                    # Generate and send audio
                    audio_b64 = await synthesize_audio(response_text)
                    if audio_b64:
                        await websocket.send_json({"type": "audio", "content": audio_b64})
                    else:
                        await websocket.send_json({"type": "error", "content": "Audio synthesis failed"})
                
                except Exception as e:
                    logger.error(f"LLM processing error: {e}")
                    await websocket.send_json({"type": "error", "content": "Failed to process your message. Please try again."})

            elif message["type"] == "text":
                # Handle text input
                user_text = message["content"].strip()
                if user_text:
                    try:
                        response_text = await process_with_llm(user_text, conversation_history)
                        
                        # Add to conversation history
                        conversation_history.append({"role": "user", "content": user_text})
                        conversation_history.append({"role": "assistant", "content": response_text})

                        # Send text response
                        await websocket.send_json({"type": "message", "content": response_text})

                        # Generate and send audio
                        audio_b64 = await synthesize_audio(response_text)
                        if audio_b64:
                            await websocket.send_json({"type": "audio", "content": audio_b64})
                    
                    except Exception as e:
                        logger.error(f"Text processing error: {e}")
                        await websocket.send_json({"type": "error", "content": "Failed to process your message."})

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        connected_clients.discard(websocket)
        keepalive_task.cancel()


@app.get("/status")
async def get_status():
    return {"status": "running", "clients": len(connected_clients)}


if __name__ == "__main__":
    import uvicorn
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=os.getenv("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("FAST_API_PORT", 8000)))
    parser.add_argument("--reload", action="store_true")
    config = parser.parse_args()

    uvicorn.run(
        "main:app",
        host=config.host,
        port=config.port,
        reload=config.reload,
    )

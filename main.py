# main.py

import uvicorn
import uuid
from typing import Any, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent import graph # Import your compiled graph variable

app = FastAPI(title="Diya AI Agent")

# Fixes CORS issues by allowing all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request and response validation
class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: Any
    thread_id: str

# API endpoint to interact with the agent
@app.post("/chat", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest):
    # Use existing thread_id or create a new one for a new conversation
    thread_id = request.thread_id or str(uuid.uuid4())
    
    # Set up the configuration for the stateful graph
    config = {"configurable": {"thread_id": thread_id}}
    inputs = {"query": [("user", request.message)]}
    
    # Invoke the graph
    final_state = graph.invoke(inputs, config)
    
    # Extract the last AI message from the final state
    agent_response = final_state["answer"]
    
    return ChatResponse(response=agent_response, thread_id=thread_id)

# Root endpoint for health check
@app.get("/")
def read_root():
    return {"status": "ok"}

# Run the server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
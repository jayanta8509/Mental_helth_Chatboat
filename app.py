"""
FastAPI Mental Health Chatbot API
Provides REST endpoints for mental health support using RAG with Pinecone

Endpoints:
- POST /chat - Main chat endpoint with user_id and query
- GET /health - Health check endpoint
- POST /chat/agent - Chat with agent mode for complex queries
- DELETE /chat/{user_id} - Clear conversation history for a user
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from contextlib import asynccontextmanager
import logging
import os
from datetime import datetime
import uvicorn

# Import our RAG functions
from rag import initialize_rag_system, get_response, get_conversation_summary, clear_conversation

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global system state
system_initialized = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    global system_initialized
    try:
        logger.info("🚀 Starting Mental Health RAG Chatbot API...")
        initialize_rag_system()
        system_initialized = True
        logger.info("✅ Mental Health RAG System initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize system: {e}")
        system_initialized = False
        # Don't raise here - let the API start but handle errors in endpoints
    
    yield
    
    # Cleanup on shutdown
    logger.info("🛑 Shutting down Mental Health RAG API...")

# Initialize FastAPI app
app = FastAPI(
    title="Mental Health RAG Chatbot API",
    description="AI-powered mental health support using Retrieval Augmented Generation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class ChatRequest(BaseModel):
    user_id: str = Field(..., description="Unique identifier for the user", min_length=1, max_length=100)
    query: str = Field(..., description="User's message/question", min_length=1, max_length=2000)
    use_agent: bool = Field(False, description="Whether to use agent mode for complex queries")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "user_id": "user123",
                "query": "I'm feeling anxious about my job interview tomorrow",
                "use_agent": False
            }
        }
    }

class ChatResponse(BaseModel):
    user_id: str = Field(..., description="User identifier")
    query: str = Field(..., description="Original user query")
    response: str = Field(..., description="AI assistant response")
    mode: str = Field(..., description="Processing mode used (counselor/agent/error)")
    crisis_detected: bool = Field(..., description="Whether crisis indicators were detected")
    data_source: Optional[str] = Field(None, description="Data source used for retrieval (csv/pdf/both/none)")
    timestamp: float = Field(..., description="Unix timestamp of response")
    error: Optional[str] = Field(None, description="Error message if any")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "user_id": "user123",
                "query": "I'm feeling anxious about my job interview tomorrow",
                "response": "I understand that you're feeling anxious about your upcoming job interview...",
                "mode": "counselor",
                "crisis_detected": False,
                "data_source": "csv",
                "timestamp": 1640995200.0,
                "error": None
            }
        }
    }

class HealthResponse(BaseModel):
    status: str
    message: str
    timestamp: float
    pinecone_connected: bool
    models_loaded: bool

class ConversationClearResponse(BaseModel):
    user_id: str
    message: str
    timestamp: float
    
class ConversationSummaryResponse(BaseModel):
    user_id: str
    summary: str
    timestamp: float


# Health check endpoint
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint to verify API and system status
    """
    try:
        status = "healthy" if system_initialized else "degraded"
        
        return HealthResponse(
            status=status,
            message="Mental Health RAG API is running",
            timestamp=datetime.now().timestamp(),
            pinecone_connected=system_initialized,
            models_loaded=system_initialized
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            message=f"Health check failed: {str(e)}",
            timestamp=datetime.now().timestamp(),
            pinecone_connected=False,
            models_loaded=False
        )

# Main chat endpoint
@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint for mental health support
    
    **Parameters:**
    - **user_id**: Unique identifier for the user (maintains conversation context)
    - **query**: User's message or question
    - **use_agent**: Set to true for complex queries requiring research (default: false)
    
    **Features:**
    - Automatic crisis detection and emergency resources
    - User-specific conversation memory
    - Evidence-based mental health guidance
    - Professional boundary maintenance
    """
    if not system_initialized:
        raise HTTPException(
            status_code=503, 
            detail="Mental health system is not initialized. Please check server logs."
        )
    
    try:
        logger.info(f"Processing chat request for user {request.user_id}")
        
        # Get response using function-based approach
        response_data = get_response(
            message=request.query,
            user_id=request.user_id,
            use_agent=request.use_agent
        )
        
        # Log crisis detection for monitoring
        if response_data.get("crisis_detected", False):
            logger.warning(f"Crisis detected for user {request.user_id}")
        
        return ChatResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing your message: {str(e)}"
        )

# Agent mode endpoint
@app.post("/chat/agent", response_model=ChatResponse, tags=["Chat"])
async def chat_agent_endpoint(request: ChatRequest):
    """
    Chat endpoint with agent mode enabled for complex queries
    
    This endpoint automatically uses agent mode for multi-step retrieval
    and comprehensive responses. Best for:
    - Research questions about mental health
    - Complex diagnostic information requests
    - Treatment option comparisons
    """
    # Force agent mode
    request.use_agent = True
    return await chat_endpoint(request)


# Run the application
if __name__ == "__main__":
    # Get port from environment or default to 8000
    port = int(os.getenv("PORT", 8000))
    
    # Run with uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=True,  # Set to False in production
        log_level="info"
    )

from pydantic import BaseModel, Field, validator
from typing import List, Dict
import time
from fastapi import FastAPI, HTTPException, Depends, Request
from ai_agent import get_response_from_ai_agent

# setup pydantic model (schema validation)
class RequestState(BaseModel):
    model_name: str
    model_provider: str
    system_prompt: str = Field(..., min_length=1, max_length=4000)
    messages: List[str] = Field(..., min_items=1)
    allow_search: bool

    @validator('system_prompt')
    def validate_system_prompt(cls, v):
        # Check for potentially harmful content
        harmful_keywords = ["rm -rf", "format", "hack", "exploit", "sudo", "chmod 777"]
        for keyword in harmful_keywords:
            if keyword.lower() in v.lower():
                raise ValueError(f"System prompt contains potentially harmful content: {keyword}")
        return v

# setup ai agent from frontend request
ALLOWED_MODELS_NAME = [
    "llama-3.3-70b-versatile",
    "llama-70b-8192",
    "mixtral-8x7b-32768",
    "qwen-qwq-32b",
    "deepseek-r1-distill-llama-70b",
    "gpt-4o-mini",
]

ALLOWED_PROVIDERS = ["openai", "anthropic", "together", "groq", "mistral"]

# Rate limiting
REQUEST_HISTORY = {}
MAX_REQUESTS_PER_MINUTE = 10
REQUEST_WINDOW = 60  # seconds

app = FastAPI(
    title="Langchain AI Agent", description="AI agent for langchain", version="0.1"
)

# Rate limiting middleware
async def check_rate_limit(request: Request):
    client_ip = request.client.host
    current_time = time.time()
    
    # Initialize or clean up old requests
    if client_ip not in REQUEST_HISTORY:
        REQUEST_HISTORY[client_ip] = []
    
    # Remove requests older than the window
    REQUEST_HISTORY[client_ip] = [t for t in REQUEST_HISTORY[client_ip] 
                                 if current_time - t < REQUEST_WINDOW]
    
    # Check if rate limit exceeded
    if len(REQUEST_HISTORY[client_ip]) >= MAX_REQUESTS_PER_MINUTE:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please try again later.")
    
    # Add current request timestamp
    REQUEST_HISTORY[client_ip].append(current_time)
    
    return True

@app.get("/")
def read_root():
    return {"message": "Welcome to Langchain AI Agent."}

@app.post("/chat")
async def chat_endpoint(request: RequestState, rate_limit_check: bool = Depends(check_rate_limit)):
    """
    API Endpoint to interact with the Chatbot using LangGraph and Search Tools.
    It Dynamically selects the model specified in the request and returns the response.
    """

    if request.model_name not in ALLOWED_MODELS_NAME:
        raise HTTPException(status_code=400, detail=f"Invalid Model Name. Allowed models: {', '.join(ALLOWED_MODELS_NAME)}")
    
    if request.model_provider not in ALLOWED_PROVIDERS:
        raise HTTPException(status_code=400, detail=f"Invalid Provider. Allowed providers: {', '.join(ALLOWED_PROVIDERS)}")
    
    # Check message length
    for msg in request.messages:
        if len(msg) > 8000:
            raise HTTPException(status_code=400, detail="Message too long. Maximum length is 8000 characters.")
    
    llm_id = request.model_name
    query = request.messages
    allow_search = request.allow_search
    system_prompt = request.system_prompt
    provider = request.model_provider

    try:
        # * Create AI Agent and get response form it
        response = get_response_from_ai_agent(
            llm_id, query, allow_search, system_prompt, provider
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

# run app & explore swagger UI docs
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
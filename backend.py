from pydantic import BaseModel
from typing import List

from fastapi import FastAPI
from ai_agent import get_response_from_ai_agent

# setup pydantic model (schema validation)
class RequestState(BaseModel):
    model_name: str
    model_provider: str
    system_prompt: str
    messages: List[str]
    allow_search: bool

# setup ai agent from frontend request
ALLOWED_MODELS_NAME = [
    "llama-3.3-70b-versatile",
    "llama-70b-8192",
    "mixtral-8x7b-32768",
    "qwen-qwq-32b",
    "deepseek-r1-distill-llama-70b",
    "gpt-4o-mini",
]

app = FastAPI(
    title="Langchain AI Agent", description="AI agent for langchain", version="0.1"
)

@app.get("/")
def read_root():
    return {"message": "Welcome to Langchain AI Agent."}

@app.post("/chat")
def chat_endpoint(request: RequestState):
    """
    API Endpoint to interact with the Chatbot using LangGraph and Search Tools.
    It Dynamically selects the model specified in the request and returns the response.
    """

    if request.model_name not in ALLOWED_MODELS_NAME:
        return {"error": "Invalid Model Name. Kindly Select a Valid AI Model."}

    llm_id = request.model_name
    query = request.messages
    allow_search = request.allow_search
    system_prompt = request.system_prompt
    provider = request.model_provider

    # * Create AI Agent and get response form it
    response = get_response_from_ai_agent(
        llm_id, query, allow_search, system_prompt, provider
    )

    return response

# run app & explore swagger UI docs
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
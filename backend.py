from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional
import time
from fastapi import FastAPI, HTTPException, Depends, Request
from ai_agent import get_response_from_ai_agent
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate


# setup pydantic model (schema validation)
class RequestState(BaseModel):
    model_name: str
    model_provider: str
    system_prompt: str = Field(..., min_length=1, max_length=4000)
    messages: List[str] = Field(..., min_items=1)
    allow_search: bool


class CodeAssistantRequest(BaseModel):
    code: str
    task_type: str
    language: str

    @field_validator("task_type")
    def validate_task_type(cls, v):
        # Check for valid task types
        valid_tasks = [
            "Code Analysis",
            "Code Generation",
            "Code Optimization",
            "Bug Finding",
        ]
        if v not in valid_tasks:
            raise ValueError(
                f"Invalid task type. Must be one of: {', '.join(valid_tasks)}"
            )
        return v


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
    REQUEST_HISTORY[client_ip] = [
        t for t in REQUEST_HISTORY[client_ip] if current_time - t < REQUEST_WINDOW
    ]

    # Check if rate limit exceeded
    if len(REQUEST_HISTORY[client_ip]) >= MAX_REQUESTS_PER_MINUTE:
        raise HTTPException(
            status_code=429, detail="Rate limit exceeded. Please try again later."
        )

    # Add current request timestamp
    REQUEST_HISTORY[client_ip].append(current_time)

    return True


# Define code assistant prompts
ANALYSIS_PROMPT = """
Analyze the following {language} code:
{code}

Provide a detailed analysis including:
1. Code structure and organization
2. Potential issues or concerns
3. Best practices adherence
4. Suggestions for improvement
"""

GENERATION_PROMPT = """
Generate {language} code based on the following description:
{code}

Requirements:
1. Follow best practices and conventions
2. Include comments explaining the code
3. Ensure code is efficient and maintainable
4. Add error handling where appropriate
"""

OPTIMIZATION_PROMPT = """
Optimize the following {language} code:
{code}

Focus on:
1. Performance improvements
2. Memory efficiency
3. Code readability
4. Best practices implementation
"""

BUG_FINDING_PROMPT = """
Analyze the following {language} code for potential bugs:
{code}

Provide:
1. Identified bugs or issues
2. Root cause analysis
3. Suggested fixes
4. Prevention recommendations
"""

# Define allowed models and providers
ALLOWED_MODELS_NAME = [
    "llama-3.3-70b-versatile",
    "llama-70b-8192",
    "mixtral-8x7b-32768",
    "qwen-qwq-32b",
    "deepseek-r1-distill-llama-70b",
    "gpt-4o-mini",
]
ALLOWED_PROVIDERS = ["Groq", "OpenAI"]


@app.get("/")
def read_root():
    return {"message": "Welcome to Langchain AI Agent."}


@app.post("/code-assistant")
async def code_assistant(request: CodeAssistantRequest):
    try:
        # Select appropriate prompt based on task type
        prompt_template = {
            "Code Analysis": ANALYSIS_PROMPT,
            "Code Generation": GENERATION_PROMPT,
            "Code Optimization": OPTIMIZATION_PROMPT,
            "Bug Finding": BUG_FINDING_PROMPT,
        }.get(request.task_type)

        if not prompt_template:
            raise HTTPException(
                status_code=400, detail=f"Invalid task type: {request.task_type}"
            )

        # Create prompt with user's code and language
        prompt = ChatPromptTemplate.from_template(prompt_template)

        # Initialize LLM (using Groq for better performance with code)
        llm = ChatGroq(model="llama-3.3-70b-versatile")

        # Generate response
        chain = prompt | llm
        response = chain.invoke({"code": request.code, "language": request.language})

        # Extract the assistant's message
        assistant_message = (
            response.content if hasattr(response, "content") else str(response)
        )

        return assistant_message

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat_endpoint(
    request: RequestState, rate_limit_check: bool = Depends(check_rate_limit)
):
    """
    API Endpoint to interact with the Chatbot using LangGraph and Search Tools.
    It Dynamically selects the model specified in the request and returns the response.
    """

    if request.model_name not in ALLOWED_MODELS_NAME:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid Model Name. Allowed models: {', '.join(ALLOWED_MODELS_NAME)}",
        )

    if request.model_provider not in ALLOWED_PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid Provider. Allowed providers: {', '.join(ALLOWED_PROVIDERS)}",
        )

    # Check message length
    for msg in request.messages:
        if len(msg) > 8000:
            raise HTTPException(
                status_code=400,
                detail="Message too long. Maximum length is 8000 characters.",
            )

    llm_id = request.model_name
    query = request.messages
    allow_search = request.allow_search
    system_prompt = request.system_prompt
    provider = request.model_provider

    try:
        # Create AI Agent and get response from it
        response = get_response_from_ai_agent(
            llm_id, query, allow_search, system_prompt, provider
        )
        return response
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing request: {str(e)}"
        )


# run app & explore swagger UI docs
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)

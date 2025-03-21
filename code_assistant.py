from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Define request models
class CodeAssistantRequest(BaseModel):
    code: str
    task_type: str
    language: str

# Initialize FastAPI app
app = FastAPI()

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

@app.post("/code-assistant")
async def code_assistant(request: CodeAssistantRequest):
    try:
        # Select appropriate prompt based on task type
        prompt_template = {
            "Code Analysis": ANALYSIS_PROMPT,
            "Code Generation": GENERATION_PROMPT,
            "Code Optimization": OPTIMIZATION_PROMPT,
            "Bug Finding": BUG_FINDING_PROMPT
        }.get(request.task_type)
        
        if not prompt_template:
            raise HTTPException(status_code=400, detail=f"Invalid task type: {request.task_type}")
        
        # Create prompt with user's code and language
        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        # Initialize LLM (using Groq for better performance with code)
        llm = ChatGroq(model="llama-3.3-70b-versatile")
        
        # Generate response
        chain = prompt | llm
        response = chain.invoke({"code": request.code, "language": request.language})
        
        # Extract the assistant's message
        assistant_message = response.content if hasattr(response, 'content') else str(response)
        
        return assistant_message
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
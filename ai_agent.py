# * step 1: setup api keys for groq and tavily
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

from langgraph.prebuilt import create_react_agent
from langchain_core.messages.ai import AIMessage

load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# * step 2: setup llm & tools
openai_llm = ChatOpenAI(model="gpt-4o-mini")
groq_llm = ChatGroq(model="llama-3.3-70b-versatile")


# * step 3: setup ai agent with search tool functionaility
system_prompt = "Act as an AI chatbot who is smart and friendly. You can search the web for information. You can also answer questions and have conversations with users."

def get_response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider):

    if provider == "Groq":
        llm = ChatGroq(model=llm_id)
    elif provider == "OpenAI":
        llm = ChatOpenAI(model=llm_id)

    tools = [TavilySearchResults(max_results=3)] if allow_search else []

    agent = create_react_agent(
        model=llm,
        tools=tools,
        state_modifier=system_prompt,
    )

    state = {"messages": query}

    response = agent.invoke(state)
    messages = response.get("messages")

    ai_messages = [
        message.content for message in messages if isinstance(message, AIMessage)
    ]
    return ai_messages[-1]
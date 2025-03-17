# step 1: setup ui with streamlit (model_name, model_provider, system_prompt, messages, allow_search)
import streamlit as st
import requests

st.set_page_config(page_title="Langchain AI Agent", page_icon="🤖", layout="centered")
st.title("AI Agent - Chatbot")
st.write("Create and Interact with AI Chatbot using LangGraph and Search Tools.")

# system_prompt = st.text_area("System Prompt", "Act as an AI chatbot who is smart and friendly. You can search the web for information. You can also answer questions and have conversations with users.")
system_prompt = st.text_area(
    "Define your Ai Agent:", height=100, placeholder="Type your system prompt here..."
)

MODEL_NAME_GROQ = [
    "llama-3.3-70b-versatile",
    "llama-70b-8192",
    "mixtral-8x7b-32768",
    "qwen-qwq-32b",
    "deepseek-r1-distill-llama-70b",
]
MODEL_NAME_OPENAI = ["gpt-4o-mini"]

provider = st.radio("Select AI Model Provider", ["Groq", "OpenAI"])

if provider == "Groq":
    model_name = st.selectbox("Select Groq Model", MODEL_NAME_GROQ)
else:
    model_name = st.selectbox("Select OpenAI Model", MODEL_NAME_OPENAI)

allow_web_search = st.checkbox("Allow Web Search")

user_query = st.text_area("Enter Your Query", height=150, placeholder="Ask Anything...")

API_URL = "http://127.0.0.1:8000/chat"

if st.button("Ask Agent"):
    if user_query.strip():
        # step 2: connect with backend api (chat_endpoint)
        payload = {
            "model_name": model_name,
            "model_provider": provider,
            "system_prompt": system_prompt,
            "messages": [user_query],
            "allow_search": allow_web_search,
        }

        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            response_data = response.json()

            if "error" in response_data:
                st.error(response_data["error"])
            else:
                st.subheader("Agent Response")
                st.markdown(f"**Final Response:** {response_data}")


# import streamlit as st
# import requests
# import os

# st.set_page_config(page_title="Langchain AI Agent", page_icon="🤖", layout="centered")
# st.title("AI Agent - Chatbot")
# st.write("Create and Interact with AI Chatbot using LangGraph and Search Tools.")

# # system_prompt area
# system_prompt = st.text_area(
#     "Define your AI Agent:", 
#     height=100, 
#     placeholder="Type your system prompt here..."
# )

# # Model selection
# MODEL_NAME_GROQ = [
#     "llama-3.3-70b-versatile",
#     "llama-70b-8192",
#     "mixtral-8x7b-32768",
#     "qwen-qwq-32b",
#     "deepseek-r1-distill-llama-70b",
# ]
# MODEL_NAME_OPENAI = ["gpt-4o-mini"]

# provider = st.radio("Select AI Model Provider", ["Groq", "OpenAI"])

# if provider == "Groq":
#     model_name = st.selectbox("Select Groq Model", MODEL_NAME_GROQ)
#     if st.checkbox("Allow Web Search"):
#         st.warning("Note: Web search is currently only fully supported with OpenAI models. When using Groq models, the agent will rely on its built-in knowledge.")
#         allow_web_search = False
#     else:
#         allow_web_search = False
# else:
#     model_name = st.selectbox("Select OpenAI Model", MODEL_NAME_OPENAI)
#     allow_web_search = st.checkbox("Allow Web Search")

# user_query = st.text_area("Enter Your Query", height=150, placeholder="Ask Anything...")

# API_URL = "http://127.0.0.1:8000/chat"

# # Check API keys
# if provider == "Groq" and not os.environ.get("GROQ_API_KEY"):
#     st.warning("GROQ_API_KEY environment variable is not set. Please set it before making requests.")

# if provider == "OpenAI" and not os.environ.get("OPENAI_API_KEY"):
#     st.warning("OPENAI_API_KEY environment variable is not set. Please set it before making requests.")

# if allow_web_search and not os.environ.get("TAVILY_API_KEY"):
#     st.warning("TAVILY_API_KEY environment variable is not set. Web search functionality may not work.")

# if st.button("Ask Agent"):
#     if user_query.strip():
#         with st.spinner("AI Agent is thinking..."):
#             # step 2: connect with backend api (chat_endpoint)
#             payload = {
#                 "model_name": model_name,
#                 "model_provider": provider,
#                 "system_prompt": system_prompt,
#                 "messages": [user_query],
#                 "allow_search": allow_web_search,
#             }

#             try:
#                 response = requests.post(API_URL, json=payload)
                
#                 if response.status_code == 200:
#                     response_data = response.json()
                    
#                     if isinstance(response_data, dict) and "error" in response_data:
#                         st.error(response_data["error"])
#                     else:
#                         st.subheader("Agent Response")
#                         st.markdown(response_data)
#                 else:
#                     st.error(f"Error: Received status code {response.status_code} from server.")
#                     st.code(response.text)
#             except requests.exceptions.ConnectionError:
#                 st.error("Could not connect to the backend server. Make sure it's running at http://127.0.0.1:8000")
#     else:
#         st.warning("Please enter a query first!")
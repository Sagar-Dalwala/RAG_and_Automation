import streamlit as st
import requests
from auth_db import init_db, create_user, verify_user, save_chat_history, get_user_chat_history

# Initialize the database
init_db()

# Initialize session state
if 'user_id' not in st.session_state:
    st.session_state.user_id = None

# Initialize selected chat state
if 'selected_chat' not in st.session_state:
    st.session_state.selected_chat = None

st.set_page_config(page_title="Langchain AI Agent", page_icon="🤖", layout="wide")

# Sidebar for authentication and chat history
with st.sidebar:
    if st.session_state.user_id is None:
        st.title("Login / Sign Up")
        tab1, tab2 = st.tabs(["Login", "Sign Up"])
        
        with tab1:
            login_username = st.text_input("Username", key="login_username")
            login_password = st.text_input("Password", type="password", key="login_password")
            if st.button("Login"):
                if login_username and login_password:
                    user_id = verify_user(login_username, login_password)
                    if user_id:
                        st.session_state.user_id = user_id
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
                else:
                    st.warning("Please enter both username and password")
        
        with tab2:
            new_username = st.text_input("Username", key="new_username")
            new_password = st.text_input("Password", type="password", key="new_password")
            if st.button("Sign Up"):
                if new_username and new_password:
                    if create_user(new_username, new_password):
                        st.success("Account created successfully! Please login.")
                    else:
                        st.error("Username already exists")
                else:
                    st.warning("Please enter both username and password")
    else:
        st.title("Chat History")
        if st.button("Logout"):
            st.session_state.user_id = None
            st.rerun()
        
        # Display chat history
        history = get_user_chat_history(st.session_state.user_id)
        for query, response, model_name, model_provider, timestamp in history:
            # Create a unique key for each chat
            chat_key = f"{timestamp}-{model_name}"
            
            # Create a clickable button for each chat
            if st.button(f"Chat at {timestamp}", key=chat_key):
                # Store the selected chat in session state
                st.session_state.selected_chat = {
                    "query": query,
                    "response": response,
                    "model_name": model_name,
                    "model_provider": model_provider,
                    "timestamp": timestamp
                }
                st.rerun()
                
            # Still show the preview in an expander
            with st.expander("Preview"):
                st.write(f"**Model:** {model_provider} - {model_name}")
                st.write(f"**You:** {query}")
                st.write(f"**AI:** {response}")

# Main chat interface (only shown when logged in)
if st.session_state.user_id is not None:
    # Check if a chat is selected to display
    if st.session_state.selected_chat is not None:
        # Display the selected chat in the main container
        st.title("Selected Chat")
        
        selected_chat = st.session_state.selected_chat
        st.subheader(f"Chat from {selected_chat['timestamp']}")
        st.write(f"**Model:** {selected_chat['model_provider']} - {selected_chat['model_name']}")
        
        st.markdown("### Your Query")
        st.write(selected_chat['query'])
        
        st.markdown("### AI Response")
        st.markdown(selected_chat['response'])
        
        # Button to go back to the chat interface
        if st.button("Back to Chat Interface"):
            st.session_state.selected_chat = None
            st.rerun()
    else:
        # Regular chat interface
        st.title("AI Agent - Chatbot")
        st.write("Create and Interact with AI Chatbot using LangGraph and Search Tools.")

        system_prompt = st.text_area(
            "Define your AI Agent:", height=100, placeholder="Type your system prompt here..."
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
                with st.spinner("AI Agent is thinking..."):
                    payload = {
                        "model_name": model_name,
                        "model_provider": provider,
                        "system_prompt": system_prompt,
                        "messages": [user_query],
                        "allow_search": allow_web_search,
                    }

                    try:
                        response = requests.post(API_URL, json=payload)
                        
                        if response.status_code == 200:
                            response_data = response.json()
                            
                            if isinstance(response_data, dict) and "error" in response_data:
                                st.error(response_data["error"])
                            else:
                                st.subheader("Agent Response")
                                st.markdown(response_data)
                                
                                # Save to chat history
                                save_chat_history(
                                    st.session_state.user_id,
                                    user_query,
                                    response_data,
                                    model_name,
                                    provider
                                )
                        else:
                            st.error(f"Error: Received status code {response.status_code} from server.")
                            st.code(response.text)
                    except requests.exceptions.ConnectionError:
                        st.error("Could not connect to the backend server. Make sure it's running at http://127.0.0.1:8000")
            else:
                st.warning("Please enter a query first!")
else:
    st.info("Please login or sign up to start chatting with the AI Agent.")
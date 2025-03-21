import streamlit as st
import requests
from auth_db import init_db, create_user, verify_user, save_chat_history, get_user_chat_history
from rag_utils import process_input, answer_question
from session_manager import init_session, login_user, logout_user

# Set page config first before any other st commands
st.set_page_config(page_title="Langchain AI Agent", page_icon="🤖", layout="wide")

# Initialize the database
init_db()

# Initialize session state for user_id if not present
if 'user_id' not in st.session_state:
    st.session_state.user_id = None

# Now import and initialize session manager after basic session state is set up


# Initialize cookie manager in session state if not present
if 'cookie_manager' not in st.session_state:
    import extra_streamlit_components as stx
    st.session_state.cookie_manager = stx.CookieManager(key="unique_cookie_manager")

# Initialize the session
init_session()

# Initialize selected chat state
if 'selected_chat' not in st.session_state:
    st.session_state.selected_chat = None

# Initialize RAG state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

# Initialize previous input type state
if 'previous_input_type' not in st.session_state:
    st.session_state.previous_input_type = None

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
                    if login_user(login_username, login_password):
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
            logout_user()
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
            # with st.expander("Preview"):
            #     st.write(f"**Model:** {model_provider} - {model_name}")
            #     st.write(f"**You:** {query}")
            #     st.write(f"**AI:** {response}")

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

        # Create tabs for different functionalities
        tab1, tab2, tab3, tab4 = st.tabs(["AI Agent Chat", "RAG Document Q&A", "Code Assistant", "Web Automation"])

        with tab1:
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
        

        with tab2:
            # RAG interface
            st.title("RAG Document Q&A")
            st.write("Upload documents or provide links to ask questions about their content.")
            
            # Input type selection
            input_type = st.selectbox("Select Input Type", ["Link", "PDF", "Text", "DOCX", "TXT"])
            
            # Check if input type has changed and clear query input if it has
            if st.session_state.previous_input_type != input_type:
                if "query_input" in st.session_state:
                    st.session_state.query_input = ""
                st.session_state.previous_input_type = input_type
            
            # Different input methods based on selection
            if input_type == "Link":
                number_input = st.number_input("Number of Links", min_value=1, max_value=20, step=1, value=1)
                input_data = []
                for i in range(number_input):
                    url = st.text_input(f"URL {i+1}")
                    input_data.append(url)
            elif input_type == "Text":
                input_data = st.text_area("Enter the text", height=150)
            elif input_type == 'PDF':
                input_data = st.file_uploader("Upload a PDF file", type=["pdf"])
            elif input_type == 'TXT':
                input_data = st.file_uploader("Upload a text file", type=['txt'])
            elif input_type == 'DOCX':
                input_data = st.file_uploader("Upload a DOCX file", type=['docx', 'doc'])
            
            # Process and Clear buttons side by side
            # Create columns with a gap by using 3 columns and making the middle one empty
            col1, col2 = st.columns([1, 1])
            with col1:
                process_button = st.button("Process Document")
            with col2:
                clear_button = st.button("Clear Document")
                
            # Process button logic
            if process_button:
                if input_data:
                    with st.spinner("Processing document..."):
                        try:
                            vectorstore = process_input(input_type, input_data)
                            st.session_state.vectorstore = vectorstore
                            st.success("Document processed successfully!")
                        except Exception as e:
                            st.error(f"Error processing document: {str(e)}")
                else:
                    st.warning("Please provide input data first!")
            
            # Clear button logic
            if clear_button:
                st.session_state.vectorstore = None
                if "query_input" in st.session_state:
                    st.session_state.query_input = ""
                st.rerun()
            
            # Query section (only shown if vectorstore exists)
            if st.session_state.vectorstore is not None:
                st.subheader("Ask Questions About Your Document")
                query = st.text_input("Enter your question", key="query_input")
                
                if st.button("Get Answer"):
                    if query.strip():
                        with st.spinner("Generating answer..."):
                            try:
                                answer = answer_question(st.session_state.vectorstore, query)
                                st.subheader("Answer")
                                st.markdown(answer['result'])
                                
                                # Save to chat history with special model name
                                save_chat_history(
                                    st.session_state.user_id,
                                    query,
                                    answer['result'],
                                    "Meta-Llama-3-8B-Instruct",
                                    "RAG-HuggingFace"
                                )
                            except Exception as e:
                                st.error(f"Error generating answer: {str(e)}")
                    else:
                        st.warning("Please enter a question first!")
                        
                # Removed the duplicate Clear Document button since we now have it above
        with tab3:
            # Code Assistant interface
            st.title("Code Assistant")
            st.write("Get help with code analysis, generation, and best practices.")
            
            code_input = st.text_area(
                "Enter your code or describe what you want to create:",
                height=200,
                placeholder="Paste your code here or describe the code you want to generate..."
            )
            
            task_type = st.radio(
                "Select Task Type",
                ["Code Analysis", "Code Generation", "Code Optimization", "Bug Finding"]
            )
            
            programming_language = st.selectbox(
                "Select Programming Language",
                ["Python", "JavaScript", "Java", "C++", "Go", "Rust", "Other"]
            )
            
            if st.button("Process Code"):
                if code_input.strip():
                    with st.spinner("Processing code..."):
                        try:
                            # Prepare the payload for the code assistant API
                            payload = {
                                "code": code_input,
                                "task_type": task_type,
                                "language": programming_language
                            }
                            
                            # Call the code assistant API (to be implemented)
                            response = requests.post("http://127.0.0.1:8000/code-assistant", json=payload)
                            
                            if response.status_code == 200:
                                response_data = response.json()
                                
                                if isinstance(response_data, dict) and "error" in response_data:
                                    st.error(response_data["error"])
                                else:
                                    st.subheader("Assistant Response")
                                    st.markdown(response_data)
                                    
                                    # Save to chat history with special model name
                                    save_chat_history(
                                        st.session_state.user_id,
                                        code_input,
                                        response_data,
                                        "Code-Assistant",
                                        "CodeLLM"
                                    )
                            else:
                                st.error(f"Error: Received status code {response.status_code} from server.")
                                st.code(response.text)
                        except requests.exceptions.ConnectionError:
                            st.error("Could not connect to the backend server. Make sure it's running at http://127.0.0.1:8000")
                else:
                    st.warning("Please enter some code or description first!")
                        
                # Removed the duplicate Clear Document button since we now have it above
        
        with tab4:
            # Web Automation interface
            st.title("Web Automation")
            st.write("Browse and extract content from web pages using automated browser.")
            
            # Initialize web automation in session state if not present
            if 'web_automation' not in st.session_state:
                from web_automation import WebAutomation
                st.session_state.web_automation = WebAutomation()
            
            # URL input
            url = st.text_input("Enter URL to browse", placeholder="https://example.com")
            
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                if st.button("Browse"):
                    if url:
                        with st.spinner("Loading page..."):
                            result = st.session_state.web_automation.navigate_to_url(url)
                            if "error" in result:
                                st.error(f"Error: {result['error']}")
                            else:
                                st.success("Page loaded successfully!")
                                
                                # Extract and display content
                                content = st.session_state.web_automation.extract_page_content()
                                if "error" not in content:
                                    st.subheader("Page Content")
                                    st.write(f"Title: {content['title']}")
                                    st.write(f"Description: {content['description']}")
                                    
                                    with st.expander("View Full Content"):
                                        st.markdown(content['content'])
                                        
                                    with st.expander("View Links"):
                                        for link in content['links']:
                                            st.markdown(f"[{link['text']}]({link['href']})")
                                else:
                                    st.error(f"Error extracting content: {content['error']}")
                    else:
                        st.warning("Please enter a URL first!")
            
            with col2:
                if st.button("Take Screenshot"):
                    if url:
                        with st.spinner("Taking screenshot..."):
                            screenshot = st.session_state.web_automation.take_screenshot("page.png")
                            if "error" not in screenshot:
                                st.image("page.png", caption="Page Screenshot")
                            else:
                                st.error(f"Error taking screenshot: {screenshot['error']}")
                    else:
                        st.warning("Please browse a page first!")
            
            with col3:
                if st.button("Close Browser"):
                    st.session_state.web_automation.close_browser()
                    st.success("Browser closed successfully!")
            
            # Search functionality
            st.subheader("Search in Page")
            search_query = st.text_input("Enter search term")
            if st.button("Search"):
                if search_query:
                    results = st.session_state.web_automation.search_in_page(search_query)
                    if "error" not in results:
                        st.write(f"Found {len(results['results'])} matches:")
                        for result in results['results']:
                            st.markdown(f"**Match:** {result['content']}")
                            st.markdown(f"**Context:** {result['context']}")
                            st.markdown("---")
                    else:
                        st.error(f"Error searching: {results['error']}")
                else:
                    st.warning("Please enter a search term!")
            
            # Page navigation
            st.subheader("Page Navigation")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Scroll Down"):
                    st.session_state.web_automation.scroll_page("down")
            with col2:
                if st.button("Scroll Up"):
                    st.session_state.web_automation.scroll_page("up")
else:
    st.info("Please login or sign up to start chatting with the AI Agent.")
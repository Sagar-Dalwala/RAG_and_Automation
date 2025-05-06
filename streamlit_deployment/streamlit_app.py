import streamlit as st
import sqlite3
import os
import json
import uuid
import time
import hashlib
import base64
from datetime import datetime
from passlib.hash import bcrypt

# Set page config
st.set_page_config(page_title="AI Agent Platform", page_icon="🤖", layout="wide")

# Database setup functions
def get_db_connection():
    """Create a database connection"""
    conn = sqlite3.connect('chat_app.db', check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize the database with required tables"""
    conn = get_db_connection()
    conn.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    )
    ''')
    
    conn.execute('''
    CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        query TEXT NOT NULL,
        response TEXT NOT NULL,
        model_name TEXT NOT NULL,
        model_provider TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
    ''')
    
    conn.commit()
    conn.close()

def create_user(username, password):
    """Create a new user in the database"""
    try:
        conn = get_db_connection()
        # Hash the password
        hashed_password = bcrypt.hash(password)
        conn.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            (username, hashed_password)
        )
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        # Username already exists
        return False

def verify_user(username, password):
    """Verify user credentials and return user_id if valid"""
    conn = get_db_connection()
    user = conn.execute(
        "SELECT id, password FROM users WHERE username = ?",
        (username,)
    ).fetchone()
    conn.close()
    
    if user and bcrypt.verify(password, user['password']):
        return user['id']
    return None

def save_chat_history(user_id, query, response, model_name, model_provider):
    """Save chat history to the database"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_db_connection()
    conn.execute(
        "INSERT INTO chat_history (user_id, query, response, model_name, model_provider, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
        (user_id, query, response, model_name, model_provider, timestamp)
    )
    conn.commit()
    conn.close()

def get_user_chat_history(user_id):
    """Get chat history for a user"""
    conn = get_db_connection()
    history = conn.execute(
        "SELECT query, response, model_name, model_provider, timestamp FROM chat_history WHERE user_id = ? ORDER BY timestamp DESC",
        (user_id,)
    ).fetchall()
    conn.close()
    return [(row['query'], row['response'], row['model_name'], row['model_provider'], row['timestamp']) for row in history]

# Session management functions
def init_session():
    """Initialize session state variables"""
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    
    if 'username' not in st.session_state:
        st.session_state.username = None

def login_user(username, password):
    """Log in a user and update session state"""
    user_id = verify_user(username, password)
    if user_id:
        st.session_state.logged_in = True
        st.session_state.user_id = user_id
        st.session_state.username = username
        return True
    return False

def logout_user():
    """Log out a user and reset session state"""
    st.session_state.logged_in = False
    st.session_state.user_id = None
    st.session_state.username = None

# Initialize database
init_db()

# Initialize session
init_session()

# Mock AI response generation (simulating the backend)
def get_mock_ai_response(query, model_name, model_provider, system_prompt="", allow_search=False):
    """Generate a mock AI response - this simulates the backend for demo purposes"""
    responses = [
        f"This is a simulated response from {model_provider}'s {model_name} model.\n\nYour query was: '{query}'\n\nIn a real deployment, this would connect to actual AI models via API.",
        f"Thank you for your question! As a {model_name} model from {model_provider}, I would normally process this query through advanced AI algorithms.\n\nFor this demo, I'm providing a placeholder response. In production, you would see real AI-generated content here.",
        f"I'm a demo version of the {model_name} model. Your question about '{query}' would typically be processed through {model_provider}'s API.\n\nThis is placeholder text to show how the interface works in the deployed version."
    ]
    
    # Use the query to deterministically select a response (for demo consistency)
    response_index = sum(ord(c) for c in query) % len(responses)
    
    # Add search info if allowed
    if allow_search:
        search_info = "\n\n[Note: In the full version, I would have access to search the web for up-to-date information.]"
        return responses[response_index] + search_info
    
    return responses[response_index]

# Mock code assistant response
def get_mock_code_analysis(code, language, task_type):
    """Generate a mock code analysis response"""
    if task_type == "Code Analysis":
        return f"""## Code Analysis for {language}

Based on the provided code:

```{language}
{code}
```

### Structure and Organization
- The code appears to be {len(code.split('\n'))} lines long
- It seems to implement a basic functionality

### Potential Issues
- No obvious issues detected in this demo
- In a production environment, detailed analysis would be provided

### Best Practices
- Consider adding more comments
- Follow {language} standard naming conventions

### Improvement Suggestions
- Add error handling
- Consider refactoring for better maintainability
"""
    elif task_type == "Code Generation":
        js_example = """
// Generated example based on your request
function processData(input) {
  // Validate input
  if (!input) return null;
  
  // Process the data
  const result = input.map(item => {
    return {
      ...item,
      processed: true,
      timestamp: new Date().toISOString()
    };
  });
  
  return result;
}

// Example usage
const sampleData = [{"name": "Item 1"}, {"name": "Item 2"}];
const processed = processData(sampleData);
console.log(processed);
"""
        py_example = """
# Generated example based on your request
def process_data(input_data):
    # Validate input
    if not input_data:
        return None
    
    # Process the data
    result = []
    for item in input_data:
        processed_item = item.copy()
        processed_item['processed'] = True
        processed_item['timestamp'] = import_datetime().now().isoformat()
        result.append(processed_item)
    
    return result

# Example usage
sample_data = [{"name": "Item 1"}, {"name": "Item 2"}]
processed = process_data(sample_data)
print(processed)
"""
        # Choose example based on language
        if language.lower() in ["javascript", "typescript", "js", "ts"]:
            code_example = js_example
        else:
            code_example = py_example
            
        return f"""## Generated {language} Code

```{language}{code_example}```

This is a demonstration of code generation. In the full application, this would be customized to your specific requirements.
"""
    else:
        return f"This is a placeholder response for {task_type} of {language} code. In the full version, this would provide detailed {task_type.lower()} based on the AI model's analysis."

# Web automation mock response
def get_mock_web_automation(url, task):
    """Generate a mock web automation response"""
    return f"""## Web Automation Results for {url}

**Task:** {task}

### Process Summary
1. Connected to target website
2. Performed requested operation
3. Gathered results

### Results
This is a simulated response for the web automation task. In the full application, this would show actual results from performing the requested task on the specified website.

**Note:** The actual implementation would execute browser automation using tools like Selenium or Playwright to interact with the website.
"""

# MAIN APP UI
# Sidebar for authentication and chat history
with st.sidebar:
    if not st.session_state.logged_in:
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
        st.title(f"Welcome, {st.session_state.username}!")
        if st.button("Logout"):
            logout_user()
            st.rerun()
        
        st.title("Chat History")
        # Display chat history
        history = get_user_chat_history(st.session_state.user_id)
        if history:
            for i, (query, response, model_name, model_provider, timestamp) in enumerate(history):
                # Create a unique key for each chat
                chat_key = f"chat_{i}"
                
                # Create a clickable button for each chat
                with st.expander(f"Chat at {timestamp}"):
                    st.write(f"**Model:** {model_provider} - {model_name}")
                    st.write(f"**You:** {query}")
                    st.write(f"**AI:** {response}")
        else:
            st.info("No chat history yet. Try asking a question!")

# Main content area - only shown when logged in
if not st.session_state.logged_in:
    st.title("AI Agent Platform")
    st.write("Please login or sign up to access the AI Agent Platform.")
    
    st.markdown("""
    ### Features
    - AI Agent Chat
    - RAG Document Q&A
    - Code Assistant
    - Web Automation
    
    ### Getting Started
    1. Create an account or login
    2. Choose a functionality from the tabs
    3. Start interacting with the AI
    """)
else:
    # Create tabs for different functionalities
    tab1, tab2, tab3, tab4 = st.tabs(["AI Agent Chat", "RAG Document Q&A", "Code Assistant", "Web Automation"])

    with tab1:
        # Regular chat interface
        st.title("AI Agent - Chatbot")
        st.write("Create and Interact with AI Chatbot using LLM models.")

        system_prompt = st.text_area(
            "Define your AI Agent:", height=100, placeholder="Type your system prompt here...",
            value="You are a helpful AI assistant that provides accurate and concise information."
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

        if st.button("Ask Agent"):
            if user_query.strip():
                with st.spinner("AI Agent is thinking..."):
                    # Simulate a delay for realism
                    time.sleep(1.5)
                    
                    # Get mock response (in production, this would call the actual API)
                    response = get_mock_ai_response(
                        user_query, 
                        model_name, 
                        provider, 
                        system_prompt, 
                        allow_web_search
                    )
                    
                    st.subheader("Agent Response")
                    st.markdown(response)
                    
                    # Save to chat history
                    save_chat_history(
                        st.session_state.user_id,
                        user_query,
                        response,
                        model_name,
                        provider
                    )
            else:
                st.warning("Please enter a query first!")

    with tab2:
        # RAG Document Q&A interface
        st.title("RAG Document Q&A")
        st.write("Upload documents and ask questions about their content.")
        
        uploaded_file = st.file_uploader("Upload Document", type=["pdf", "txt", "docx"])
        
        if uploaded_file:
            st.success(f"File '{uploaded_file.name}' uploaded successfully for RAG processing!")
            
            user_question = st.text_area("Ask a question about the document:", height=100, placeholder="What does the document say about...?")
            
            if st.button("Get Answer"):
                if user_question.strip():
                    with st.spinner("Processing document and finding answer..."):
                        # Simulate processing time
                        time.sleep(2)
                        
                        # Mock response for RAG
                        rag_response = f"""Based on the document '{uploaded_file.name}', I found the following information:

This is a simulated RAG (Retrieval Augmented Generation) response that would normally extract relevant passages from your document and provide an accurate answer based on the document content.

In the full version, the system would:
1. Process and index your document
2. Retrieve relevant passages based on your question
3. Generate a comprehensive answer using those passages

For this demo, I'm providing a placeholder response to demonstrate the interface functionality.

Query: "{user_question}"
"""
                        
                        st.subheader("Document Answer")
                        st.markdown(rag_response)
                        
                        # Save to chat history
                        save_chat_history(
                            st.session_state.user_id,
                            f"[RAG] {user_question}",
                            rag_response,
                            "RAG System",
                            "Document QA"
                        )
                else:
                    st.warning("Please enter a question first!")
        else:
            st.info("Please upload a document to start asking questions.")

    with tab3:
        # Code Assistant interface
        st.title("Code Assistant")
        st.write("Get help with code analysis, generation, optimization, and bug finding.")
        
        language = st.selectbox(
            "Select Programming Language", 
            ["JavaScript", "Python", "Java", "C++", "Go", "Rust", "TypeScript", "PHP", "C#", "Swift"]
        )
        
        task_type = st.selectbox(
            "Select Task", 
            ["Code Analysis", "Code Generation", "Code Optimization", "Bug Finding"]
        )
        
        if task_type == "Code Generation":
            code = st.text_area(
                "Describe what code you want to generate:", 
                height=200, 
                placeholder="Create a function that..."
            )
        else:
            code = st.text_area(
                "Enter your code:", 
                height=200, 
                placeholder=f"Paste your {language} code here..."
            )
            
        if st.button("Process Code"):
            if code.strip():
                with st.spinner(f"Processing {task_type.lower()}..."):
                    # Simulate processing time
                    time.sleep(1.5)
                    
                    result = get_mock_code_analysis(code, language, task_type)
                    
                    st.subheader("Code Assistant Result")
                    st.markdown(result)
                    
                    # Save to chat history
                    save_chat_history(
                        st.session_state.user_id,
                        f"[Code Assistant - {task_type}] {code[:100]}...",
                        result,
                        "Code Assistant",
                        language
                    )
            else:
                st.warning("Please enter code or description first!")
                
    with tab4:
        # Web Automation interface
        st.title("Web Automation")
        st.write("Automate web tasks using AI.")
        
        website_url = st.text_input("Website URL", placeholder="https://example.com")
        
        automation_task = st.selectbox(
            "Select Task", 
            ["Data Extraction", "Form Filling", "Monitoring", "Screenshot", "Content Analysis"]
        )
        
        task_description = st.text_area(
            "Describe the task in detail:", 
            height=100, 
            placeholder="Extract all product prices and names..."
        )
            
        if st.button("Run Automation"):
            if website_url and task_description:
                with st.spinner("Running web automation..."):
                    # Simulate processing time
                    time.sleep(2)
                    
                    result = get_mock_web_automation(website_url, task_description)
                    
                    st.subheader("Automation Results")
                    st.markdown(result)
                    
                    # Save to chat history
                    save_chat_history(
                        st.session_state.user_id,
                        f"[Web Automation] {website_url} - {task_description[:100]}...",
                        result,
                        "Web Automation",
                        automation_task
                    )
            else:
                st.warning("Please enter a website URL and task description!")

# Add footer
st.markdown("---")
st.markdown("© 2024 AI Agent Platform - Seminar Demo Version") 
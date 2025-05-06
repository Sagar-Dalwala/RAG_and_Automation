import streamlit as st
import requests
from auth_db import init_db, create_user, verify_user, save_chat_history, get_user_chat_history
from rag_utils import process_input, answer_question
from session_manager import init_session, login_user, logout_user
import pandas as pd
import json

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
            # Web Automation interface with advanced features
            st.title("Advanced Web Scraping & Analysis")
            st.write("Extract, analyze, and visualize content from websites with our advanced web scraping tools.")
            
            # Initialize web automation in session state if not present
            if 'web_automation' not in st.session_state:
                from ai_web_automation import AdvancedWebScraper
                st.session_state.web_automation = AdvancedWebScraper()
            
            # Create tabs for different web automation features
            automation_tab1, automation_tab2, automation_tab3, automation_tab4 = st.tabs([
                "Basic Scraping", "Content Analysis", "Site Crawler", "Data Extraction"
            ])
            
            with automation_tab1:
                # URL input
                url = st.text_input("Enter URL to browse", 
                                   placeholder="https://example.com",
                                   key="basic_url")
                
                col1, col2, col3 = st.columns([1, 1, 1])
                
                with col1:
                    if st.button("Load Page"):
                        if url:
                            with st.spinner("Loading page..."):
                                result = st.session_state.web_automation.navigate_to_url(url)
                                if "error" in result:
                                    st.error(f"Error: {result['error']}")
                                else:
                                    st.success("Page loaded successfully!")
                                    st.subheader("Page Metrics")
                                    metrics = result.get("metrics", {})
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric("Load Time", f"{metrics.get('load_time_seconds', 0)} sec")
                                    with col2:
                                        st.metric("Page Size", f"{metrics.get('page_size_bytes', 0) // 1024} KB")
                                    
                                    # Extract and display content
                                    with st.spinner("Extracting content..."):
                                        content = st.session_state.web_automation.extract_page_content()
                                        if "error" not in content:
                                            st.subheader("Page Content")
                                            st.write(f"Title: {content['title']}")
                                            
                                            # Display meta tags
                                            with st.expander("Meta Information"):
                                                if 'meta_tags' in content and content['meta_tags']:
                                                    for name, value in content['meta_tags'].items():
                                                        st.text(f"{name}: {value}")
                                                else:
                                                    st.text("No meta tags found")
                                            
                                            # Display main content
                                            with st.expander("Main Content"):
                                                st.markdown(content['main_content'][:1000] + 
                                                         ("..." if len(content['main_content']) > 1000 else ""))
                                            
                                            # Display links
                                            with st.expander(f"Links ({content['links_count']})"):
                                                for i, link in enumerate(content['links'][:20]):
                                                    st.markdown(f"[{link['text'] or link['href']}]({link['href']})")
                                                if content['links_count'] > 20:
                                                    st.write(f"... and {content['links_count'] - 20} more links")
                                            
                                            # Display images
                                            with st.expander(f"Images ({content['images_count']})"):
                                                image_grid = st.columns(3)
                                                for i, img in enumerate(content['images'][:9]):
                                                    with image_grid[i % 3]:
                                                        if img['src'].startswith(('http://', 'https://')):
                                                            st.image(img['src'], use_container_width=True)
                                                            st.text(f"Source: {img['src'][:30]}...")
                                        else:
                                            st.error(f"Error extracting content: {content['error']}")
                        else:
                            st.warning("Please enter a URL first!")
                
                with col2:
                    if st.button("Take Screenshot"):
                        if st.session_state.web_automation.current_url:
                            with st.spinner("Taking screenshot..."):
                                screenshot = st.session_state.web_automation.take_screenshot(full_page=True)
                                if "error" not in screenshot:
                                    st.image(screenshot["image_data"])
                                else:
                                    st.error(f"Error taking screenshot: {screenshot['error']}")
                        else:
                            st.warning("Please browse a page first!")
                
                with col3:
                    if st.button("Close Browser"):
                        close_result = st.session_state.web_automation.close_browser()
                        st.success(close_result.get("message", "Browser closed successfully!"))
                
                # Search functionality with improved display
                st.subheader("Search in Page")
                search_query = st.text_input("Enter search term", key="basic_search")
                context_size = st.slider("Context Size (words)", min_value=10, max_value=100, value=30)
                
                if st.button("Search Content"):
                    if search_query:
                        with st.spinner("Searching..."):
                            results = st.session_state.web_automation.search_in_page(search_query, context_size)
                            if "error" not in results:
                                st.write(f"Found {results['occurrences']} matches:")
                                
                                for i, result in enumerate(results['results'][:10]):
                                    with st.container():
                                        st.markdown(f"**Match {i+1}:** {result.get('match', '')}")
                                        context = result.get('context', '')
                                        # Highlight the match in the context
                                        highlighted_context = context.replace(
                                            result.get('match', ''), 
                                            f"**{result.get('match', '')}**"
                                        )
                                        st.markdown(f"**Context:** {highlighted_context}")
                                        st.markdown("---")
                                
                                if results['occurrences'] > 10:
                                    st.info(f"Showing 10 of {results['occurrences']} matches. Refine your search for more specific results.")
                            else:
                                st.error(f"Error searching: {results['error']}")
                    else:
                        st.warning("Please enter a search term!")
            
            with automation_tab2:
                st.subheader("Content Analysis")
                
                if st.button("Analyze Current Page"):
                    if st.session_state.web_automation.current_url:
                        with st.spinner("Analyzing page content..."):
                            analysis_result = st.session_state.web_automation.analyze_content()
                            
                            if "error" not in analysis_result:
                                analysis = analysis_result.get("analysis", {})
                                
                                # Display basic metrics
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Word Count", analysis.get("word_count", 0))
                                with col2:
                                    link_analysis = analysis.get("link_analysis", {})
                                    st.metric("Total Links", link_analysis.get("total_links", 0))
                                with col3:
                                    st.metric("Internal/External", 
                                            f"{link_analysis.get('internal_links', 0)}/{link_analysis.get('external_links', 0)}")
                                
                                # Word frequency visualization
                                st.subheader("Word Frequency Analysis")
                                visualizations = analysis.get("visualizations", {})
                                if "word_frequency" in visualizations:
                                    st.image(visualizations["word_frequency"])
                                
                                # Display most common words
                                if "most_common_words" in analysis:
                                    st.subheader("Most Common Words")
                                    
                                    # Create a table of the most common words
                                    word_data = analysis["most_common_words"]
                                    df = pd.DataFrame(word_data, columns=["Word", "Count"])
                                    st.dataframe(df)
                                
                                # Link distribution visualization
                                st.subheader("Link Distribution")
                                if "link_distribution" in visualizations:
                                    st.image(visualizations["link_distribution"])
                            else:
                                st.error(f"Error analyzing content: {analysis_result['error']}")
                    else:
                        st.warning("Please load a page first!")
                
                # Extract structured data
                st.subheader("Extract Structured Data")
                if st.button("Extract Metadata & Structured Data"):
                    if st.session_state.web_automation.current_url:
                        with st.spinner("Extracting structured data..."):
                            content = st.session_state.web_automation.extract_page_content()
                            if "error" not in content and "structured_data" in content:
                                structured_data = content["structured_data"]
                                
                                # Display JSON-LD data
                                if structured_data.get("json_ld"):
                                    with st.expander("JSON-LD Data"):
                                        for i, data in enumerate(structured_data["json_ld"]):
                                            st.json(data)
                                
                                # Display OpenGraph data
                                if structured_data.get("opengraph"):
                                    with st.expander("OpenGraph Data"):
                                        for prop, content in structured_data["opengraph"].items():
                                            st.text(f"{prop}: {content}")
                                
                                # Display Twitter Card data
                                if structured_data.get("twitter_cards"):
                                    with st.expander("Twitter Card Data"):
                                        for name, content in structured_data["twitter_cards"].items():
                                            st.text(f"{name}: {content}")
                            else:
                                st.info("No structured data found or page not loaded")
                    else:
                        st.warning("Please load a page first!")
                
                # Extract contact information
                st.subheader("Contact Information Extraction")
                if st.button("Extract Contact Info"):
                    if st.session_state.web_automation.current_url:
                        with st.spinner("Extracting contact information..."):
                            contact_info = st.session_state.web_automation.extract_contact_info()
                            if "error" not in contact_info:
                                # Display emails
                                if contact_info.get("emails"):
                                    st.subheader("Emails")
                                    for email in contact_info["emails"]:
                                        st.text(email)
                                else:
                                    st.info("No emails found")
                                    
                                # Display phones
                                if contact_info.get("phones"):
                                    st.subheader("Phone Numbers")
                                    for phone in contact_info["phones"]:
                                        st.text(phone)
                                else:
                                    st.info("No phone numbers found")
                                    
                                # Display addresses
                                if contact_info.get("addresses"):
                                    st.subheader("Addresses")
                                    for address in contact_info["addresses"]:
                                        st.text(address)
                                else:
                                    st.info("No addresses found")
                                    
                                # Display social media
                                if contact_info.get("social_media"):
                                    st.subheader("Social Media")
                                    for social in contact_info["social_media"]:
                                        st.markdown(f"[{social['platform']}]({social['url']})")
                                else:
                                    st.info("No social media links found")
                            else:
                                st.error(f"Error extracting contact info: {contact_info['error']}")
                    else:
                        st.warning("Please load a page first!")
            
            with automation_tab3:
                st.subheader("Website Crawler")
                st.write("Crawl multiple pages of a website and analyze the structure")
                
                crawl_url = st.text_input("Enter Starting URL", placeholder="https://example.com", key="crawl_url")
                
                col1, col2 = st.columns(2)
                with col1:
                    max_pages = st.slider("Maximum Pages", min_value=2, max_value=50, value=10)
                    crawl_depth = st.slider("Crawl Depth", min_value=1, max_value=5, value=2)
                with col2:
                    stay_on_domain = st.checkbox("Stay on Same Domain", value=True)
                
                if st.button("Start Crawling"):
                    if crawl_url:
                        with st.spinner(f"Crawling website starting from {crawl_url}..."):
                            crawl_results = st.session_state.web_automation.crawl_site(
                                crawl_url, 
                                max_pages=max_pages,
                                stay_on_domain=stay_on_domain,
                                depth=crawl_depth
                            )
                            
                            if "error" not in crawl_results:
                                results = crawl_results.get("crawl_results", {})
                                
                                # Display crawl statistics
                                st.subheader("Crawl Statistics")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Pages Visited", results.get("pages_visited", 0))
                                with col2:
                                    st.metric("URLs Found", results.get("urls_found", 0))
                                with col3:
                                    st.metric("Errors", len(results.get("errors", [])))
                                
                                # Display site structure graph
                                if "site_graph" in results:
                                    st.subheader("Site Structure")
                                    st.image(results["site_graph"])
                                
                                # Display crawled pages
                                st.subheader("Crawled Pages")
                                for url, data in results.get("data", {}).items():
                                    with st.expander(f"{data.get('title', url)}"):
                                        st.markdown(f"**URL:** {url}")
                                        st.markdown(f"**Links:** {data.get('links_count', 0)}")
                                        st.markdown(f"**Images:** {data.get('images_count', 0)}")
                                
                                # Display errors
                                if results.get("errors"):
                                    with st.expander("Crawl Errors"):
                                        for error in results["errors"]:
                                            st.markdown(f"**URL:** {error.get('url')}")
                                            st.markdown(f"**Error:** {error.get('error')}")
                            else:
                                st.error(f"Error during crawl: {crawl_results['error']}")
                    else:
                        st.warning("Please enter a starting URL!")
            
            with automation_tab4:
                st.subheader("Custom Data Extraction")
                st.write("Extract specific content using CSS selectors")
                
                extract_url = st.text_input("Enter URL", placeholder="https://example.com", key="extract_url")
                
                # Common selector templates
                st.write("Common selector templates:")
                selector_cols = st.columns(4)
                with selector_cols[0]:
                    if st.button("Main Content"):
                        st.session_state.css_selector_value = "main, #content, .content, article"
                        st.rerun()
                with selector_cols[1]:
                    if st.button("Headlines"):
                        st.session_state.css_selector_value = "h1, h2, h3"
                        st.rerun()
                with selector_cols[2]:
                    if st.button("All Links"):
                        st.session_state.css_selector_value = "a[href]"
                        st.rerun()
                with selector_cols[3]:
                    if st.button("All Images"):
                        st.session_state.css_selector_value = "img[src]"
                        st.rerun()
                
                # Initialize the css_selector_value in session state if not present
                if 'css_selector_value' not in st.session_state:
                    st.session_state.css_selector_value = ""
                
                # Use the session state value in the text input
                css_selector = st.text_input(
                    "Enter CSS Selector", 
                    value=st.session_state.css_selector_value,
                    placeholder="article .content, #main-content, .product-card",
                    help="Examples: 'h1' (all h1 headers), '.product' (elements with 'product' class), '#main' (element with 'main' id)"
                )
                
                if st.button("Extract Data"):
                    if extract_url and css_selector:
                        # Navigate to URL if it's different from current
                        if st.session_state.web_automation.current_url != extract_url:
                            with st.spinner(f"Loading {extract_url}..."):
                                result = st.session_state.web_automation.navigate_to_url(extract_url)
                                if "error" in result:
                                    st.error(f"Error loading page: {result['error']}")
                                    st.stop()
                        
                        # Extract data using selector
                        with st.spinner(f"Extracting data using selector: {css_selector}"):
                            extraction_result = st.session_state.web_automation.extract_by_css_selector(css_selector)
                            if "error" not in extraction_result:
                                st.success(f"Found {extraction_result.get('count', 0)} matching elements!")
                                
                                # Display results
                                for i, item in enumerate(extraction_result.get('results', [])):
                                    with st.expander(f"Element {i+1} ({item.get('tag_name', 'unknown')})"):
                                        # Display extracted text
                                        st.markdown("**Text Content:**")
                                        st.text(item.get('text', 'No text content'))
                                        
                                        # Display attributes
                                        st.markdown("**Attributes:**")
                                        for attr, value in item.get('attributes', {}).items():
                                            st.text(f"{attr}: {value}")
                                        
                                        # Display HTML
                                        st.markdown("**Raw HTML:**")
                                        show_html = st.checkbox(f"Show HTML for Element {i+1}", key=f"show_html_{i}")
                                        if show_html:
                                            st.code(item.get('html', ''), language='html')
                            
                                # Download data as JSON
                                if extraction_result.get('count', 0) > 0:
                                    json_data = json.dumps(extraction_result.get('results', []), indent=2)
                                    st.download_button(
                                        "Download as JSON",
                                        data=json_data,
                                        file_name="extracted_data.json",
                                        mime="application/json"
                                    )
                            else:
                                st.error(f"Error extracting data: {extraction_result['error']}")
                    else:
                        st.warning("Please enter both URL and CSS selector!")

                # Advanced Image Extraction section
                st.subheader("Advanced Image Extraction")
                st.write("Find and extract all images from the page with filtering options")
                
                col1, col2 = st.columns(2)
                with col1:
                    min_width = st.number_input("Minimum Width (px)", min_value=0, value=100)
                    min_height = st.number_input("Minimum Height (px)", min_value=0, value=100)
                with col2:
                    image_format = st.multiselect("Image Format", ["jpg", "jpeg", "png", "gif", "webp", "svg"], default=["jpg", "png"])
                    exclude_icons = st.checkbox("Exclude Small Icons", value=True)
                
                if st.button("Extract All Images"):
                    if st.session_state.web_automation.current_url:
                        with st.spinner("Extracting images..."):
                            # Call the web automation method to extract images
                            image_result = st.session_state.web_automation.extract_images(
                                min_width=min_width,
                                min_height=min_height,
                                formats=image_format,
                                exclude_icons=exclude_icons
                            )
                            
                            if "error" not in image_result:
                                st.success(f"Found {len(image_result.get('images', []))} images!")
                                
                                # Display images in a grid
                                if image_result.get('images'):
                                    # Create columns for the grid
                                    num_cols = 3
                                    image_grid = [st.columns(num_cols) for _ in range((len(image_result['images']) + num_cols - 1) // num_cols)]
                                    
                                    for i, img in enumerate(image_result['images']):
                                        col_idx = i % num_cols
                                        row_idx = i // num_cols
                                        
                                        with image_grid[row_idx][col_idx]:
                                            st.image(img['src'], use_container_width=True)
                                            st.write(f"Size: {img.get('width', 'Unknown')}x{img.get('height', 'Unknown')}")
                                            
                                            # Add download button for each image
                                            st.markdown(f"[Download Image]({img['src']})")
                                            
                                            # Show image details on expander
                                            with st.expander("Image Details"):
                                                st.write(f"Alt Text: {img.get('alt', 'None')}")
                                                st.write(f"Source: {img['src']}")
                                                if img.get('title'):
                                                    st.write(f"Title: {img['title']}")
                                    
                                    # Add bulk download option
                                    st.subheader("Bulk Download")
                                    image_urls = [img['src'] for img in image_result['images']]
                                    st.download_button(
                                        "Download All Image URLs",
                                        "\n".join(image_urls),
                                        file_name="image_urls.txt",
                                        mime="text/plain"
                                    )
                            else:
                                st.error(f"Error extracting images: {image_result['error']}")
                    else:
                        st.warning("Please load a page first!")
else:
    st.info("Please login or sign up to start chatting with the AI Agent.")
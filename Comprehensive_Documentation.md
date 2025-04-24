# Advanced RAG and Web Automation System
## Comprehensive Documentation

## Table of Contents

1. [Introduction](#1-introduction)
   1. [Project Overview](#11-project-overview)
   1. [Problem Statement](#12-problem-statement)
   1. [Objectives](#13-objectives)

2. [Key Features](#2-key-features)
   1. [RAG System Features](#21-rag-system-features)
   1. [Web Automation Features](#22-web-automation-features)
   1. [Code Assistant Features](#23-code-assistant-features)
   1. [Session Management Features](#24-session-management-features)

3. [System Architecture](#3-system-architecture)
   1. [Core Components](#31-core-components)
   1. [Technology Stack](#32-technology-stack)
   1. [Component Interaction](#33-component-interaction)
   1. [Data Flow](#34-data-flow)

4. [Technical Implementation](#4-technical-implementation)
   1. [RAG System Implementation](#41-rag-system-implementation)
      1. [Document Processing Pipeline](#411-document-processing-pipeline)
      1. [Question Answering Process](#412-question-answering-process)
   1. [Web Automation System](#42-web-automation-system)
      1. [Core Components](#421-core-components)
      1. [Advanced Features](#422-advanced-features)
   1. [Code Assistant Implementation](#43-code-assistant-implementation)
   1. [Session Management](#44-session-management)
   1. [Database Architecture](#45-database-architecture)

5. [AI Technologies](#5-ai-technologies)
   1. [Large Language Models (LLMs)](#51-large-language-models-llms)
      1. [Hugging Face Models](#511-hugging-face-models)
      1. [Groq Models](#512-groq-models)
   1. [Retrieval-Augmented Generation (RAG)](#52-retrieval-augmented-generation-rag)
      1. [Vector Embeddings and Search](#521-vector-embeddings-and-search)
      1. [Generation with Context](#522-generation-with-context)
   1. [Web Automation and NLP](#53-web-automation-and-nlp)
   1. [Code Assistant AI](#54-code-assistant-ai)
   1. [Integration Architecture](#55-integration-architecture)

6. [Setup and Installation](#6-setup-and-installation)
   1. [Prerequisites](#61-prerequisites)
   1. [Installation Steps](#62-installation-steps)
   1. [Configuration](#63-configuration)
   1. [Running the Application](#64-running-the-application)

7. [User Guide](#7-user-guide)
   1. [Using the RAG System](#71-using-the-rag-system)
   1. [Using the Web Automation System](#72-using-the-web-automation-system)
   1. [Using the Code Assistant](#73-using-the-code-assistant)
   1. [Troubleshooting](#74-troubleshooting)

8. [Performance Considerations](#8-performance-considerations)
   1. [Scalability](#81-scalability)
   2. [Memory Usage](#82-memory-usage)
   3. [Processing Time](#83-processing-time)
   4. [Performance Optimizations](#84-performance-optimizations)

9. [Security Considerations](#9-security-considerations)
   1. [API Security](#91-api-security)
   2. [Data Security](#92-data-security)
   3. [Web Automation Security](#93-web-automation-security)

10. [Future Enhancements](#10-future-enhancements)
    1. [RAG System Enhancements](#101-rag-system-enhancements)
    2. [Web Automation Enhancements](#102-web-automation-enhancements)
    3. [Code Assistant Enhancements](#103-code-assistant-enhancements)

11. [References and Resources](#11-references-and-resources)
    1. [Academic Papers](#111-academic-papers)
    2. [Libraries and Frameworks](#112-libraries-and-frameworks)
    3. [Tutorials and Guides](#113-tutorials-and-guides)
    4. [Models Used](#114-models-used)

## 1. Introduction

### 1.1 Project Overview

This project combines several advanced AI technologies to create a multipurpose system capable of Retrieval-Augmented Generation (RAG), web automation, and code assistance. The system is designed to help users process documents, answer questions based on document content, automate web scraping tasks, and provide code-related assistance.

The integration of these technologies creates a powerful tool that addresses multiple aspects of information processing and automation, demonstrating the practical application of cutting-edge AI in solving real-world problems.

### 1.2 Problem Statement

Modern information workers face several challenges that this system addresses:

1. **Information Overload**: The exponential growth of digital documents makes it difficult to extract relevant information efficiently.

2. **Data Extraction Complexity**: Web content is vast and difficult to process manually, requiring automated tools to extract and analyze data.

3. **Code Development Inefficiency**: Software development often involves repetitive tasks and requires specialized knowledge that could be supplemented with AI assistance.

4. **Context Continuity**: Maintaining context across interactions with information systems is challenging but crucial for productivity.

### 1.3 Objectives

The primary objectives of this project are:

1. Develop a comprehensive RAG system that allows users to query document content using natural language.

2. Create an advanced web automation system for efficient web content extraction and analysis.

3. Implement a code assistant system that helps with code analysis, generation, optimization, and bug finding.

4. Design a session management system that maintains context across user interactions.

5. Integrate these components into a cohesive system with a user-friendly interface.

## 2. Key Features

### 2.1 RAG System Features

The RAG component offers the following key features:

- **Multiple Input Types**: Support for various document formats (PDF, DOCX, TXT, URL links).
- **Document Processing**: Text chunking, embedding generation, and vector storage.
- **Question Answering**: AI-powered answers based on document content.
- **Visualization**: Data visualization of document content and relationships.
- **Analytics**: Performance metrics and content analysis.
- **Configurable Parameters**: Adjustable chunk size and overlap for optimization.
- **Source Attribution**: Tracking which document sections contributed to answers.
- **Confidence Scoring**: Assessment of answer reliability.

### 2.2 Web Automation Features

The web automation system provides the following capabilities:

- **Web Navigation**: Automated browsing and page visiting.
- **Content Extraction**: Structured extraction of web page content.
- **Site Crawling**: Automated exploration of websites with configurable parameters.
- **Data Analysis**: Analysis of website structure, content, and relationships.
- **Visualization**: Generation of site maps and content relationship graphs.
- **Screenshot Capture**: Ability to capture full or partial page screenshots.
- **Structured Data Extraction**: Parsing of JSON-LD, OpenGraph, and other metadata.
- **Content Analysis**: Keyword frequency analysis and entity recognition.
- **Data Export**: Various export formats for extracted data.

### 2.3 Code Assistant Features

The code assistant provides AI-powered code-related services:

- **Code Analysis**: Structure analysis, best practices review.
- **Code Generation**: AI-powered code generation based on requirements.
- **Code Optimization**: Performance and readability improvements.
- **Bug Finding**: Identification of potential issues and suggested fixes.
- **Multiple Language Support**: Ability to work with various programming languages.
- **Task-Specific Prompting**: Specialized prompts for different code operations.
- **API Integration**: Accessible through a simple REST API.

### 2.4 Session Management Features

The session management system offers:

- **Session Persistence**: Maintaining user sessions across interactions.
- **State Management**: Tracking and managing application state.
- **User Preferences**: Storing and retrieving user preferences.
- **Authentication**: Secure user authentication with JWT tokens.
- **History Tracking**: Recording of user interactions for analytics and continuity.
- **Recovery Mechanisms**: Handling of crashes or disconnections.

## 3. System Architecture

### 3.1 Core Components

The system consists of four main components:

1. **RAG System**: Handles document processing and question answering.
   - Key Files: `advanced_rag_app.py`, `rag_utils.py`

2. **Web Automation System**: Manages web scraping and analysis.
   - Key Files: `web_automation.py`, `ai_web_automation.py`

3. **Code Assistant**: Provides code analysis and generation capabilities.
   - Key Files: `code_assistant.py`

4. **Session Management**: Handles user sessions and state persistence.
   - Key Files: `session_manager.py`, `auth_db.py`

### 3.2 Technology Stack

The system is built using the following technologies:

- **Programming Language**: Python 3.8+
- **Web Framework**: Streamlit (UI), FastAPI (backend services)
- **AI and ML**: 
  - LangChain: Framework for LLM applications
  - FAISS: Vector database for similarity search
  - HuggingFace Transformers: Access to state-of-the-art language models
  - Groq: High-performance LLM API
- **Web Automation**: 
  - Selenium: Browser automation
  - BeautifulSoup: HTML parsing
  - NetworkX: Graph analysis and visualization
- **Data Storage**: 
  - SQLite: Lightweight relational database
  - FAISS: In-memory vector store
- **Visualization**: 
  - Matplotlib
  - Plotly
  - Seaborn
- **Document Processing**:
  - PyPDF2: PDF handling
  - python-docx: DOCX processing

### 3.3 Component Interaction

The components interact in the following ways:

1. The RAG System processes documents and stores them in the vector database for later retrieval.

2. The Web Automation System can extract content from websites, which can then be processed by the RAG System.

3. The Code Assistant operates independently, providing code-related services through its API.

4. The Session Management System provides authentication and state persistence services to all other components.

### 3.4 Data Flow

The data flow through the system follows these general patterns:

1. **Document Processing Flow**:
   - Document Input → Text Extraction → Chunking → Embedding Generation → Vector Storage

2. **Question Answering Flow**:
   - Query → Query Embedding → Similarity Search → Context Retrieval → LLM Processing → Response

3. **Web Automation Flow**:
   - URL Input → Browser Navigation → Content Extraction → Data Processing → Analysis → Visualization

4. **Code Assistant Flow**:
   - Code Input → Task Selection → Prompt Construction → LLM Processing → Response Formatting

The system architecture is visualized in the following diagram:

```
┌─────────────────────┐    ┌──────────────────┐    ┌───────────────┐
│                     │    │                  │    │               │
│  Document Input     │───▶│  Text Processing │───▶│  Embedding    │
│  (PDF, DOCX, URLs)  │    │  & Chunking      │    │  Generation   │
│                     │    │                  │    │               │
└─────────────────────┘    └──────────────────┘    └───────┬───────┘
                                                           │
                                                           ▼
┌─────────────────────┐    ┌──────────────────┐    ┌───────────────┐
│                     │    │                  │    │               │
│  User Interface     │◀───│  LLM Processing  │◀───│  Vector Store │
│  (Streamlit)        │    │  (RAG Pipeline)  │    │  (FAISS)      │
│                     │    │                  │    │               │
└─────────────────────┘    └──────────────────┘    └───────────────┘

┌─────────────────────┐    ┌──────────────────┐    ┌───────────────┐
│                     │    │                  │    │               │
│  Web Automation     │───▶│  Content         │───▶│  Data         │
│  (Selenium)         │    │  Extraction      │    │  Analysis     │
│                     │    │                  │    │               │
└─────────────────────┘    └──────────────────┘    └───────────────┘

┌─────────────────────┐
│                     │
│  Code Assistant     │
│  (FastAPI)          │
│                     │
└─────────────────────┘
```

## 4. Technical Implementation

### 4.1 RAG System Implementation

The RAG system is implemented using a combination of document processing, vector storage, and LLM-based generation.

#### 4.1.1 Document Processing Pipeline

The RAG system uses a sophisticated document processing pipeline:

1. **Document Ingestion**:
   - The system accepts various document formats (PDF, DOCX, TXT) and URLs
   - The `process_input()` function in `rag_utils.py` handles different input types
   - For PDF files, PyPDF2 is used to extract text content
   - For DOCX files, the python-docx library extracts text
   - For URLs, WebBaseLoader fetches and processes web content

2. **Text Chunking**:
   - The `CharacterTextSplitter` from LangChain splits documents into manageable chunks
   - Default chunk size is 1000 characters with 100 character overlap for context preservation
   - These parameters are configurable via the UI for optimization

3. **Embedding Generation**:
   - The system uses HuggingFace's sentence transformers for generating embeddings
   - Default model is "sentence-transformers/all-mpnet-base-v2"
   - The `HuggingFaceEmbeddings` class handles embedding creation

4. **Vector Storage**:
   - FAISS (Facebook AI Similarity Search) provides efficient vector storage and retrieval
   - The system creates a vector store from text chunks using the generated embeddings
   - FAISS enables fast similarity search for finding relevant content

The implementation details can be found in the `rag_utils.py` file:

```python
def process_input(input_type, input_data):
    """Processes different input types and returns a vectorstore."""
    # Handle different input types (PDF, DOCX, TXT, URL)
    # ...
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    # Split text into chunks
    # ...
    
    model_name = "sentence-transformers/all-mpnet-base-v2"
    hf_embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )
    
    # Create FAISS vector store
    vector_store = FAISS.from_texts(
        texts=texts,
        embedding=hf_embeddings
    )
    
    return vector_store
```

#### 4.1.2 Question Answering Process

When a user asks a question, the following process occurs:

1. The question is passed to the `answer_question()` function in `rag_utils.py`
2. The system uses the FAISS vector store to find relevant document chunks
3. These chunks are passed to the LLM (Qwen/QwQ-32B via HuggingFace) as context
4. The LLM generates an answer based on the retrieved context and the question
5. The system returns the answer with associated metadata and source references

The implementation details:

```python
def answer_question(vectorstore, query):
    """Answers a question based on the provided vectorstore."""
    llm = HuggingFaceEndpoint(
        repo_id='Qwen/QwQ-32B', 
        huggingfacehub_api_token=huggingface_api_key,
        temperature=0.6
    )
    
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
    
    # Use the invoke method instead of __call__
    answer = qa.invoke({"query": query})
    return answer
```

### 4.2 Web Automation System

The web automation system provides tools for automated web browsing, data extraction, and site analysis.

#### 4.2.1 Core Components

1. **Browser Management**:
   - The `AdvancedWebScraper` class in `ai_web_automation.py` handles browser initialization
   - Chrome in headless mode is used for web automation
   - Browser configuration includes security settings and performance optimizations

2. **Navigation and Page Loading**:
   - The `navigate_to_url()` method handles URL navigation with error handling
   - Page loading metrics (load time, page size) are captured for analysis
   - A wait mechanism ensures the page is fully loaded before processing

3. **Content Extraction**:
   - BeautifulSoup processes the page HTML for structured content extraction
   - The system extracts text, links, images, and metadata
   - Specialized methods handle structured data like JSON-LD and OpenGraph tags

4. **Site Crawling**:
   - The `crawl_site()` method implements intelligent web crawling
   - Parameters include max pages, domain restrictions, and crawl depth
   - A NetworkX graph tracks site structure during crawling

5. **Data Analysis**:
   - Content analysis identifies key topics, entities, and keyword frequencies
   - Link analysis maps site structure and identifies important pages
   - Contact information extraction finds emails, phone numbers, and social links

Implementation details from `ai_web_automation.py`:

```python
def navigate_to_url(self, url, wait_time=10):
    """Navigate to a specific URL with advanced error handling and metrics"""
    try:
        if not self.driver:
            result = self.start_browser()
            if "error" in result:
                return result
        
        start_time = time.time()
        self.driver.get(url)
        load_time = time.time() - start_time
        
        # Wait for page to be fully loaded
        WebDriverWait(self.driver, wait_time).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        self.current_url = url
        self.visited_urls.add(url)
        
        # Record metrics
        # ...
        
        return {
            "success": True, 
            "metrics": metrics
        }
    except Exception as e:
        return {"error": f"Error navigating to {url}: {str(e)}"}
```

#### 4.2.2 Advanced Features

1. **Screenshot Capabilities**:
   - Full-page and element-specific screenshots
   - Image processing for optimization and annotation

2. **Visualization**:
   - Site graph visualization with NetworkX and Matplotlib
   - Content analytics visualization with charts and heatmaps

3. **Data Export**:
   - Structured data export in JSON, CSV, and Excel formats
   - Analysis reports with visualizations and metrics

### 4.3 Code Assistant Implementation

The code assistant component provides AI-powered code analysis and generation:

1. **API Structure**:
   - FastAPI provides the backend for code assistance functionality
   - A single endpoint (/code-assistant) handles different code-related tasks

2. **Task Types**:
   - Code Analysis: Structure evaluation and best practices assessment
   - Code Generation: Creates code based on descriptions
   - Code Optimization: Improves existing code for performance and readability
   - Bug Finding: Identifies potential issues and suggests fixes

3. **LLM Integration**:
   - Uses Groq's llama-3.3-70b-versatile model for code understanding
   - Task-specific prompts guide the LLM for different code operations
   - The system handles various programming languages through language-specific instructions

Implementation details from `code_assistant.py`:

```python
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
```

### 4.4 Session Management

The session management system handles user state and persistence:

1. **User Sessions**:
   - The `session_manager.py` file implements session handling
   - Sessions track user preferences and interaction history
   - JWT-based authentication for secure session management

2. **State Persistence**:
   - SQLite database stores session information
   - In-memory caching for frequently accessed session data
   - Session recovery mechanisms for handling crashes or disconnections

The session management system is designed to maintain continuity across user interactions, enabling a seamless experience with the various system components.

### 4.5 Database Architecture

#### Schema Design

The system uses SQLite for persistence with the following key tables:

1. **Users Table**:
   - User authentication information
   - Preferences and settings

2. **Sessions Table**:
   - Active user sessions
   - Session metadata and timestamps

3. **Documents Table**:
   - Document metadata
   - Processing status and parameters

4. **QueryHistory Table**:
   - User query history
   - Associated answers and performance metrics

#### Database Operations

- `auth_db.py` handles authentication-related database operations
- `check_db.py` provides utilities for database maintenance and verification
- The system uses SQLAlchemy for ORM-based database interactions

Typical database interactions are handled through dedicated functions that abstract the underlying SQL operations, as shown in the following example from `auth_db.py`:

```python
def create_user(username, password_hash, email=None):
    """Create a new user in the database."""
    try:
        conn = sqlite3.connect('chat_app.db')
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO users (username, password_hash, email, created_at) VALUES (?, ?, ?, ?)",
            (username, password_hash, email, datetime.datetime.now())
        )
        
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        
        return user_id
    except sqlite3.Error as e:
        return None
```

## 5. AI Technologies

### 5.1 Large Language Models (LLMs)

The system integrates with several LLMs to power its various components:

#### 5.1.1 Hugging Face Models

1. **Qwen/QwQ-32B**
   - Used for: Question answering in the RAG system
   - Key characteristics:
     - 32 billion parameter model
     - Strong performance on knowledge-intensive tasks
     - Effective context utilization
   - Implementation: Used through the `HuggingFaceEndpoint` class in `rag_utils.py`

2. **Sentence Transformers**
   - Used for: Text embedding generation
   - Model: "sentence-transformers/all-mpnet-base-v2"
   - Key characteristics:
     - Specialized for semantic similarity
     - Produces 768-dimensional embeddings
     - Effective for document retrieval
   - Implementation: Used through the `HuggingFaceEmbeddings` class in `rag_utils.py`

#### 5.1.2 Groq Models

1. **llama-3.3-70b-versatile**
   - Used for: Code assistance functionality
   - Key characteristics:
     - 70 billion parameter model
     - Specialized for code understanding and generation
     - Low latency through Groq's specialized hardware
   - Implementation: Used through the `ChatGroq` class in `code_assistant.py`

### 5.2 Retrieval-Augmented Generation (RAG)

The RAG system combines retrieval and generation capabilities:

#### 5.2.1 Vector Embeddings and Search

1. **Embedding Generation Process**
   - Documents are split into chunks using LangChain's `CharacterTextSplitter`
   - Each chunk is embedded using sentence transformers
   - The embeddings are stored in a FAISS vector database
   - Mathematical representation: Each text chunk T is transformed into a vector v = E(T) where E is the embedding function

2. **Similarity Search Mechanism**
   - When a query q is received, it is embedded using the same embedding model
   - FAISS conducts a similarity search using cosine similarity
   - The search retrieves the k most similar document chunks
   - Mathematical representation: For a query embedding q and document embeddings {d₁, d₂, ..., dₙ}, find top-k documents that maximize cos(q, dᵢ)

3. **FAISS Implementation**
   - Uses the `FAISS` class from LangChain
   - Implements IndexFlatL2 for exact nearest neighbor search
   - Enables efficient similarity searches at scale

#### 5.2.2 Generation with Context

1. **Context Preparation**
   - Retrieved document chunks are formatted as context
   - Context is combined with the user query in a prompt template
   - The full prompt is sent to the language model

2. **LLM Response Generation**
   - The LLM generates an answer based on both the query and context
   - The response is filtered and formatted before returning to the user
   - The system tracks confidence scores based on retrieval metrics

### 5.3 Web Automation and NLP

The web automation system incorporates several NLP techniques:

1. **Key Topic Extraction**
   - Uses keyword frequency analysis 
   - Implements TF-IDF (Term Frequency-Inverse Document Frequency) for content weighting
   - Mathematical representation: For a term t and document d, TF-IDF(t,d) = TF(t,d) × IDF(t)

2. **Entity Recognition**
   - Simple pattern matching for basic entities (emails, phone numbers)
   - Regular expressions for structured data extraction
   - Implementation available in the `extract_contact_info()` method

3. **Data Visualization**
   - Graph-based visualization with NetworkX
   - Content analytics visualization with Matplotlib and Seaborn
   - Interactive visualizations with Plotly

### 5.4 Code Assistant AI

The code assistant uses specialized prompts and models:

1. **Task-Specific Prompting**
   - Code Analysis Prompt: Focuses on structure, issues, best practices, and improvements
   - Code Generation Prompt: Creates code based on descriptions with emphasis on best practices
   - Code Optimization Prompt: Improves existing code for performance and readability
   - Bug Finding Prompt: Identifies issues and suggests fixes

2. **LLM Chain Implementation**
   - Uses `ChatPromptTemplate` to create consistent prompts
   - Variables include code content and programming language
   - Uses the Groq model through LangChain's integration

Example prompts from `code_assistant.py`:

```python
ANALYSIS_PROMPT = """
Analyze the following {language} code:
{code}

Provide a detailed analysis including:
1. Code structure and organization
2. Potential issues or concerns
3. Best practices adherence
4. Suggestions for improvement
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
```

### 5.5 Integration Architecture

The AI components are integrated through a modular architecture:

1. **API-Based Integration**
   - Code assistant functionality exposed via RESTful API
   - JSON-based request/response format
   - Stateless design for scalability

2. **Streamlit UI Integration**
   - RAG system integrated into Streamlit UI
   - Direct function calls for embedded components
   - UI designed for intuitive interaction with AI capabilities

3. **Session-Based State Management**
   - Streamlit's session state for maintaining context
   - Session storage for persistent history and preferences

## 6. Setup and Installation

### 6.1 Prerequisites

Before installing the system, ensure you have the following prerequisites:

- Python 3.8 or higher
- Chrome browser (for web automation features)
- API keys for:
  - HuggingFace
  - Groq (optional)
  - OpenAI (optional)
- Sufficient disk space for dependencies and databases
- Internet connection for API access

### 6.2 Installation Steps

1. Clone the repository to your local machine:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Run the setup script to install dependencies and configure the environment:
   ```bash
   python setup.py
   ```

3. Follow the prompts to enter your API keys and configure the system.

### 6.3 Configuration

The system configuration is managed through environment variables in a `.env` file:

```
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
SECRET_KEY=your_secret_key
API_HOST=127.0.0.1
API_PORT=8000
```

You can modify these variables to adjust the system's behavior, such as changing API endpoints or authentication parameters.

### 6.4 Running the Application

You can start individual components or the entire system:

- For the full system:
  ```bash
  python run_all.py
  ```

- For just the RAG application:
  ```bash
  python advanced_rag_app.py
  ```

- For just the code assistant:
  ```bash
  python code_assistant.py
  ```

- For checking the database:
  ```bash
  python check_db.py
  ```

Once started, the main application will be available at http://localhost:8501 (Streamlit interface) and the API at http://localhost:8000.

## 7. User Guide

### 7.1 Using the RAG System

The RAG (Retrieval-Augmented Generation) system allows you to ask questions based on document content.

#### Step 1: Document Input

1. Open the RAG application in your browser (typically at http://localhost:8501).
2. Navigate to the "Input" tab.
3. Select the input type:
   - **Link**: Enter one or more URLs
   - **PDF**: Upload a PDF file
   - **Text**: Enter text directly
   - **DOCX**: Upload a Word document
   - **TXT**: Upload a text file
4. Configure advanced options (optional):
   - **Chunk Size**: Adjust the size of text chunks (larger chunks provide more context, smaller chunks are more precise)
   - **Chunk Overlap**: Set overlap between chunks to maintain context
   - **Include Metadata**: Toggle inclusion of document metadata
5. Click "Process Document" to analyze and embed the content.

#### Step 2: Asking Questions

1. Navigate to the "Q&A" tab.
2. Enter your question in the text field.
3. Configure options:
   - **Include Sources**: Toggle to show source information
   - **Confidence Threshold**: Adjust the minimum confidence score for answers
4. Click "Submit Question" to get an answer based on your document.

#### Step 3: Exploring Results

1. Review the answer provided by the system.
2. Check source references that show which parts of the document contributed to the answer.
3. Confidence scores indicate how reliable the system considers the answer.

#### Step 4: Using Visualizations

1. Navigate to the "Visualization" tab to see:
   - Document structure visualization
   - Content relationship maps
   - Key concept visualization
2. Use the visualization tools to understand document content better.

#### Step 5: Analytics

1. Navigate to the "Analytics" tab to see:
   - Query performance metrics
   - Document statistics
   - Content analysis

### 7.2 Using the Web Automation System

The web automation system helps you extract and analyze content from websites.

#### Basic Web Scraping

1. Import the necessary components:
   ```python
   from web_automation import WebAutomation
   ```
2. Initialize the web automation tool:
   ```python
   web_tool = WebAutomation()
   ```
3. Navigate to a URL:
   ```python
   result = web_tool.navigate_to_url("https://example.com")
   ```
4. Extract content:
   ```python
   content = web_tool.extract_page_content()
   ```
5. Close the browser when done:
   ```python
   web_tool.close_browser()
   ```

#### Advanced Web Scraping

1. Import the advanced components:
   ```python
   from ai_web_automation import AdvancedWebScraper
   ```
2. Initialize the scraper:
   ```python
   scraper = AdvancedWebScraper()
   ```
3. Navigate to a URL:
   ```python
   result = scraper.navigate_to_url("https://example.com")
   ```
4. Extract structured data:
   ```python
   structured_data = scraper.extract_structured_data()
   ```
5. Analyze content:
   ```python
   analysis = scraper.analyze_content()
   ```
6. Crawl a site:
   ```python
   crawl_results = scraper.crawl_site("https://example.com", max_pages=10, stay_on_domain=True)
   ```
7. Generate visualizations:
   ```python
   graph = scraper._generate_site_graph_visualization()
   ```

### 7.3 Using the Code Assistant

The code assistant helps with code analysis, generation, optimization, and bug finding.

#### Using the API Directly

1. Start the code assistant service:
   ```bash
   python code_assistant.py
   ```
2. Send a request to the API endpoint:
   ```python
   import requests
   import json

   url = "http://localhost:8001/code-assistant"
   payload = {
       "code": "def example_function():\n    print('Hello, world!')",
       "task_type": "Code Analysis",
       "language": "Python"
   }
   response = requests.post(url, json=payload)
   result = response.text
   print(result)
   ```

#### Task Types

The code assistant supports different task types:

1. **Code Analysis**:
   - Analyzes code structure and best practices
   - Example payload:
     ```json
     {
       "code": "your code here",
       "task_type": "Code Analysis",
       "language": "Python"
     }
     ```

2. **Code Generation**:
   - Generates code based on a description
   - Example payload:
     ```json
     {
       "code": "Create a function that calculates the factorial of a number",
       "task_type": "Code Generation",
       "language": "Python"
     }
     ```

3. **Code Optimization**:
   - Optimizes existing code for performance and readability
   - Example payload:
     ```json
     {
       "code": "your code here",
       "task_type": "Code Optimization",
       "language": "Python"
     }
     ```

4. **Bug Finding**:
   - Identifies potential bugs and suggests fixes
   - Example payload:
     ```json
     {
       "code": "your code here",
       "task_type": "Bug Finding",
       "language": "Python"
     }
     ```

### 7.4 Troubleshooting

#### Common Issues

1. **API Key Errors**:
   - Ensure your API keys are correctly set in the .env file
   - Verify you have sufficient credits for the services you're using

2. **Browser Automation Issues**:
   - Ensure Chrome is installed on your system
   - Check if you need to update Chrome or chromedriver
   - Try running without headless mode for debugging by modifying the options:
     ```python
     self.options.remove_argument('--headless')
     ```

3. **Document Processing Errors**:
   - Verify the document format is supported
   - Check if the document is password-protected
   - Try breaking large documents into smaller parts

4. **Performance Issues**:
   - Adjust chunk size for optimal performance
   - Consider using a more powerful machine for large documents
   - Limit the scope of web crawling operations

#### Getting Help

If you encounter issues not covered in this guide:

1. Check the console output for error messages
2. Review the documentation for specific components
3. Check the project repository for updates or known issues

## 8. Performance Considerations

### 8.1 Scalability

The system has been designed with the following scalability considerations:

- **Single-User to Multi-User**: The system is primarily designed for single-user operation but can be expanded for multi-user scenarios with appropriate modifications to the session management system.

- **Vector Storage Scaling**: The FAISS vector storage scales with document size and count. For very large document collections, consider using:
  - Disk-based FAISS indexes
  - Sharded vector stores
  - Optimized index types (e.g., IVF indexes for approximate search)

- **Web Automation Scaling**: Web automation can be resource-intensive for large-scale crawling. Consider:
  - Distributed crawling across multiple instances
  - Rate limiting and politeness controls
  - Queue-based processing for large crawl jobs

### 8.2 Memory Usage

Memory usage considerations include:

- **Document Embedding Generation**: Generating embeddings for large documents requires significant memory, especially with larger models. The system uses:
  - Chunking to break documents into manageable pieces
  - Batch processing to control memory spikes
  - CPU-based inference as a default (with GPU optional for performance)

- **FAISS Optimization**: FAISS optimizes vector search operations for efficient retrieval, with:
  - In-memory indexes for speed
  - Configurable index types based on collection size
  - Quantization options for very large collections

- **Web Automation Memory**: Web automation's memory usage increases with the complexity of web pages. The system implements:
  - Browser resource management
  - Page cleanup after extraction
  - Configurable limits on crawl depth and breadth

### 8.3 Processing Time

Processing time considerations include:

- **Document Processing Time**: Processing time scales with document size and complexity:
  - PDF processing is typically the slowest
  - Text processing is the fastest
  - Web content varies based on page complexity

- **Question Answering Latency**: Question answering latency depends on:
  - The selected AI model (Qwen/QwQ-32B is optimized for throughput)
  - Vector search performance
  - Context length (more context = longer processing)

- **Web Automation Speed**: Web automation speeds vary based on:
  - Network conditions
  - Page complexity
  - JavaScript processing requirements
  - Waiting strategies (dynamic vs. fixed timeouts)

### 8.4 Performance Optimizations

The system implements several performance optimizations:

1. **Embedding Generation**:
   - Batch processing for efficient embedding generation
   - Model selection based on performance/quality trade-offs
   - Optional GPU acceleration for supported environments

2. **Vector Search**:
   - FAISS index optimization for faster similarity search
   - In-memory caching of frequent queries
   - Configurable k-parameter for retrieval speed/quality balance

3. **Web Automation**:
   - Browser resource management for reduced memory usage
   - Parallel processing for multi-page crawling (with rate limiting)
   - Selective content extraction based on page structure

4. **LLM Integration**:
   - Context optimization to reduce token usage
   - Result caching for common queries
   - Model selection based on task requirements

## 9. Security Considerations

### 9.1 API Security

The system implements several API security measures:

1. **Authentication**:
   - JWT token-based authentication
   - Token expiration and refresh mechanisms
   - Rate limiting for authentication endpoints

2. **API Keys**:
   - Environment variable storage for API keys
   - No hardcoded credentials in source code
   - Scoped permissions for different API endpoints

3. **Request Validation**:
   - Input validation for all API endpoints
   - Type checking with Pydantic models
   - Error handling with appropriate status codes

### 9.2 Data Security

Data security measures include:

1. **User Data**:
   - Password hashing with bcrypt
   - Sensitive data encryption
   - Data minimization principles

2. **Document Storage**:
   - Temporary storage of processed documents
   - Optional encryption for persistent storage
   - Access controls for document retrieval

3. **Query History**:
   - User-specific query history
   - Optional anonymization of queries
   - Configurable history retention

### 9.3 Web Automation Security

Web automation security considerations include:

1. **Ethical Crawling**:
   - Respects robots.txt directives
   - Implements rate limiting for polite crawling
   - User-agent identification

2. **Browser Security**:
   - Sandboxed browser environment
   - No persistent cookies or storage
   - Disabled plugins and extensions

3. **Content Security**:
   - Safe handling of extracted content
   - Sanitization of HTML content
   - Protection against common web vulnerabilities

## 10. Future Enhancements

### 10.1 RAG System Enhancements

Planned enhancements for the RAG system include:

1. **Advanced Retrieval Techniques**:
   - Hybrid search combining sparse and dense retrievers
   - Multi-stage retrieval pipeline
   - Query rewriting for improved accuracy

2. **Model Integration**:
   - Support for additional model providers
   - Fine-tuning capabilities for domain-specific knowledge
   - Model evaluation and selection automation

3. **User Experience**:
   - Conversational memory for follow-up questions
   - Interactive document exploration
   - Multi-document comparison and synthesis

### 10.2 Web Automation Enhancements

Planned enhancements for the web automation system include:

1. **Advanced Browser Automation**:
   - Support for JavaScript-heavy sites and dynamic content
   - Headless browser rendering improvements
   - Form interaction capabilities

2. **Content Analysis**:
   - Enhanced entity recognition
   - Sentiment analysis for extracted content
   - Topic modeling and document classification

3. **Integration Features**:
   - Direct export to the RAG system
   - Scheduled crawling and monitoring
   - Diff detection for content changes

### 10.3 Code Assistant Enhancements

Planned enhancements for the code assistant include:

1. **Language Support**:
   - Additional programming languages
   - Language-specific optimization suggestions
   - Framework-specific patterns and practices

2. **Interactive Mode**:
   - Interactive code improvement sessions
   - Step-by-step code generation guidance
   - Real-time feedback during development

3. **Integration Features**:
   - IDE plugins and extensions
   - Version control integration
   - CI/CD pipeline suggestions

## 11. References and Resources

### 11.1 Academic Papers

1. Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., ... & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. Advances in Neural Information Processing Systems, 33, 9459-9474.

2. Karpukhin, V., Oguz, B., Min, S., Lewis, P., Wu, L., Edunov, S., ... & Yih, W. T. (2020). Dense passage retrieval for open-domain question answering. arXiv preprint arXiv:2004.04906.

3. Johnson, J., Douze, M., & Jégou, H. (2019). Billion-scale similarity search with GPUs. IEEE Transactions on Big Data, 7(3), 535-547.

4. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. arXiv preprint arXiv:1908.10084.

5. Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., ... & Amodei, D. (2020). Language models are few-shot learners. arXiv preprint arXiv:2005.14165.

### 11.2 Libraries and Frameworks

#### RAG and LLM Integration

1. LangChain
   - Website: [https://www.langchain.com/](https://www.langchain.com/)
   - Documentation: [https://python.langchain.com/docs/get_started/introduction](https://python.langchain.com/docs/get_started/introduction)
   - GitHub: [https://github.com/langchain-ai/langchain](https://github.com/langchain-ai/langchain)

2. FAISS (Facebook AI Similarity Search)
   - Documentation: [https://faiss.ai/](https://faiss.ai/)
   - GitHub: [https://github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss)

3. Hugging Face Transformers
   - Website: [https://huggingface.co/](https://huggingface.co/)
   - Documentation: [https://huggingface.co/docs/transformers/index](https://huggingface.co/docs/transformers/index)
   - GitHub: [https://github.com/huggingface/transformers](https://github.com/huggingface/transformers)

4. Sentence Transformers
   - Documentation: [https://www.sbert.net/](https://www.sbert.net/)
   - GitHub: [https://github.com/UKPLab/sentence-transformers](https://github.com/UKPLab/sentence-transformers)

5. Groq API
   - Website: [https://groq.com/](https://groq.com/)
   - Documentation: [https://console.groq.com/docs/quickstart](https://console.groq.com/docs/quickstart)

#### Web Automation and Scraping

1. Selenium
   - Website: [https://www.selenium.dev/](https://www.selenium.dev/)
   - Documentation: [https://www.selenium.dev/documentation/](https://www.selenium.dev/documentation/)
   - GitHub: [https://github.com/SeleniumHQ/selenium](https://github.com/SeleniumHQ/selenium)

2. BeautifulSoup
   - Documentation: [https://www.crummy.com/software/BeautifulSoup/bs4/doc/](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
   - GitHub: [https://github.com/wention/BeautifulSoup4](https://github.com/wention/BeautifulSoup4)

3. NetworkX
   - Website: [https://networkx.org/](https://networkx.org/)
   - Documentation: [https://networkx.org/documentation/stable/](https://networkx.org/documentation/stable/)
   - GitHub: [https://github.com/networkx/networkx](https://github.com/networkx/networkx)

#### Web Development

1. FastAPI
   - Website: [https://fastapi.tiangolo.com/](https://fastapi.tiangolo.com/)
   - Documentation: [https://fastapi.tiangolo.com/tutorial/](https://fastapi.tiangolo.com/tutorial/)
   - GitHub: [https://github.com/tiangolo/fastapi](https://github.com/tiangolo/fastapi)

2. Streamlit
   - Website: [https://streamlit.io/](https://streamlit.io/)
   - Documentation: [https://docs.streamlit.io/](https://docs.streamlit.io/)
   - GitHub: [https://github.com/streamlit/streamlit](https://github.com/streamlit/streamlit)

#### Document Processing

1. PyPDF2
   - Documentation: [https://pypdf2.readthedocs.io/en/latest/](https://pypdf2.readthedocs.io/en/latest/)
   - GitHub: [https://github.com/py-pdf/pypdf2](https://github.com/py-pdf/pypdf2)

2. python-docx
   - Documentation: [https://python-docx.readthedocs.io/en/latest/](https://python-docx.readthedocs.io/en/latest/)
   - GitHub: [https://github.com/python-openxml/python-docx](https://github.com/python-openxml/python-docx)

### 11.3 Tutorials and Guides

1. "Building RAG Applications with LangChain"
   - [https://python.langchain.com/docs/use_cases/question_answering/](https://python.langchain.com/docs/use_cases/question_answering/)

2. "How to Build a Production-Ready RAG Application"
   - [https://www.pinecone.io/learn/series/langchain/langchain-rag/](https://www.pinecone.io/learn/series/langchain/langchain-rag/)

3. "Web Scraping with Selenium and Python"
   - [https://www.scrapingbee.com/blog/selenium-python/](https://www.scrapingbee.com/blog/selenium-python/)

4. "Building Machine Learning Powered Applications"
   - [https://www.oreilly.com/library/view/building-machine-learning/9781492045106/](https://www.oreilly.com/library/view/building-machine-learning/9781492045106/)

5. "Creating Web Applications with FastAPI"
   - [https://fastapi.tiangolo.com/tutorial/first-steps/](https://fastapi.tiangolo.com/tutorial/first-steps/)

### 11.4 Models Used

1. Qwen/QwQ-32B
   - Model Card: [https://huggingface.co/Qwen/QwQ-32B](https://huggingface.co/Qwen/QwQ-32B)

2. sentence-transformers/all-mpnet-base-v2
   - Model Card: [https://huggingface.co/sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)

3. llama-3.3-70b-versatile
   - Model Information: [https://console.groq.com/docs/models](https://console.groq.com/docs/models) 
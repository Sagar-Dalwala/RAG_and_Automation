# Advanced RAG and Web Automation System

## Project Overview
This project combines several advanced AI technologies to create a multipurpose system capable of Retrieval-Augmented Generation (RAG), web automation, and code assistance. The system is designed to help users process documents, answer questions based on document content, automate web scraping tasks, and provide code-related assistance.

## Core Components

### 1. RAG System (Retrieval-Augmented Generation)
The RAG component allows users to upload documents in various formats (PDF, DOCX, TXT), extract content from web links, or input text directly. The system then processes this information to enable AI-powered question answering based on the document content.

#### Key Files
- `advanced_rag_app.py`: Main Streamlit application for the RAG interface
- `rag_utils.py`: Core utilities for document processing and question answering

#### Features
- **Multiple Input Types**: Support for various document formats (PDF, DOCX, TXT, URL links)
- **Document Processing**: Text chunking, embedding generation, and vector storage
- **Question Answering**: AI-powered answers based on document content
- **Visualization**: Data visualization of document content and relationships
- **Analytics**: Performance metrics and content analysis

### 2. Web Automation System
The web automation system provides tools for automated web browsing, data extraction, and site analysis.

#### Key Files
- `web_automation.py`: Basic web automation functionality
- `ai_web_automation.py`: Advanced web scraping and analysis tools

#### Features
- **Web Navigation**: Automated browsing and page visiting
- **Content Extraction**: Structured extraction of web page content
- **Site Crawling**: Automated exploration of websites with configurable parameters
- **Data Analysis**: Analysis of website structure, content, and relationships
- **Visualization**: Generation of site maps and content relationship graphs
- **Screenshot Capture**: Ability to capture full or partial page screenshots

### 3. Code Assistant
The code assistant provides AI-powered code analysis, generation, optimization, and bug finding.

#### Key Files
- `code_assistant.py`: API for code assistance functionality

#### Features
- **Code Analysis**: Structure analysis, best practices review
- **Code Generation**: AI-powered code generation based on requirements
- **Code Optimization**: Performance and readability improvements
- **Bug Finding**: Identification of potential issues and suggested fixes

### 4. Session Management
The system includes session management capabilities for maintaining state across interactions.

#### Key Files
- `session_manager.py`: User session management and persistence

#### Features
- **Session Persistence**: Maintaining user sessions across interactions
- **State Management**: Tracking and managing application state
- **User Preferences**: Storing and retrieving user preferences

## Technical Implementation

### Languages and Frameworks
- **Python**: Primary programming language
- **Streamlit**: Web interface for the RAG application
- **FastAPI**: API endpoints for various services
- **Selenium**: Web automation and scraping
- **LangChain**: Framework for LLM applications

### AI and Machine Learning
- **Vector Embeddings**: Using HuggingFace embeddings for document vectorization
- **FAISS**: Vector database for efficient similarity search
- **Large Language Models**: Integration with Hugging Face, Groq, and OpenAI models

### Data Storage
- **SQLite**: Lightweight database for persistence
- **In-memory Vector Store**: FAISS for document embeddings

## Setup and Usage

### Prerequisites
- Python 3.8 or higher
- Chrome browser (for web automation features)
- API keys for HuggingFace, Groq, or OpenAI (as needed)

### Installation
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables in .env file:
   - `HUGGINGFACEHUB_API_TOKEN`: HuggingFace API token
   - `GROQ_API_KEY`: Groq API key (optional)
   - `OPENAI_API_KEY`: OpenAI API key (optional)
   - `SECRET_KEY`: Secret key for the application

### Running the Application
- **RAG System**: `python advanced_rag_app.py`
- **Web Automation Demo**: `python ai_web_automation.py`
- **Code Assistant**: `python code_assistant.py`
- **Full System**: `python run_all.py`

## System Architecture

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

## Performance Considerations

### Scalability
- The system is designed for single-user operation but can be expanded for multi-user scenarios
- Vector storage scales with document size and count
- Web automation can be resource-intensive for large-scale crawling

### Memory Usage
- Document embedding generation requires significant memory for large documents
- FAISS optimizes vector search operations for efficient retrieval
- Web automation's memory usage increases with the complexity of web pages

### Processing Time
- Document processing time scales with document size and complexity
- Question answering latency depends on the selected AI model
- Web automation speeds vary based on network conditions and page complexity

## Security Considerations

- API keys are stored in environment variables for security
- Web automation respects robots.txt and implements rate limiting
- Database access is controlled via appropriate authentication

## Future Enhancements

### Potential Improvements
1. **Multi-user Support**: Enhance session management for multiple concurrent users
2. **Advanced RAG Techniques**: Implement hybrid search and query rewriting
3. **Enhanced Web Automation**: Add support for JavaScript-heavy sites and dynamic content
4. **Extended Code Assistant**: Support more programming languages and frameworks
5. **Integrated Analytics**: Comprehensive analytics across all system components

## Conclusion

This system demonstrates the integration of multiple AI technologies to create a versatile tool for document processing, web automation, and code assistance. The modular architecture allows for future expansion and enhancement to meet evolving requirements. 
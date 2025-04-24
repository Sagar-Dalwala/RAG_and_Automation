# Advanced RAG Application with Hierarchical Chat History

This application demonstrates an advanced implementation of Retrieval-Augmented Generation (RAG) with hierarchical chat history, allowing users to branch conversations and maintain context across related discussions.

## Features

- Authentication with JWT tokens and cookie-based sessions
- Hierarchical chat history with parent-child relationships
- Support for multiple AI models from Groq and OpenAI
- Document processing and RAG integration
- Modern React frontend with Chakra UI
- FastAPI backend with SQLite database

## Project Structure

The project is divided into two main parts:

### Backend (FastAPI)

- `api/` - FastAPI application modules
  - `auth.py` - Authentication with JWT tokens
  - `chat_history.py` - Hierarchical chat history
  - `rag.py` - Document processing and RAG
  - `ai_agent.py` - AI model integration
  - `main.py` - Main application entry point

### Frontend (React)

- `frontend/` - React application
  - `src/components/` - Reusable UI components
  - `src/context/` - React context for state management
  - `src/pages/` - Page components
  - `src/utils/` - Utility functions

## Setup

### Environment Setup

1. Create a `.env` file in the root directory with the following variables:

```
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
SECRET_KEY=your_secret_key
API_HOST=127.0.0.1
API_PORT=8000
REACT_APP_API_URL=http://127.0.0.1:8000
```

### Backend Setup

1. Install Python dependencies:

```bash
pip install -r requirements.txt
```

2. Run the FastAPI server:

```bash
python run_api.py
```

The API will be available at [http://127.0.0.1:8000](http://127.0.0.1:8000) with documentation at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### Frontend Setup

1. Navigate to the frontend directory:

```bash
cd frontend
```

2. Install Node.js dependencies:

```bash
npm install
```

3. Start the development server:

```bash
npm start
```

The frontend will be available at [http://localhost:3000](http://localhost:3000).

## Usage

1. Register a new account or log in
2. Create a new chat session
3. Upload documents in the Documents section
4. Start chatting with the AI using the selected model
5. Branch conversations to explore different paths
6. Use RAG to query your documents

## Technologies Used

- **Backend**: FastAPI, SQLite, LangChain, FAISS
- **Frontend**: React, Chakra UI, React Router, React Query
- **Authentication**: JWT tokens, HTTP-only cookies
- **AI**: Groq and OpenAI models
- **Vector Database**: FAISS for document embeddings

## License

MIT


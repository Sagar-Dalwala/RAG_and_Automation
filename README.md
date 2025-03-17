# Beyond Chatbots - AI Agents with Real-Time Web Search & LLMs

## Overview
This project provides an AI Agent that interacts with users through a chatbot interface. It leverages Large Language Models (LLMs) such as LLaMA and GPT-4o, along with real-time web search capabilities. The application is built using FastAPI for the backend and Streamlit for the frontend, enabling dynamic AI interactions with a user-friendly interface.

## Features
- **AI Chatbot**: Interact with an AI chatbot powered by LangGraph.
- **Multiple AI Models**: Supports different LLMs like LLaMA, Mixtral, DeepSeek, and GPT-4o.
- **Real-Time Web Search**: Option to enable web search for more informed responses.
- **User-Friendly Interface**: Built with Streamlit for easy interaction.
- **REST API Support**: A FastAPI-based backend with a `/chat` endpoint for AI interactions.

## Tech Stack
- **Backend**: FastAPI
- **Frontend**: Streamlit
- **AI Models**: OpenAI, Groq (LLaMA, Mixtral, DeepSeek)
- **API Documentation**: Swagger UI (provided by FastAPI)

## Installation & Setup
### Prerequisites
Ensure you have Python installed (version 3.8+ recommended).

### Clone the Repository
```sh
 git clone <repository-url>
 cd <repository-folder>
```

### Install Dependencies
```sh
 pip install -r requirements.txt
```

### Run the Backend
```sh
 uvicorn main:app --reload
```
- The FastAPI server will start at `http://127.0.0.1:8000`
- Visit `http://127.0.0.1:8000/docs` for API documentation.

### Run the Frontend
```sh
 streamlit run app.py
```

## Usage
### 1. Using the API
- **GET `/`**: Returns a welcome message.
- **POST `/chat`**: Accepts a JSON payload to interact with the AI agent.

Example Request:
```json
{
    "model_name": "gpt-4o-mini",
    "model_provider": "OpenAI",
    "system_prompt": "Act as an AI assistant.",
    "messages": ["Hello, how can you help me?"],
    "allow_search": true
}
```

### 2. Using the UI
- Open the Streamlit app.
- Enter a system prompt to define the agent’s behavior.
- Select an AI model provider and model.
- Enter a query and interact with the AI agent.

## Future Enhancements
- Implement caching for responses.
- Expand support for more AI models.
- Optimize API response time.

## License
This project is open-source under the MIT License.

## Contributors
Feel free to contribute by submitting issues or pull requests!


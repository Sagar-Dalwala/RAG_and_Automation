import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get configuration from environment variables or use defaults
host = os.environ.get("API_HOST", "127.0.0.1")
port = int(os.environ.get("API_PORT", 8000))

if __name__ == "__main__":
    print(f"Starting API server at http://{host}:{port}")
    print(f"API documentation available at http://{host}:{port}/docs")
    
    # Start the server
    uvicorn.run("api.main:app", host=host, port=port, reload=True) 
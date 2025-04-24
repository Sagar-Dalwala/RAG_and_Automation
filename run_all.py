import subprocess
import os
import sys
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def is_windows():
    return sys.platform.startswith('win')

def run_backend():
    """Run the FastAPI backend server"""
    print("Starting backend server...")
    if is_windows():
        return subprocess.Popen(["python", "run_api.py"], creationflags=subprocess.CREATE_NEW_CONSOLE)
    else:
        return subprocess.Popen(["python", "run_api.py"])

def run_frontend():
    """Run the React frontend development server"""
    print("Starting frontend server...")
    os.chdir("frontend")
    if is_windows():
        return subprocess.Popen(["npm", "start"], creationflags=subprocess.CREATE_NEW_CONSOLE)
    else:
        return subprocess.Popen(["npm", "start"])

def main():
    """Run both backend and frontend servers"""
    try:
        # Start the backend
        backend_process = run_backend()
        print("Backend server started!")
        
        # Wait a bit for the backend to start
        time.sleep(2)
        
        # Start the frontend
        frontend_process = run_frontend()
        print("Frontend server started!")
        
        # Keep the script running to maintain the processes
        print("\nPress Ctrl+C to stop both servers...\n")
        
        # Wait for user to press Ctrl+C
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping servers...")
        
        # Terminate processes
        backend_process.terminate()
        frontend_process.terminate()
        
        print("Servers stopped!")

if __name__ == "__main__":
    main() 
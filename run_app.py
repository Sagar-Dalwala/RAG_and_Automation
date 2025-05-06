import subprocess
import sys
import time
import os

def is_windows():
    return sys.platform.startswith('win')

def run_backend():
    """Run the FastAPI backend server"""
    print("Starting backend server...")
    if is_windows():
        return subprocess.Popen(["python", "backend.py"], creationflags=subprocess.CREATE_NEW_CONSOLE)
    else:
        return subprocess.Popen(["python", "backend.py"])

def run_frontend():
    """Run the Streamlit frontend"""
    print("Starting frontend server...")
    if is_windows():
        return subprocess.Popen(["streamlit", "run", "frontend.py"], creationflags=subprocess.CREATE_NEW_CONSOLE)
    else:
        return subprocess.Popen(["streamlit", "run", "frontend.py"])

def main():
    """Run both backend and frontend servers"""
    try:
        # Check if the required files exist
        if not os.path.exists("backend.py"):
            print("Error: backend.py not found!")
            return
            
        if not os.path.exists("frontend.py"):
            print("Error: frontend.py not found!")
            return
        
        # Start the backend
        backend_process = run_backend()
        print("Backend server started!")
        
        # Wait a bit for the backend to start
        time.sleep(2)
        
        # Start the frontend
        frontend_process = run_frontend()
        print("Frontend server started!")
        
        # Keep the script running to maintain the processes
        print("\nBoth servers are now running.")
        print("Access the frontend at: http://localhost:8501")
        print("The backend API is available at: http://localhost:8000")
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
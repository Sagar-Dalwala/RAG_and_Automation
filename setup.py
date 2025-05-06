import subprocess
import os
import platform
import sys
import secrets
import time

def print_step(step, message):
    """Format step messages consistently"""
    print(f"\n\n{'=' * 80}")
    print(f"Step {step}: {message}")
    print(f"{'=' * 80}\n")

def run_command(command, cwd=None):
    """Run a shell command and display output"""
    try:
        subprocess.run(command, check=True, cwd=cwd, shell=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        return False

def is_command_available(command):
    """Check if a command is available on the system"""
    try:
        if platform.system() == "Windows":
            subprocess.run(["where", command], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        else:
            subprocess.run(["which", command], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError:
        return False

def setup_environment():
    """Create or update the .env file"""
    print_step(1, "Setting up environment variables")
    
    env_file = ".env"
    env_vars = {}
    
    # Read existing .env file if it exists
    if os.path.exists(env_file):
        print("Reading existing .env file...")
        with open(env_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    env_vars[key] = value
    
    # Required API keys
    required_keys = {
        "GROQ_API_KEY": "Enter your Groq API key (or leave empty):",
        "OPENAI_API_KEY": "Enter your OpenAI API key (or leave empty):",
        "HUGGINGFACEHUB_API_TOKEN": "Enter your HuggingFace API token (or leave empty):",
        "SECRET_KEY": None  # Will be auto-generated if not present
    }
    
    # Configuration values
    config_values = {
        "API_HOST": "127.0.0.1",
        "PORT": "8000",
        "REACT_APP_API_URL": "http://127.0.0.1:8000"
    }
    
    # Prompt for missing required keys
    for key, prompt in required_keys.items():
        if key not in env_vars or not env_vars[key]:
            if key == "SECRET_KEY":
                # Generate a random secret key
                env_vars[key] = secrets.token_hex(32)
            elif prompt:
                value = input(f"{prompt} ")
                env_vars[key] = value
    
    # Add configuration values
    for key, value in config_values.items():
        if key not in env_vars:
            env_vars[key] = value
    
    # Write the .env file
    with open(env_file, "w") as f:
        for key, value in env_vars.items():
            f.write(f"{key}={value}\n")
    
    print("Environment variables have been set up successfully.")

def install_python_dependencies():
    """Install Python dependencies using pip"""
    print_step(2, "Installing Python dependencies")
    
    if not is_command_available("pip"):
        print("Error: pip is not installed. Please install pip and try again.")
        return False
    
    return run_command("pip install -r requirements.txt")

def install_node_dependencies():
    """Install Node.js dependencies"""
    print_step(3, "Installing Node.js dependencies")
    
    if not is_command_available("npm"):
        print("Error: npm is not installed. Please install Node.js and npm, then try again.")
        return False
    
    # Navigate to frontend directory and install dependencies
    return run_command("npm install", cwd="frontend")

def initialize_database():
    """Initialize the SQLite database"""
    print_step(4, "Initializing database")
    
    # Run a Python script to initialize the database
    initialize_script = """
import sqlite3
import os

def init_db():
    # Create database directory if it doesn't exist
    if not os.path.exists('chat_app.db'):
        # Connect to database (will create it if it doesn't exist)
        conn = sqlite3.connect('chat_app.db')
        conn.close()
        print("Database initialized successfully.")
    else:
        print("Database already exists.")

if __name__ == "__main__":
    init_db()
"""
    
    with open("init_db.py", "w") as f:
        f.write(initialize_script)
    
    success = run_command("python init_db.py")
    
    # Remove the temporary script
    if os.path.exists("init_db.py"):
        os.remove("init_db.py")
    
    return success

def main():
    """Main setup function"""
    print("Advanced RAG Application Setup")
    print("------------------------------")
    
    # Setup steps
    steps = [
        setup_environment,
        install_python_dependencies,
        install_node_dependencies,
        initialize_database
    ]
    
    success = True
    for step_func in steps:
        if not step_func():
            success = False
            break
        time.sleep(1)  # Brief pause between steps
    
    if success:
        print_step("✅", "Setup complete!")
        print("You can now run the application using:")
        print("  Python only:  python run_api.py")
        print("  Full stack:   python run_all.py")
    else:
        print_step("❌", "Setup failed")
        print("Please fix the errors above and try again.")

if __name__ == "__main__":
    main() 

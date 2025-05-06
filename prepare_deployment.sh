#!/bin/bash
# Script to prepare your app for deployment

echo "Preparing AI Agent Platform for deployment..."

# Create a deployment directory
mkdir -p deployment
cd deployment

# Copy the necessary files
cp ../frontend.py ../backend.py ../run_app.py . 
cp ../requirements-deployment.txt requirements.txt
cp ../Procfile .

# If needed, adjust frontend to connect to proper backend in production
echo "Do you want to modify the frontend to connect to a remote backend? (y/n)"
read modify_frontend

if [ "$modify_frontend" = "y" ]; then
    echo "Enter the URL of your deployed backend (e.g., https://your-app.onrender.com):"
    read backend_url
    
    # Simple replacement of localhost URL in frontend.py
    sed -i "s|http://127.0.0.1:8000|$backend_url|g" frontend.py
    echo "Updated frontend.py to use backend at: $backend_url"
fi

# Create a simple README
cat > README.md << EOF
# AI Agent Platform

This is the deployment package for the AI Agent Platform.

## Files
- frontend.py - Streamlit frontend
- backend.py - FastAPI backend
- run_app.py - Script to run both services
- requirements.txt - Dependencies
- Procfile - Deployment configuration

## Deployment
Follow the instructions in the DEPLOYMENT.md file in the main repository.
EOF

# Initialize git repository if needed
echo "Do you want to initialize a git repository for deployment? (y/n)"
read init_git

if [ "$init_git" = "y" ]; then
    git init
    git add .
    git commit -m "Initial commit for deployment"
    
    echo "Enter your GitHub username:"
    read github_username
    
    echo "Enter a name for your new repository (e.g., ai-agent-platform):"
    read repo_name
    
    echo "Would you like to create the repository on GitHub and push? (y/n)"
    read create_remote
    
    if [ "$create_remote" = "y" ]; then
        echo "You'll need a GitHub Personal Access Token with 'repo' permissions"
        echo "Visit: https://github.com/settings/tokens to create one if you haven't already"
        
        echo "Enter your GitHub Personal Access Token:"
        read -s github_token
        
        curl -u "$github_username:$github_token" https://api.github.com/user/repos -d "{\"name\":\"$repo_name\", \"description\":\"AI Agent Platform for demo purposes\"}"
        
        git remote add origin "https://github.com/$github_username/$repo_name.git"
        git branch -M main
        git push -u origin main
        
        echo "Repository created and code pushed to GitHub!"
        echo "Repository URL: https://github.com/$github_username/$repo_name"
    else
        echo "Git repository initialized locally. You can push to GitHub later."
    fi
fi

echo "Deployment package prepared successfully in the 'deployment' directory!"
echo "Follow the instructions in DEPLOYMENT.md to deploy your application." 
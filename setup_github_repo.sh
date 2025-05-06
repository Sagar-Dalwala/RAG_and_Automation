#!/bin/bash
# Script to set up a GitHub repository for deploying the Streamlit app

echo "Setting up GitHub repository for Streamlit deployment..."

# Create a new directory for the deployment
mkdir -p streamlit_deployment
cd streamlit_deployment

# Copy the necessary files
cp ../streamlit_app.py .
cp ../requirements-streamlit.txt requirements.txt
cp ../README-deployment.md README.md

# Initialize git repository
git init
git add .
git commit -m "Initial commit for Streamlit deployment"

# Prompt for GitHub username and repository name
echo "Enter your GitHub username:"
read github_username

echo "Enter a name for your new repository (e.g., ai-agent-platform):"
read repo_name

# Create a new repository on GitHub
echo "Creating repository $repo_name on GitHub..."
echo "You'll need to create a Personal Access Token with 'repo' permissions"
echo "Visit: https://github.com/settings/tokens to create one if you haven't already"

echo "Enter your GitHub Personal Access Token:"
read -s github_token

curl -u "$github_username:$github_token" https://api.github.com/user/repos -d "{\"name\":\"$repo_name\", \"description\":\"AI Agent Platform for demo purposes\"}"

# Add the remote repository and push
git remote add origin "https://github.com/$github_username/$repo_name.git"
git branch -M main
git push -u origin main

echo "Repository setup complete!"
echo "Now you can deploy the app to Streamlit Cloud by visiting:"
echo "https://streamlit.io/cloud"
echo "Sign in with your GitHub account and select the repository: $repo_name" 
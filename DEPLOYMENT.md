# Deployment Guide for AI Agent Platform

This guide will walk you through deploying your AI Agent Platform to Render.com, which offers a free tier suitable for demos.

## Prerequisites

1. A GitHub account
2. The following files from your project:
   - `frontend.py` - Your Streamlit frontend
   - `backend.py` - Your FastAPI backend
   - `run_app.py` - The script to run both services
   - `requirements-deployment.txt` - Dependencies for both services
   - `Procfile` - Instructions for the deployment platform

## Steps to Deploy

### 1. Create a GitHub Repository

1. Go to GitHub.com and create a new repository
2. Clone the repository to your local machine
3. Copy the necessary files to the repository folder:
   ```
   cp frontend.py backend.py run_app.py requirements-deployment.txt Procfile /path/to/repo/
   ```
4. Rename `requirements-deployment.txt` to `requirements.txt`:
   ```
   mv requirements-deployment.txt requirements.txt
   ```
5. Commit and push to GitHub:
   ```
   git add .
   git commit -m "Initial commit for deployment"
   git push origin main
   ```

### 2. Deploy to Render.com

1. Create an account at [Render.com](https://render.com)
2. Click "New" and select "Web Service"
3. Connect your GitHub account and select your repository
4. Configure the deployment:
   - Name: `ai-agent-platform` (or any name you prefer)
   - Environment: `Python`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python run_app.py`
5. Select the Free plan
6. Click "Create Web Service"

### 3. Access Your Deployed Application

After the deployment is complete (which may take a few minutes), you can access your application at the URL provided by Render:
```
https://your-app-name.onrender.com
```

## Alternative Deployment Options

### Railway.app

Railway.app is another good option for free hosting:

1. Create an account at [Railway.app](https://railway.app)
2. Create a new project and connect to your GitHub repository
3. Configure the deployment settings similarly to Render

### Streamlit Community Cloud

For a simpler deployment focused just on the frontend:

1. Visit [Streamlit Cloud](https://streamlit.io/cloud)
2. Connect your GitHub repository
3. Select just the `frontend.py` file for deployment

## Troubleshooting

- **Connection Issues**: If the frontend cannot connect to the backend, check the API URL settings
- **Memory Errors**: Free tiers have memory limitations; optimize your code if needed
- **Cold Starts**: Free services may experience "cold starts" when not used for a while 
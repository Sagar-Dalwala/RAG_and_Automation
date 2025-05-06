# AI Agent Platform - Streamlit Deployment Guide

This is a simplified version of the AI Agent Platform for demonstration purposes. It contains all the essential features but uses mock data instead of actual AI model calls.

## Deployment to Streamlit Cloud

To deploy this application to Streamlit Cloud (https://streamlit.io/cloud), follow these steps:

1. Create a GitHub repository and upload the following files:
   - `streamlit_app.py`
   - `requirements-streamlit.txt` (rename to `requirements.txt` when uploading)

2. Go to https://streamlit.io/cloud and sign up/login with your GitHub account.

3. Click "New app" and select the repository containing the Streamlit app.

4. Configure the deployment:
   - Set the main file path to `streamlit_app.py`
   - Leave other settings as default

5. Click "Deploy" and wait for the deployment to complete. 

The app will be available at a URL like: `https://[your-username]-[app-name]-[random-string].streamlit.app`

## Local Development

To run this app locally:

1. Install the requirements:
   ```
   pip install -r requirements-streamlit.txt
   ```

2. Run the Streamlit app:
   ```
   streamlit run streamlit_app.py
   ```

## Using the Demo

1. Sign up for a new account
2. Log in with your credentials
3. Explore the different features:
   - AI Agent Chat
   - RAG Document Q&A
   - Code Assistant
   - Web Automation

Note: This is a demo version with simulated responses. No actual AI model calls are made. 
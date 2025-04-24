# User Guide

This guide provides instructions on how to use the various components of the Advanced RAG and Web Automation System.

## Getting Started

### Installation

1. Ensure you have Python 3.8 or higher installed.
2. Clone the repository to your local machine.
3. Run the setup script to install dependencies and configure the environment:
   ```
   python setup.py
   ```
4. Follow the prompts to enter your API keys.

### Starting the System

You can start individual components or the entire system:

- For the full system: `python run_all.py`
- For just the RAG application: `python advanced_rag_app.py`
- For just the code assistant: `python code_assistant.py`

## Using the RAG System

The RAG (Retrieval-Augmented Generation) system allows you to ask questions based on document content.

### Step 1: Document Input

1. Open the RAG application in your browser (typically at http://localhost:8501).
2. Navigate to the "Input" tab.
3. Select the input type:
   - **Link**: Enter one or more URLs
   - **PDF**: Upload a PDF file
   - **Text**: Enter text directly
   - **DOCX**: Upload a Word document
   - **TXT**: Upload a text file
4. Configure advanced options (optional):
   - **Chunk Size**: Adjust the size of text chunks (larger chunks provide more context, smaller chunks are more precise)
   - **Chunk Overlap**: Set overlap between chunks to maintain context
   - **Include Metadata**: Toggle inclusion of document metadata
5. Click "Process Document" to analyze and embed the content.

### Step 2: Asking Questions

1. Navigate to the "Q&A" tab.
2. Enter your question in the text field.
3. Configure options:
   - **Include Sources**: Toggle to show source information
   - **Confidence Threshold**: Adjust the minimum confidence score for answers
4. Click "Submit Question" to get an answer based on your document.

### Step 3: Exploring Results

1. Review the answer provided by the system.
2. Check source references that show which parts of the document contributed to the answer.
3. Confidence scores indicate how reliable the system considers the answer.

### Step 4: Using Visualizations

1. Navigate to the "Visualization" tab to see:
   - Document structure visualization
   - Content relationship maps
   - Key concept visualization
2. Use the visualization tools to understand document content better.

### Step 5: Analytics

1. Navigate to the "Analytics" tab to see:
   - Query performance metrics
   - Document statistics
   - Content analysis

## Using the Web Automation System

The web automation system helps you extract and analyze content from websites.

### Basic Web Scraping

1. Import the necessary components:
   ```python
   from web_automation import WebAutomation
   ```
2. Initialize the web automation tool:
   ```python
   web_tool = WebAutomation()
   ```
3. Navigate to a URL:
   ```python
   result = web_tool.navigate_to_url("https://example.com")
   ```
4. Extract content:
   ```python
   content = web_tool.extract_page_content()
   ```
5. Close the browser when done:
   ```python
   web_tool.close_browser()
   ```

### Advanced Web Scraping

1. Import the advanced components:
   ```python
   from ai_web_automation import AdvancedWebScraper
   ```
2. Initialize the scraper:
   ```python
   scraper = AdvancedWebScraper()
   ```
3. Navigate to a URL:
   ```python
   result = scraper.navigate_to_url("https://example.com")
   ```
4. Extract structured data:
   ```python
   structured_data = scraper.extract_structured_data()
   ```
5. Analyze content:
   ```python
   analysis = scraper.analyze_content()
   ```
6. Crawl a site:
   ```python
   crawl_results = scraper.crawl_site("https://example.com", max_pages=10, stay_on_domain=True)
   ```
7. Generate visualizations:
   ```python
   graph = scraper._generate_site_graph_visualization()
   ```
8. Close the browser:
   ```python
   scraper.close_browser()
   ```

## Using the Code Assistant

The code assistant helps with code analysis, generation, optimization, and bug finding.

### Using the API Directly

1. Start the code assistant service:
   ```
   python code_assistant.py
   ```
2. Send a request to the API endpoint:
   ```python
   import requests
   import json

   url = "http://localhost:8001/code-assistant"
   payload = {
       "code": "def example_function():\n    print('Hello, world!')",
       "task_type": "Code Analysis",
       "language": "Python"
   }
   response = requests.post(url, json=payload)
   result = response.text
   print(result)
   ```

### Task Types

The code assistant supports different task types:

1. **Code Analysis**:
   - Analyzes code structure and best practices
   - Example payload:
     ```json
     {
       "code": "your code here",
       "task_type": "Code Analysis",
       "language": "Python"
     }
     ```

2. **Code Generation**:
   - Generates code based on a description
   - Example payload:
     ```json
     {
       "code": "Create a function that calculates the factorial of a number",
       "task_type": "Code Generation",
       "language": "Python"
     }
     ```

3. **Code Optimization**:
   - Optimizes existing code for performance and readability
   - Example payload:
     ```json
     {
       "code": "your code here",
       "task_type": "Code Optimization",
       "language": "Python"
     }
     ```

4. **Bug Finding**:
   - Identifies potential bugs and suggests fixes
   - Example payload:
     ```json
     {
       "code": "your code here",
       "task_type": "Bug Finding",
       "language": "Python"
     }
     ```

## Troubleshooting

### Common Issues

1. **API Key Errors**:
   - Ensure your API keys are correctly set in the .env file
   - Verify you have sufficient credits for the services you're using

2. **Browser Automation Issues**:
   - Ensure Chrome is installed on your system
   - Check if you need to update Chrome or chromedriver
   - Try running without headless mode for debugging

3. **Document Processing Errors**:
   - Verify the document format is supported
   - Check if the document is password-protected
   - Try breaking large documents into smaller parts

4. **Performance Issues**:
   - Adjust chunk size for optimal performance
   - Consider using a more powerful machine for large documents
   - Limit the scope of web crawling operations

### Getting Help

If you encounter issues not covered in this guide:

1. Check the console output for error messages
2. Review the documentation for specific components
3. Check the project repository for updates or known issues 
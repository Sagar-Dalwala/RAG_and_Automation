import asyncio
from langchain_groq import ChatGroq
from ai_web_automation import AIWebAutomation

async def test_ai_web_automation():
    # Initialize LLM and AI Web Automation
    llm = ChatGroq(model="llama-3.3-70b-versatile")
    automation = AIWebAutomation(llm)
    
    try:
        # Test simple navigation and content extraction
        command = "Go to amazon.in and extract the main content"
        result = await automation.execute_command(command)
        print(f"\nTest 1 - Simple Navigation:\n{result}")
        
        # Test form interaction
        command = "Go to google.com, search for 'python web automation' and click the first result"
        result = await automation.execute_command(command)
        print(f"\nTest 2 - Form Interaction:\n{result}")
        
        # Test complex interaction sequence
        command = """Go to github.com, wait for the page to load,
        scroll down 500 pixels, then find and extract all repository names"""
        result = await automation.execute_command(command)
        print(f"\nTest 3 - Complex Interaction:\n{result}")
        
    except Exception as e:
        print(f"Test failed with error: {str(e)}")

def main():
    # Run the async test function
    asyncio.run(test_ai_web_automation())

if __name__ == "__main__":
    main()
import os
from web_automation import WebAutomation

def test_web_automation():
    # Initialize web automation
    automation = WebAutomation()
    
    # Test browser initialization
    print("Testing browser initialization...")
    if not automation.start_browser():
        print("Failed to start browser")
        return
    
    try:
        # Test URL navigation
        print("\nTesting URL navigation...")
        result = automation.navigate_to_url("https://www.youtube.com")
        if "error" in result:
            print(f"Navigation failed: {result['error']}")
            return
        print("Navigation successful")
        
        # Test content extraction
        print("\nTesting content extraction...")
        content = automation.extract_page_content()
        if "error" in content:
            print(f"Content extraction failed: {content['error']}")
        else:
            print(f"Page title: {content['title']}")
        
        # Test screenshot functionality
        print("\nTesting screenshot capture...")
        screenshot_path = os.path.join(os.path.dirname(__file__), "test_screenshot.png")
        screenshot_result = automation.take_screenshot(screenshot_path)
        
        if "error" in screenshot_result:
            print(f"Screenshot failed: {screenshot_result['error']}")
        else:
            print(f"Screenshot saved to: {screenshot_result['filename']}")
            if os.path.exists(screenshot_path):
                print(f"Screenshot file exists, size: {os.path.getsize(screenshot_path)} bytes")
            else:
                print("Screenshot file was not created")
    
    except Exception as e:
        print(f"\nTest failed with error: {str(e)}")
    
    finally:
        # Clean up
        print("\nClosing browser...")
        automation.close_browser()

if __name__ == "__main__":
    test_web_automation()
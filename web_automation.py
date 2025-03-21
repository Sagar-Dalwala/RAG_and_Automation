from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from bs4 import BeautifulSoup
import time
import json

class WebAutomation:
    def __init__(self):
        self.options = webdriver.ChromeOptions()
        self.options.add_argument('--headless')
        self.options.add_argument('--no-sandbox')
        self.options.add_argument('--disable-dev-shm-usage')
        self.driver = None
        
    def start_browser(self):
        """Initialize the browser session"""
        try:
            self.driver = webdriver.Chrome(options=self.options)
            return True
        except WebDriverException as e:
            print(f"Error starting browser: {str(e)}")
            return False
            
    def close_browser(self):
        """Close the browser session"""
        if self.driver:
            self.driver.quit()
            self.driver = None
            
    def navigate_to_url(self, url):
        """Navigate to a specific URL"""
        try:
            if not self.driver:
                if not self.start_browser():
                    return {"error": "Failed to start browser"}
                    
            self.driver.get(url)
            return {"success": True, "url": url}
        except Exception as e:
            return {"error": str(e)}
            
    def extract_page_content(self):
        """Extract content from the current page"""
        try:
            # Wait for body to be present
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Get page source and parse with BeautifulSoup
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            
            # Extract text content
            text_content = soup.get_text(separator='\n', strip=True)
            
            # Extract links
            links = [{
                'text': a.get_text(strip=True),
                'href': a.get('href')
            } for a in soup.find_all('a', href=True)]
            
            # Extract metadata
            title = soup.title.string if soup.title else ''
            meta_description = soup.find('meta', {'name': 'description'})
            description = meta_description['content'] if meta_description else ''
            
            return {
                "success": True,
                "title": title,
                "description": description,
                "content": text_content,
                "links": links
            }
            
        except TimeoutException:
            return {"error": "Timeout waiting for page to load"}
        except Exception as e:
            return {"error": str(e)}
            
    def search_in_page(self, query):
        """Search for specific content in the page"""
        try:
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            results = []
            
            # Search in text content
            for element in soup.find_all(text=lambda text: query.lower() in text.lower()):
                results.append({
                    "type": "text",
                    "content": element.strip(),
                    "context": element.parent.get_text(strip=True)
                })
                
            return {"success": True, "results": results}
        except Exception as e:
            return {"error": str(e)}
            
    def take_screenshot(self, filename):
        """Take a screenshot of the current page"""
        try:
            self.driver.save_screenshot(filename)
            return {"success": True, "filename": filename}
        except Exception as e:
            return {"error": str(e)}
            
    def scroll_page(self, direction="down"):
        """Scroll the page up or down"""
        try:
            if direction == "down":
                self.driver.execute_script("window.scrollBy(0, window.innerHeight);")
            else:
                self.driver.execute_script("window.scrollBy(0, -window.innerHeight);")
            return {"success": True}
        except Exception as e:
            return {"error": str(e)}
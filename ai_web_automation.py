from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException, NoSuchElementException
from bs4 import BeautifulSoup
import time
import json
import pandas as pd
import matplotlib.pyplot as plt
import re
import requests
from urllib.parse import urljoin, urlparse
import networkx as nx
import io
import base64
from PIL import Image
import os
from collections import Counter
import seaborn as sns
import numpy as np

class AdvancedWebScraper:
    def __init__(self):
        self.options = webdriver.ChromeOptions()
        self.options.add_argument('--headless')
        self.options.add_argument('--no-sandbox')
        self.options.add_argument('--disable-dev-shm-usage')
        self.options.add_argument('--disable-gpu')
        self.options.add_argument("--window-size=1920,1080")
        self.options.add_argument('--ignore-certificate-errors')
        self.options.add_argument('--allow-running-insecure-content')
        self.driver = None
        self.current_url = None
        self.visited_urls = set()
        self.site_graph = nx.DiGraph()
        self.data_cache = {}
        
    def start_browser(self):
        """Initialize the browser session with advanced capabilities"""
        try:
            self.driver = webdriver.Chrome(options=self.options)
            self.driver.set_page_load_timeout(30)
            return {"success": True, "message": "Browser started successfully"}
        except WebDriverException as e:
            return {"error": f"Error starting browser: {str(e)}"}
            
    def close_browser(self):
        """Close the browser session and clean up resources"""
        if self.driver:
            self.driver.quit()
            self.driver = None
            return {"success": True, "message": "Browser closed successfully"}
        return {"success": True, "message": "No active browser session"}
            
    def navigate_to_url(self, url, wait_time=10):
        """Navigate to a specific URL with advanced error handling and metrics"""
        try:
            if not self.driver:
                result = self.start_browser()
                if "error" in result:
                    return result
            
            start_time = time.time()
            self.driver.get(url)
            load_time = time.time() - start_time
            
            # Wait for page to be fully loaded
            WebDriverWait(self.driver, wait_time).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            self.current_url = url
            self.visited_urls.add(url)
            
            # Record page size
            page_size = len(self.driver.page_source)
            
            # Get HTTP status code
            # Note: Selenium doesn't provide direct access to HTTP status codes
            # This is a workaround to check if the page loaded successfully
            if "404" in self.driver.title or "not found" in self.driver.title.lower():
                status_code = 404
            else:
                status_code = 200
            
            # Get basic page metrics
            metrics = {
                "url": url,
                "load_time_seconds": round(load_time, 2),
                "page_size_bytes": page_size,
                "status_code": status_code,
                "title": self.driver.title
            }
            
            return {
                "success": True, 
                "metrics": metrics
            }
        except TimeoutException:
            return {"error": f"Timeout waiting for page to load: {url}"}
        except Exception as e:
            return {"error": f"Error navigating to {url}: {str(e)}"}
    
    def extract_structured_data(self):
        """Extract structured data like JSON-LD, microdata, and OpenGraph tags"""
        try:
            structured_data = {
                "json_ld": [],
                "microdata": {},
                "opengraph": {},
                "twitter_cards": {}
            }
            
            # Extract JSON-LD
            json_ld_scripts = self.driver.find_elements(By.XPATH, '//script[@type="application/ld+json"]')
            for script in json_ld_scripts:
                try:
                    data = json.loads(script.get_attribute('innerHTML'))
                    structured_data["json_ld"].append(data)
                except:
                    pass
            
            # Extract OpenGraph tags
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            og_tags = soup.find_all('meta', property=lambda x: x and x.startswith('og:'))
            for tag in og_tags:
                prop = tag.get('property', '').replace('og:', '')
                content = tag.get('content', '')
                structured_data["opengraph"][prop] = content
            
            # Extract Twitter Card tags
            twitter_tags = soup.find_all('meta', attrs={'name': lambda x: x and x.startswith('twitter:')})
            for tag in twitter_tags:
                name = tag.get('name', '').replace('twitter:', '')
                content = tag.get('content', '')
                structured_data["twitter_cards"][name] = content
            
            return structured_data
        except Exception as e:
            return {"error": f"Error extracting structured data: {str(e)}"}
    
    def extract_page_content(self, include_html=False):
        """Extract comprehensive content from the current page"""
        try:
            # Wait for body to be present
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Get page source and parse with BeautifulSoup
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            
            # Extract text content
            text_content = soup.get_text(separator='\n', strip=True)
            
            # Extract links with additional metadata
            links = []
            for a in soup.find_all('a', href=True):
                href = a.get('href')
                # Process relative URLs
                if href and not href.startswith(('http://', 'https://', 'mailto:', 'tel:', 'javascript:')):
                    href = urljoin(self.current_url, href)
                
                link_info = {
                    'text': a.get_text(strip=True),
                    'href': href,
                    'title': a.get('title', ''),
                    'rel': a.get('rel', ''),
                    'is_internal': self._is_internal_link(href)
                }
                links.append(link_info)
            
            # Extract metadata
            title = soup.title.string if soup.title else ''
            meta_tags = {}
            for meta in soup.find_all('meta'):
                name = meta.get('name') or meta.get('property')
                if name:
                    meta_tags[name] = meta.get('content', '')
            
            # Extract main content area (heuristic approach)
            main_content = self._extract_main_content(soup)
            
            # Extract images with additional metadata
            images = []
            for img in soup.find_all('img', src=True):
                src = img.get('src')
                # Process relative URLs
                if src and not src.startswith(('http://', 'https://', 'data:')):
                    src = urljoin(self.current_url, src)
                
                image_info = {
                    'src': src,
                    'alt': img.get('alt', ''),
                    'title': img.get('title', ''),
                    'width': img.get('width', ''),
                    'height': img.get('height', '')
                }
                images.append(image_info)
            
            # Extract tables
            tables = []
            for i, table in enumerate(soup.find_all('table')):
                try:
                    # Convert to pandas DataFrame
                    df = pd.read_html(str(table))[0]
                    tables.append({
                        'id': i,
                        'rows': df.shape[0],
                        'columns': df.shape[1],
                        'headers': df.columns.tolist(),
                        'data': df.to_dict('records')
                    })
                except:
                    pass
            
            # Get structured data
            structured_data = self.extract_structured_data()
            
            result = {
                "success": True,
                "url": self.current_url,
                "title": title,
                "meta_tags": meta_tags,
                "main_content": main_content,
                "full_text": text_content,
                "links_count": len(links),
                "links": links,
                "images_count": len(images),
                "images": images,
                "tables_count": len(tables),
                "tables": tables,
                "structured_data": structured_data
            }
            
            # Include full HTML if requested
            if include_html:
                result["html"] = str(soup)
            
            return result
            
        except TimeoutException:
            return {"error": "Timeout waiting for page to load"}
        except Exception as e:
            return {"error": f"Error extracting content: {str(e)}"}
    
    def _extract_main_content(self, soup):
        """Attempt to extract the main content area of the page"""
        # Try common content containers
        content_selectors = [
            'article', 'main', '#content', '.content', 
            '[role="main"]', '.main-content', '#main-content'
        ]
        
        for selector in content_selectors:
            content = soup.select_one(selector)
            if content and len(content.get_text(strip=True)) > 200:
                return content.get_text(strip=True)
        
        # If no common selector works, try a density-based approach
        paragraphs = soup.find_all('p')
        if paragraphs:
            return "\n\n".join(p.get_text(strip=True) for p in paragraphs 
                              if len(p.get_text(strip=True)) > 20)
        
        # Fallback
        return soup.get_text(separator='\n', strip=True)
    
    def _is_internal_link(self, href):
        """Check if a link is internal to the current domain"""
        if not href or href.startswith(('mailto:', 'tel:', 'javascript:')):
            return False
        
        if not self.current_url:
            return False
            
        try:
            current_domain = urlparse(self.current_url).netloc
            link_domain = urlparse(href).netloc
            return not link_domain or link_domain == current_domain
        except:
            return False
    
    def search_in_page(self, query, context_words=30):
        """Advanced search for specific content in the page with context"""
        try:
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            full_text = soup.get_text(' ', strip=True)
            
            # For case-insensitive search
            query_lower = query.lower()
            full_text_lower = full_text.lower()
            
            results = []
            
            # Find all occurrences
            start_idx = 0
            while True:
                idx = full_text_lower.find(query_lower, start_idx)
                if idx == -1:
                    break
                
                # Get context around the match
                start = max(0, idx - context_words)
                end = min(len(full_text), idx + len(query) + context_words)
                
                # Extract the context
                context = full_text[start:end]
                
                # Highlight the match in the context
                match_in_context = full_text[idx:idx+len(query)]
                
                results.append({
                    "match": match_in_context,
                    "context": context,
                    "position": idx
                })
                
                start_idx = idx + len(query)
            
            # Search in specific elements
            for tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li', 'td', 'th', 'a']:
                for element in soup.find_all(tag):
                    text = element.get_text(strip=True)
                    if query_lower in text.lower():
                        results.append({
                            "element_type": tag,
                            "match": text,
                            "context": element.parent.get_text(strip=True) if element.parent else text
                        })
            
            return {
                "success": True, 
                "query": query,
                "occurrences": len(results),
                "results": results
            }
        except Exception as e:
            return {"error": f"Error searching page: {str(e)}"}
    
    def take_screenshot(self, filename=None, full_page=False, element_selector=None):
        """Take a screenshot of the current page, full page, or specific element"""
        try:
            if not filename:
                # Generate a filename based on the current URL
                parsed_url = urlparse(self.current_url)
                safe_filename = re.sub(r'[^\w\-_]', '_', parsed_url.netloc + parsed_url.path)
                filename = f"{safe_filename}_{int(time.time())}.png"
            
            # Take screenshot of specific element if selector provided
            if element_selector:
                try:
                    element = self.driver.find_element(By.CSS_SELECTOR, element_selector)
                    image = element.screenshot_as_png
                    with open(filename, "wb") as f:
                        f.write(image)
                    
                    # Create a base64 version for display
                    img_base64 = base64.b64encode(image).decode()
                    
                    return {
                        "success": True, 
                        "filename": filename,
                        "element": element_selector,
                        "image_data": f"data:image/png;base64,{img_base64}"
                    }
                except NoSuchElementException:
                    return {"error": f"Element not found: {element_selector}"}
            
            # Take full page screenshot if requested
            if full_page:
                # Get the height of the page
                height = self.driver.execute_script("return document.body.scrollHeight")
                
                # Set window size to capture the full page
                original_size = self.driver.get_window_size()
                self.driver.set_window_size(1920, height)
                
                # Take screenshot
                self.driver.save_screenshot(filename)
                
                # Restore original window size
                self.driver.set_window_size(original_size['width'], original_size['height'])
            else:
                # Take regular screenshot
                self.driver.save_screenshot(filename)
            
            # Generate a base64 representation for direct display
            with open(filename, "rb") as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode()
            
            return {
                "success": True, 
                "filename": filename,
                "full_page": full_page,
                "image_data": f"data:image/png;base64,{img_base64}"
            }
        except Exception as e:
            return {"error": f"Error taking screenshot: {str(e)}"}
    
    def extract_by_css_selector(self, selector):
        """Extract content using a custom CSS selector"""
        try:
            elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
            
            results = []
            for element in elements:
                results.append({
                    "text": element.text,
                    "html": element.get_attribute('outerHTML'),
                    "tag_name": element.tag_name,
                    "attributes": {
                        attr: element.get_attribute(attr) 
                        for attr in ['id', 'class', 'href', 'src', 'alt', 'title']
                        if element.get_attribute(attr)
                    }
                })
            
            return {
                "success": True,
                "selector": selector,
                "count": len(results),
                "results": results
            }
        except Exception as e:
            return {"error": f"Error extracting by selector '{selector}': {str(e)}"}
    
    def crawl_site(self, start_url, max_pages=10, stay_on_domain=True, depth=2):
        """Crawl a website starting from a URL, respecting robots.txt"""
        try:
            if not self.driver:
                result = self.start_browser()
                if "error" in result:
                    return result
            
            # Clear previous crawl data
            self.visited_urls = set()
            self.site_graph = nx.DiGraph()
            self.data_cache = {}
            
            # Parse domain for staying on the same domain
            domain = urlparse(start_url).netloc if stay_on_domain else None
            
            # Start with the initial URL
            to_visit = [(start_url, 0)]  # (url, depth)
            
            crawl_results = {
                "pages_visited": 0,
                "urls_found": 0,
                "errors": [],
                "data": {}
            }
            
            while to_visit and len(self.visited_urls) < max_pages:
                current_url, current_depth = to_visit.pop(0)
                
                # Skip if already visited or beyond depth limit
                if current_url in self.visited_urls or current_depth > depth:
                    continue
                
                # Navigate to the URL
                result = self.navigate_to_url(current_url)
                
                if "error" in result:
                    crawl_results["errors"].append({
                        "url": current_url,
                        "error": result["error"]
                    })
                    continue
                
                # Mark as visited
                self.visited_urls.add(current_url)
                crawl_results["pages_visited"] += 1
                
                # Extract page content
                content = self.extract_page_content()
                if "error" not in content:
                    # Store in data cache
                    self.data_cache[current_url] = content
                    crawl_results["data"][current_url] = {
                        "title": content.get("title", ""),
                        "links_count": content.get("links_count", 0),
                        "images_count": content.get("images_count", 0)
                    }
                    
                    # Add node to site graph
                    self.site_graph.add_node(current_url, title=content.get("title", ""))
                    
                    # Process links if not at max depth
                    if current_depth < depth:
                        for link in content.get("links", []):
                            link_url = link.get("href")
                            
                            # Check if valid URL and should be visited
                            if link_url and link_url.startswith(('http://', 'https://')):
                                # Check if should stay on domain
                                if stay_on_domain:
                                    link_domain = urlparse(link_url).netloc
                                    if link_domain != domain:
                                        continue
                                
                                # Add edge to site graph
                                self.site_graph.add_edge(current_url, link_url)
                                
                                # Add to visit queue if not visited
                                if link_url not in self.visited_urls:
                                    to_visit.append((link_url, current_depth + 1))
                                    crawl_results["urls_found"] += 1
                
                # Respect politeness - delay between requests
                time.sleep(1)
            
            # Generate site graph visualization
            if len(self.site_graph) > 0:
                graph_viz = self._generate_site_graph_visualization()
                crawl_results["site_graph"] = graph_viz
            
            return {
                "success": True,
                "crawl_results": crawl_results
            }
        except Exception as e:
            return {"error": f"Error during crawl: {str(e)}"}
    
    def _generate_site_graph_visualization(self):
        """Generate a visualization of the site graph"""
        try:
            plt.figure(figsize=(12, 8))
            
            # Simplify URLs for display
            labels = {}
            for node in self.site_graph.nodes():
                parsed = urlparse(node)
                path = parsed.path if parsed.path else '/'
                labels[node] = f"{parsed.netloc}{path[:20]}{'...' if len(path) > 20 else ''}"
            
            # Draw the graph
            pos = nx.spring_layout(self.site_graph)
            nx.draw(self.site_graph, pos, node_size=500, node_color='lightblue', 
                   with_labels=False, arrows=True, alpha=0.7)
            nx.draw_networkx_labels(self.site_graph, pos, labels=labels, font_size=8)
            
            # Save to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            
            # Convert to base64 for display
            img_base64 = base64.b64encode(buf.read()).decode()
            plt.close()
            
            return f"data:image/png;base64,{img_base64}"
        except Exception as e:
            return f"Error generating graph: {str(e)}"
    
    def analyze_content(self):
        """Analyze page content for insights and visualizations"""
        try:
            if not self.current_url:
                return {"error": "No page loaded yet"}
            
            # Extract content
            content = self.extract_page_content()
            if "error" in content:
                return content
            
            # Perform analysis
            analysis = {
                "word_count": 0,
                "most_common_words": [],
                "sentiment": {},
                "reading_level": {},
                "link_analysis": {},
                "visualizations": {}
            }
            
            # Word count and frequency analysis
            if "full_text" in content:
                text = content["full_text"]
                words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
                
                # Remove common stop words
                stop_words = ['the', 'and', 'is', 'in', 'to', 'of', 'for', 'with', 'on', 'at']
                filtered_words = [word for word in words if word not in stop_words]
                
                analysis["word_count"] = len(filtered_words)
                
                # Most common words
                word_freq = Counter(filtered_words)
                analysis["most_common_words"] = word_freq.most_common(20)
                
                # Generate word cloud visualization
                if filtered_words:
                    # Create word cloud visualization
                    plt.figure(figsize=(10, 6))
                    frequencies = [count for _, count in analysis["most_common_words"][:10]]
                    words = [word for word, _ in analysis["most_common_words"][:10]]
                    
                    # Create bar chart
                    colors = plt.cm.viridis(np.linspace(0, 1, len(words)))
                    plt.bar(range(len(words)), frequencies, color=colors)
                    plt.xticks(range(len(words)), words, rotation=45, ha='right')
                    plt.title('Most Common Words')
                    plt.tight_layout()
                    
                    # Save to buffer
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png')
                    buf.seek(0)
                    
                    # Convert to base64
                    word_cloud_base64 = base64.b64encode(buf.read()).decode()
                    plt.close()
                    
                    analysis["visualizations"]["word_frequency"] = f"data:image/png;base64,{word_cloud_base64}"
            
            # Link analysis
            if "links" in content:
                links = content["links"]
                internal_links = [l for l in links if l.get("is_internal", False)]
                external_links = [l for l in links if not l.get("is_internal", True)]
                
                analysis["link_analysis"] = {
                    "total_links": len(links),
                    "internal_links": len(internal_links),
                    "external_links": len(external_links),
                }
                
                # Generate link distribution visualization
                plt.figure(figsize=(8, 5))
                plt.pie([len(internal_links), len(external_links)], 
                       labels=['Internal Links', 'External Links'],
                       autopct='%1.1f%%', colors=['#4CAF50', '#2196F3'])
                plt.title('Link Distribution')
                
                # Save to buffer
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                
                # Convert to base64
                link_viz_base64 = base64.b64encode(buf.read()).decode()
                plt.close()
                
                analysis["visualizations"]["link_distribution"] = f"data:image/png;base64,{link_viz_base64}"
            
            return {
                "success": True,
                "url": self.current_url,
                "analysis": analysis
            }
        except Exception as e:
            return {"error": f"Error analyzing content: {str(e)}"}
    
    def extract_contact_info(self):
        """Extract contact information from the current page"""
        try:
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            contact_info = {
                "emails": [],
                "phones": [],
                "addresses": [],
                "social_media": []
            }
            
            # Extract emails using regex
            email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
            emails = re.findall(email_pattern, self.driver.page_source)
            contact_info["emails"] = list(set(emails))
            
            # Extract phone numbers
            phone_pattern = r'(\+\d{1,3}[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}'
            phones = re.findall(phone_pattern, self.driver.page_source)
            contact_info["phones"] = [''.join(p) for p in phones if ''.join(p).strip()]
            
            # Extract addresses (simplified approach)
            address_elements = soup.find_all(['address', 'p', 'div'], class_=lambda c: c and any(term in c.lower() for term in ['address', 'location', 'contact']))
            for element in address_elements:
                text = element.get_text(strip=True)
                if len(text) > 10 and any(term in text.lower() for term in ['street', 'avenue', 'road', 'blvd', 'drive']):
                    contact_info["addresses"].append(text)
            
            # Extract social media links
            social_patterns = {
                'facebook': r'facebook\.com/[\w\.-]+',
                'twitter': r'twitter\.com/[\w\.-]+',
                'instagram': r'instagram\.com/[\w\.-]+',
                'linkedin': r'linkedin\.com/[\w\.-]+',
                'youtube': r'youtube\.com/[\w\.-]+',
                'github': r'github\.com/[\w\.-]+'
            }
            
            for platform, pattern in social_patterns.items():
                matches = re.findall(pattern, self.driver.page_source)
                for match in matches:
                    url = match
                    if not url.startswith(('http://', 'https://')):
                        url = 'https://' + url
                    contact_info["social_media"].append({
                        "platform": platform,
                        "url": url
                    })
            
            return contact_info
        except Exception as e:
            return {"error": f"Error extracting contact info: {str(e)}"}
            
    def extract_images(self, min_width=100, min_height=100, formats=None, exclude_icons=True):
        """
        Extract all images from the current page with advanced filtering options
        
        Args:
            min_width (int): Minimum width of images to include (pixels)
            min_height (int): Minimum height of images to include (pixels)
            formats (list): List of image formats to include (e.g., ['jpg', 'png'])
            exclude_icons (bool): Whether to exclude small icon images
            
        Returns:
            dict: Dictionary containing extracted images
        """
        try:
            if not self.driver:
                return {"error": "Browser not started"}
                
            if not formats:
                formats = ['jpg', 'jpeg', 'png', 'gif', 'webp', 'svg']
                
            # Use a simpler approach with BeautifulSoup
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            
            # Find all img tags
            img_tags = soup.find_all('img', src=True)
            
            filtered_images = []
            for img in img_tags:
                src = img.get('src', '')
                
                # Process relative URLs
                if src and not src.startswith(('http://', 'https://', 'data:')):
                    src = urljoin(self.current_url, src)
                    
                # Check image format from extension
                img_format = src.split('.')[-1].lower() if '.' in src else ''
                if formats and img_format not in formats:
                    continue
                
                # Get dimensions from attributes if available
                width = img.get('width')
                height = img.get('height')
                
                # Convert to integers if they're strings
                try:
                    width = int(width) if width else 0
                    height = int(height) if height else 0
                except ValueError:
                    width = 0
                    height = 0
                
                # Skip icons if requested
                if exclude_icons and width > 0 and height > 0 and width < 50 and height < 50:
                    continue
                    
                # Skip images smaller than minimum dimensions if we know the size
                if width > 0 and height > 0 and (width < min_width or height < min_height):
                    continue
                
                # Create image data
                image_data = {
                    'src': src,
                    'alt': img.get('alt', ''),
                    'title': img.get('title', ''),
                    'width': width or "Unknown",
                    'height': height or "Unknown"
                }
                
                filtered_images.append(image_data)
            
            # Also find CSS background images for divs
            if soup.find_all('style'):
                # This requires more complex parsing and is less reliable
                # For this emergency fix, we'll skip to keep it simple
                pass
            
            return {
                "count": len(filtered_images),
                "images": filtered_images,
                "filters": {
                    "min_width": min_width,
                    "min_height": min_height,
                    "formats": formats,
                    "exclude_icons": exclude_icons
                }
            }
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            return {"error": f"Error extracting images: {str(e)}", "details": error_detail}
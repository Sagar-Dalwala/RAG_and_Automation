# Advanced Web Scraping Module

This module provides comprehensive web scraping, analysis and visualization capabilities for the RAG application. It's designed to be presented in a college seminar setting to demonstrate sophisticated data extraction techniques.

## Key Features

### 1. Basic Web Scraping
- **Automated Browser Navigation**: Headless browser automation using Selenium
- **Metrics Collection**: Performance data including load time and page size
- **Content Extraction**: Title, meta tags, main content, links, and images
- **Screenshot Capability**: Full-page and element-specific screenshots

### 2. Content Analysis
- **Text Analysis**: Word frequency, most common terms, content density
- **Link Analysis**: Internal vs. external link distribution with visualization
- **Data Visualization**: Interactive charts and graphs of webpage content
- **Structured Data Extraction**: JSON-LD, OpenGraph, Twitter Cards

### 3. Website Crawling
- **Multi-page Crawling**: Explore website structure with configurable depth and limits
- **Site Structure Visualization**: Generate interactive site maps
- **Domain Restriction**: Option to stay within the same domain
- **Content Aggregation**: Collect and analyze data across multiple pages

### 4. Custom Data Extraction
- **CSS Selector Queries**: Extract specific elements using custom selectors
- **Template Selectors**: Pre-defined selectors for common website elements
- **Data Export**: Download extracted data in JSON format
- **HTML Inspection**: View raw HTML for extracted elements

### 5. Contact Information Extraction
- **Email Detection**: Find and extract email addresses
- **Phone Numbers**: Detect various phone number formats
- **Physical Addresses**: Extract potential physical address information
- **Social Media Links**: Identify and categorize social media profiles

## Technical Details

The module uses a combination of:
- **Selenium**: For browser automation and JavaScript rendering
- **BeautifulSoup**: For HTML parsing and content extraction
- **NetworkX**: For site structure visualization
- **Pandas**: For data manipulation and analysis
- **Matplotlib/Seaborn**: For data visualization
- **Regular Expressions**: For pattern matching and extraction

## Usage in Your Project

The Advanced Web Scraper is integrated into the 4th tab of the application's interface. Users can:
1. Enter URLs to analyze
2. Extract and visualize website content
3. Crawl multi-page websites
4. Export structured data
5. Search for specific content patterns

This module significantly enhances the capabilities of the RAG application by providing rich, structured data from web sources that can be used for more comprehensive analysis. 
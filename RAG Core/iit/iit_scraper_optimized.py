import asyncio
import aiohttp
import aiofiles
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, unquote
import os
import json
import logging
import re
import hashlib

# Configuration
BASE_URL = "https://iit.tn"
ALLOWED_DOMAIN = "iit.tn"
OUTPUT_DIR = "scraped_data_categorized"

# Keep only document and text extensions needed for RAG
FILE_EXTENSIONS = {
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".csv",
    ".ppt", ".pptx", ".txt", ".xml", ".zip", ".rar"
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

class IITScraperOptimized:
    def __init__(self, start_url, max_concurrent=20):
        self.start_url = start_url
        self.visited_urls = set()
        # Track hashes of content to avoid redundant data
        self.seen_content_hashes = set()
        
        self.queue = asyncio.Queue()
        self.queue.put_nowait(start_url)
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.session = None

    def is_valid_url(self, url):
        parsed = urlparse(url)
        if parsed.netloc and ALLOWED_DOMAIN not in parsed.netloc:
            return False
        if url.startswith(("mailto:", "tel:", "javascript:")):
            return False
        return True

    def get_category_from_url(self, url):
        """Extracts category from the URL path."""
        parsed = urlparse(url)
        path_parts = [p for p in parsed.path.strip('/').split('/') if p]
        
        if not path_parts:
            return "general"
        
        category = path_parts[0].lower()
        category = re.sub(r'[^a-z0-9]', '_', category)
        return category if category else "general"

    def setup_category_dirs(self, category):
        cat_dir = os.path.join(OUTPUT_DIR, category)
        file_dir = os.path.join(cat_dir, "file")
        text_dir = os.path.join(cat_dir, "text")
        
        os.makedirs(file_dir, exist_ok=True)
        os.makedirs(text_dir, exist_ok=True)
        return file_dir, text_dir

    def clean_text(self, text):
        """Aggressive text cleaning for pristine RAG data."""
        # Replace multiple newlines or tabs with a single space or neat newline
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        return text.strip()

    async def download_file(self, url, category, response):
        """Securely download a binary document/PDF and prevent duplicates using hashing."""
        file_dir, _ = self.setup_category_dirs(category)
        
        # Read the binary content
        content = await response.read()
        content_hash = hashlib.md5(content).hexdigest()
        
        # Skip if we already downloaded this exact same file somewhere else (Redundancy check)
        if content_hash in self.seen_content_hashes:
            logging.debug(f"Duplicate file skipped: {url}")
            return None
        
        self.seen_content_hashes.add(content_hash)

        # Generate safe filename
        url_filename = unquote(url.split("/")[-1])
        if not url_filename or "?" in url_filename or len(url_filename) > 60:
            ext = ".pdf" if "pdf" in response.headers.get("Content-Type", "") else ".doc"
            url_filename = f"document_{content_hash[:8]}{ext}"
            
        file_path = os.path.join(file_dir, url_filename)
        
        try:
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(content)
            logging.info(f"Downloaded File [{category}]: {url_filename}")
            return file_path
        except Exception as e:
            logging.error(f"Failed to save file {file_path}: {e}")
            return None

    async def save_text(self, page_data, category):
        """Saves clean HTML text. Rejects empty or duplicated texts."""
        content = page_data.get("content", "")
        if not content or len(content) < 50: # Skip uselessly short pages
            return
            
        # Redundancy check for text: Hash the actual readable text
        text_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        if text_hash in self.seen_content_hashes:
            logging.debug(f"Duplicate text content skipped for URL: {page_data['url']}")
            return
            
        self.seen_content_hashes.add(text_hash)
            
        _, text_dir = self.setup_category_dirs(category)
        
        safe_title = re.sub(r'[^a-zA-Z0-9]', '_', page_data['title'])[:40]
        if not safe_title:
            safe_title = "html_page"
            
        filename = f"{safe_title}_{text_hash[:8]}.json"
        file_path = os.path.join(text_dir, filename)
        
        try:
            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(page_data, ensure_ascii=False, indent=4))
            logging.info(f"Saved Text [{category}]: {filename}")
        except Exception as e:
            logging.error(f"Failed to save text {file_path}: {e}")

    def extract_page_data(self, html_content, url):
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Aggressively remove noise elements
        for element in soup(["script", "style", "nav", "footer", "header", "noscript", "svg", "button", "iframe"]):
            element.extract()
            
        title = soup.title.string if soup.title else "No Title"
        title = self.clean_text(title)
        
        # Extract purely readable paragraphs, lists, and headings for highest RAG quality
        text_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'td', 'th', 'span', 'div'])
        
        extracted_text = []
        for elem in text_elements:
            # Get direct text, ignoring deeply nested noise if possible
            clean_str = self.clean_text(elem.get_text(separator=' ', strip=True))
            if clean_str and len(clean_str) > 5: # Only keep meaningful phrases
                extracted_text.append(clean_str)
                
        # Deduplicate consecutive identical lines (common in badly structured HTML)
        final_text = []
        for line in extracted_text:
            if not final_text or final_text[-1] != line:
                final_text.append(line)
        
        joined_text = "\n".join(final_text)
        
        return {
            "url": url,
            "title": title,
            "category": self.get_category_from_url(url),
            "content": joined_text
        }
        
    async def process_page(self, url):
        # Intercept nested PDF viewer URLs (e.g. ?file=https%3A%2F%2F...pdf)
        parsed_url = urlparse(url)
        from urllib.parse import parse_qs
        query_params = parse_qs(parsed_url.query)
        
        is_viewer = False
        for param, values in query_params.items():
            for val in values:
                # If a query parameter is a nested URL pointing to a PDF
                if val.startswith("http") and ".pdf" in val.lower():
                    is_viewer = True
                    if self.is_valid_url(val) and val not in self.visited_urls:
                        self.visited_urls.add(val)
                        self.queue.put_nowait(val)
        
        # If this URL is just a PDF viewer or nested PDF wrapper, abort scraping its HTML
        if is_viewer or "get_viewer" in url.lower() or "pdfjs" in url.lower():
            return
            
        category = self.get_category_from_url(url)
        
        try:
            async with self.semaphore:
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
                async with self.session.get(url, headers=headers, timeout=20, allow_redirects=True) as response:
                    if response.status != 200:
                        return

                    content_type = response.headers.get("Content-Type", "").lower()
                    
                    if any(media in content_type for media in ['image/', 'video/', 'audio/']):
                        return
                    
                    is_doc = any(doc_type in content_type for doc_type in ['application/pdf', 'application/msword', 'application/vnd', 'application/octet-stream', 'application/zip'])
                    if is_doc:
                        await self.download_file(url, category, response)
                        return

                    if "text/html" in content_type:
                        html_content = await response.text()
                        
                        page_data = self.extract_page_data(html_content, url)
                        await self.save_text(page_data, category)

                        soup = BeautifulSoup(html_content, 'html.parser')
                        for element in soup.find_all(['a', 'link', 'iframe']):
                            link = element.get('href') or element.get('src')
                            if link:
                                full_url = urljoin(url, link).split('#')[0]
                                
                                skip_exts = {".jpg", ".png", ".gif", ".svg", ".webp", ".mp4", ".css", ".js"}
                                if os.path.splitext(urlparse(full_url).path)[1].lower() in skip_exts:
                                    continue

                                if self.is_valid_url(full_url) and full_url not in self.visited_urls:
                                    self.visited_urls.add(full_url)
                                    self.queue.put_nowait(full_url)
                                    
        except Exception as e:
            logging.debug(f"Error processing {url}: {e}")

    async def run(self):
        self.visited_urls.add(self.start_url)
        # Using a custom connector to bypass SSL issues
        connector = aiohttp.TCPConnector(ssl=False)
        async with aiohttp.ClientSession(connector=connector) as session:
            self.session = session
            
            # Start 30 worker tasks
            workers = [asyncio.create_task(self.worker()) for _ in range(30)]
            
            await self.queue.join()
            
            for w in workers:
                w.cancel()
                
            logging.info("Optimized Crawling completed successfully.")

    async def worker(self):
        while True:
            url = await self.queue.get()
            try:
                await self.process_page(url)
            finally:
                self.queue.task_done()

if __name__ == "__main__":
    if os.name == 'nt':
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        except Exception:
            pass
        
    logging.info("Starting IIT Scraper OPTIMIZED (No Redundancy, Strict Content-Type)...")
    scraper = IITScraperOptimized(start_url=BASE_URL, max_concurrent=20)
    
    try:
        asyncio.run(scraper.run())
    except KeyboardInterrupt:
        logging.info("Interrupted by user. Exiting...")
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        
    logging.info("Done! Check 'scraped_data_categorized' folder.")

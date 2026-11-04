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
    ".ppt", ".pptx", ".txt", ".json", ".xml", ".zip", ".rar"
}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

class IITScraper:
    def __init__(self, start_url, max_concurrent=20):
        self.start_url = start_url
        self.visited_urls = set()
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
        """Extracts category from the URL path. e.g. /inscription/xyz -> inscription"""
        parsed = urlparse(url)
        path_parts = [p for p in parsed.path.strip('/').split('/') if p]
        
        if not path_parts:
            return "general"
        
        category = path_parts[0].lower()
        # Clean category name to safely use as folder name
        category = re.sub(r'[^a-z0-9]', '_', category)
        return category if category else "general"

    def setup_category_dirs(self, category):
        """Creates category folders if they don't exist: {category}/file and {category}/text"""
        cat_dir = os.path.join(OUTPUT_DIR, category)
        file_dir = os.path.join(cat_dir, "file")
        text_dir = os.path.join(cat_dir, "text")
        
        os.makedirs(file_dir, exist_ok=True)
        os.makedirs(text_dir, exist_ok=True)
        
        return file_dir, text_dir

    def get_file_type(self, url):
        parsed_url = urlparse(url)
        ext = os.path.splitext(unquote(parsed_url.path))[1].lower()
        if ext in FILE_EXTENSIONS:
            return "doc"
        return None

    def clean_text(self, text):
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    async def download_file(self, url, category):
        file_dir, _ = self.setup_category_dirs(category)
        
        filename = unquote(url.split("/")[-1])
        if not filename or "?" in filename:
            filename = hashlib.md5(url.encode()).hexdigest() + ".pdf"
            
        file_path = os.path.join(file_dir, filename)
        
        if os.path.exists(file_path):
            return file_path

        try:
            async with self.semaphore:
                async with self.session.get(url, timeout=30) as response:
                    if response.status == 200:
                        async with aiofiles.open(file_path, 'wb') as f:
                            await f.write(await response.read())
                        logging.info(f"Downloaded File [{category}]: {filename}")
                        return file_path
        except Exception as e:
            logging.debug(f"Error downloading {url}: {e}")
            
        return None

    async def save_text(self, page_data, category):
        """Saves individual page content into a JSON file in the text folder."""
        if not page_data["content"]:
            return
            
        _, text_dir = self.setup_category_dirs(category)
        
        url_hash = hashlib.md5(page_data['url'].encode()).hexdigest()[:8]
        safe_title = re.sub(r'[^a-zA-Z0-9]', '_', page_data['title'])[:40]
        
        if not safe_title:
            safe_title = "page"
            
        filename = f"{safe_title}_{url_hash}.json"
        file_path = os.path.join(text_dir, filename)
        
        # Don't overwrite if it exists already
        if os.path.exists(file_path):
            return
            
        try:
            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(page_data, ensure_ascii=False, indent=4))
            logging.info(f"Saved Text [{category}]: {filename}")
        except Exception as e:
            logging.debug(f"Failed to save text {file_path}: {e}")

    def extract_page_data(self, html_content, url):
        soup = BeautifulSoup(html_content, 'html.parser')
        
        for element in soup(["script", "style", "nav", "footer", "header", "noscript", "svg"]):
            element.extract()
            
        title = soup.title.string if soup.title else "No Title"
        title = self.clean_text(title)
        
        text = soup.get_text(separator=' ')
        clean_ext_text = self.clean_text(text)
        
        return {
            "url": url,
            "title": title,
            "category": self.get_category_from_url(url),
            "content": clean_ext_text
        }
        
    async def process_page(self, url):
        category = self.get_category_from_url(url)
        
        try:
            async with self.semaphore:
                headers = {'User-Agent': 'Mozilla/5.0 (compatible; UniHelpBot/2.0; +https://iit.tn)'}
                async with self.session.get(url, headers=headers, timeout=20, allow_redirects=True) as response:
                    if response.status != 200:
                        logging.debug(f"Failed {url} HTTP {response.status}")
                        return

                    content_type = response.headers.get("Content-Type", "").lower()
                    file_type = self.get_file_type(url)
                    
                    # If it's a media file (image/video/audio) discovered by header, SKIP IT entirely.
                    if 'image/' in content_type or 'video/' in content_type or 'audio/' in content_type:
                        return
                    
                    # Direct check from Headers for documents
                    is_doc_header = any(x in content_type for x in ['application/pdf', 'application/msword', 'application/vnd', 'application/octet-stream', 'application/zip'])
                    
                    if file_type == "doc" or is_doc_header:
                        await self.download_file(url, category)
                        return

                    if "text/html" in content_type:
                        html_content = await response.text()
                        
                        page_data = self.extract_page_data(html_content, url)
                        await self.save_text(page_data, category)

                        soup = BeautifulSoup(html_content, 'html.parser')
                        
                        # Grab all potential URLs to crawl further
                        # Deliberately removed img, video, audio tags to avoid getting flooded with media URLs
                        tags_attributes = {
                            'a': 'href', 'link': 'href', 'iframe': 'src', 'script': 'src'
                        }
                        
                        for tag, attr in tags_attributes.items():
                            for element in soup.find_all(tag, **{attr: True}):
                                link = element.get(attr)
                                if link:
                                    full_url = urljoin(url, link).split('#')[0]
                                    
                                    # Very basic check against media extensions in URL to prevent wasteful requests
                                    skip_extensions = {".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".mp4", ".mp3"}
                                    ext = os.path.splitext(urlparse(full_url).path)[1].lower()
                                    if ext in skip_extensions:
                                        continue

                                    if self.is_valid_url(full_url) and full_url not in self.visited_urls:
                                        self.visited_urls.add(full_url)
                                        self.queue.put_nowait(full_url)
                                    
        except Exception as e:
            logging.debug(f"Error processing {url}: {e}")

    async def run(self):
        self.visited_urls.add(self.start_url)
        connector = aiohttp.TCPConnector(ssl=False)
        async with aiohttp.ClientSession(connector=connector) as session:
            self.session = session
            
            # Start 30 worker tasks to go fast
            workers = [asyncio.create_task(self.worker()) for _ in range(30)]
            
            await self.queue.join()
            
            for w in workers:
                w.cancel()
                
            logging.info("Crawling completed.")

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
        
    logging.info("Starting IIT Categorized Scraper FOR TEXT AND FILES Only...")
    scraper = IITScraper(start_url=BASE_URL, max_concurrent=20)
    
    try:
        asyncio.run(scraper.run())
    except KeyboardInterrupt:
        logging.info("Interrupted by user. Exiting...")
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        
    logging.info("Done! Check 'scraped_data_categorized' folder.")

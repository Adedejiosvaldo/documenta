# import asyncio
# import aiohttp
# from bs4 import BeautifulSoup
# from urllib.parse import urljoin, urlparse
# import os
# import time
# from concurrent.futures import ThreadPoolExecutor


# class FastDocScraper:
#     def __init__(self, base_url, max_concurrent=10):
#         self.base_url = base_url
#         self.visited_urls = set()
#         self.domain = urlparse(base_url).netloc
#         self.output_dir = "./src/scraped_docs"
#         self.max_concurrent = max_concurrent
#         self.semaphore = asyncio.Semaphore(max_concurrent)

#         if not os.path.exists(self.output_dir):
#             os.makedirs(self.output_dir)

#     def is_valid_url(self, url):
#         parsed = urlparse(url)
#         return (
#             parsed.netloc == self.domain or parsed.netloc == ""
#         ) and url not in self.visited_urls

#     def save_content(self, url, content):
#         filename = (
#             url.replace("https://", "").replace("http://", "").replace("/", "_")
#             + ".txt"
#         )
#         filepath = os.path.join(self.output_dir, filename)
#         with open(filepath, "w", encoding="utf-8") as f:
#             f.write(content)
#         print(f"Saved: {url}")

#     async def fetch_page(self, session, url):
#         async with self.semaphore:
#             try:
#                 headers = {
#                     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
#                 }
#                 async with session.get(
#                     url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)
#                 ) as response:
#                     return await response.text()
#             except Exception as e:
#                 print(f"Error fetching {url}: {e}")
#                 return None

#     def parse_content(self, html, url):
#         soup = BeautifulSoup(html, "html.parser")
#         content = soup.get_text(separator="\n", strip=True)

#         links = set()
#         for a_tag in soup.find_all("a", href=True):
#             absolute_url = urljoin(url, a_tag["href"])
#             if self.is_valid_url(absolute_url):
#                 links.add(absolute_url)

#         return content, links

#     async def scrape_page(self, session, url):
#         html = await self.fetch_page(session, url)
#         if html:
#             with ThreadPoolExecutor() as executor:
#                 loop = asyncio.get_event_loop()
#                 content, links = await loop.run_in_executor(
#                     executor, self.parse_content, html, url
#                 )
#                 self.save_content(url, content)
#                 return links
#         return set()

#     async def crawl(self, max_pages=100):
#         urls_to_visit = {self.base_url}
#         tasks = set()

#         async with aiohttp.ClientSession() as session:
#             while (urls_to_visit or tasks) and len(self.visited_urls) < max_pages:
#                 # Add new tasks for unvisited URLs
#                 while urls_to_visit and len(tasks) < self.max_concurrent:
#                     url = urls_to_visit.pop()
#                     if url not in self.visited_urls:
#                         print(f"Scraping: {url}")
#                         task = asyncio.create_task(self.scrape_page(session, url))
#                         tasks.add(task)
#                         self.visited_urls.add(url)

#                 # Wait for at least one task to complete
#                 done, tasks = await asyncio.wait(
#                     tasks, return_when=asyncio.FIRST_COMPLETED
#                 )

#                 # Process completed tasks
#                 for task in done:
#                     new_links = task.result()
#                     urls_to_visit.update(new_links - self.visited_urls)

#                 # Small delay to prevent overwhelming servers
#                 await asyncio.sleep(0.1)

#         print(f"Completed scraping. Total pages scraped: {len(self.visited_urls)}")


# def scrape_documentation(doc_url, max_concurrent=10, max_pages=100):
#     scraper = FastDocScraper(doc_url, max_concurrent)
#     asyncio.run(scraper.crawl(max_pages))


# # Example usage
# if __name__ == "__main__":
#     doc_url = "https://mongoosejs.com/docs/"  # Replace with your URL
#     start_time = time.time()
#     scrape_documentation(doc_url, max_concurrent=10, max_pages=100)
#     print(f"Time taken: {time.time() - start_time:.2f} seconds")


# scrapper.py
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os
import time
from concurrent.futures import ThreadPoolExecutor


import trafilatura  # Added for better content extraction


class FastDocScraper:
    def __init__(self, base_url, max_concurrent=10):
        self.base_url = base_url
        self.visited_urls = set()
        self.domain = urlparse(base_url).netloc
        self.output_dir = "./src/scraped_docs"
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def is_valid_url(self, url):
        parsed = urlparse(url)
        return (
            parsed.netloc == self.domain or parsed.netloc == ""
        ) and url not in self.visited_urls

    def save_content(self, url, content):
        filename = (
            url.replace("https://", "").replace("http://", "").replace("/", "_")
            + ".txt"
        )
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        # Enhanced logging to show content length
        print(f"Saved {len(content)} characters from {url} to {filepath}")

    async def fetch_page(self, session, url):
        async with self.semaphore:
            try:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5",
                }
                async with session.get(
                    url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    print(f"Status code for {url}: {response.status}")
                    if response.status != 200:
                        print(f"Non-200 status for {url}: {response.status}")
                        return None
                    html = await response.text()
                    print(f"Fetched {len(html)} characters from {url}")
                    # Save raw HTML for debugging
                    with open(
                        f"raw_{url.replace('/', '_')}.html", "w", encoding="utf-8"
                    ) as f:
                        f.write(html)
                    return html
            except Exception as e:
                print(f"Error fetching {url}: {e}")
                return None

    def parse_content(self, html, url):
        # Use trafilatura for robust content extraction
        content = trafilatura.extract(html)
        if not content:
            content = ""
            print(f"Warning: No content extracted from {url}")

        # Parse links using BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        links = set()
        for a_tag in soup.find_all("a", href=True):
            absolute_url = urljoin(url, a_tag["href"])
            if self.is_valid_url(absolute_url):
                links.add(absolute_url)

        return content, links

    async def scrape_page(self, session, url):
        html = await self.fetch_page(session, url)
        if html:
            with ThreadPoolExecutor() as executor:
                loop = asyncio.get_event_loop()
                content, links = await loop.run_in_executor(
                    executor, self.parse_content, html, url
                )
                self.save_content(url, content)
                return links
        return set()

    async def crawl(self, max_pages=None):
        urls_to_visit = {self.base_url}
        tasks = set()

        async with aiohttp.ClientSession() as session:
            while urls_to_visit or tasks:
                if max_pages and len(self.visited_urls) >= max_pages:
                    break
                while urls_to_visit and len(tasks) < self.max_concurrent:
                    url = urls_to_visit.pop()
                    if url not in self.visited_urls:
                        print(f"Scraping: {url}")
                        task = asyncio.create_task(self.scrape_page(session, url))
                        tasks.add(task)
                        self.visited_urls.add(url)

                done, tasks = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED
                )

                for task in done:
                    new_links = task.result()
                    urls_to_visit.update(new_links - self.visited_urls)

                await asyncio.sleep(0.1)

        print(f"Completed scraping. Total pages scraped: {len(self.visited_urls)}")


def scrape_documentation(doc_url, max_concurrent=10, max_pages=300):
    scraper = FastDocScraper(doc_url, max_concurrent)
    asyncio.run(scraper.crawl(max_pages))


if __name__ == "__main__":
    doc_url = "https://mongoosejs.com/docs/"  # Mongoose docs base URL
    start_time = time.time()
    scrape_documentation(doc_url, max_concurrent=10, max_pages=None)  # No limit
    print(f"Time taken: {time.time() - start_time:.2f} seconds")

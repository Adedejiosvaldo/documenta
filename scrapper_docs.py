import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import os


class DocScraper:
    def __init__(self, base_url):
        self.base_url = base_url
        self.visited_urls = set()
        self.domain = urlparse(base_url).netloc
        self.output_dir = "scraped_docs"

        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def is_valid_url(self, url):
        """Check if URL is valid and within the same domain"""
        parsed = urlparse(url)
        return (
            parsed.netloc == self.domain or parsed.netloc == ""
        ) and url not in self.visited_urls

    def save_content(self, url, content):
        """Save scraped content to a file"""
        # Create a safe filename from URL
        filename = (
            url.replace("https://", "").replace("http://", "").replace("/", "_")
            + ".txt"
        )
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Saved: {url}")

    def scrape_page(self, url):
        """Scrape a single page and return its content and links"""
        try:
            # Add headers to mimic browser request
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }

            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Extract text content
            content = soup.get_text(separator="\n", strip=True)

            # Find all links
            links = set()
            for a_tag in soup.find_all("a", href=True):
                absolute_url = urljoin(url, a_tag["href"])
                if self.is_valid_url(absolute_url):
                    links.add(absolute_url)

            return content, links

        except requests.RequestException as e:
            print(f"Error scraping {url}: {e}")
            return None, set()

    def crawl(self, max_pages=100):
        """Crawl the documentation site"""
        urls_to_visit = {self.base_url}

        while urls_to_visit and len(self.visited_urls) < max_pages:
            current_url = urls_to_visit.pop()

            if current_url in self.visited_urls:
                continue

            print(f"Scraping: {current_url}")
            content, links = self.scrape_page(current_url)

            if content:
                self.save_content(current_url, content)
                self.visited_urls.add(current_url)
                urls_to_visit.update(links)

            # Be nice to the server
            time.sleep(1)

        print(f"Completed scraping. Total pages scraped: {len(self.visited_urls)}")


def scrape_documentation(doc_url):
    """Main function to start the scraping process"""
    try:
        scraper = DocScraper(doc_url)
        scraper.crawl()
    except Exception as e:
        print(f"An error occurred: {e}")


# Example usage
if __name__ == "__main__":
    # Replace with your documentation URL
    doc_url = "https://docs.python.org/3/"  # Example URL
    scrape_documentation(doc_url)

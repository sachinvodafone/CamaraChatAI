import os
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms import OpenAI
from dotenv import load_dotenv
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load API keys from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY missing in .env")

# Configure global settings
from app.settings import init_settings
init_settings()

BASE_URL = "https://camaraproject.org/"
BASE_DOMAIN = "camaraproject.org"
CRAWL_LIMIT = 100

visited_urls = set()

def is_valid_url(url):
    parsed = urlparse(url)
    return (
        parsed.netloc == BASE_DOMAIN and
        not any(x in parsed.path for x in ["/wp-admin", "/login", "/feed", "/wp-json", ".pdf"])
    )

def clean_text(text):
    text = re.sub(r'\n\s*\n', '\n', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'(Home|About|Contact|Privacy Policy|Terms of Use|Sitemap)', '', text, flags=re.IGNORECASE)
    return text

def extract_links_and_text(url):
    try:
        logger.info(f"Fetching: {url}")
        response = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if response.status_code != 200:
            logger.error(f"Failed ({response.status_code}) for {url}")
            return "", set()

        soup = BeautifulSoup(response.content, "html.parser")
        main = (
            soup.find("main") or
            soup.find("div", {"class": re.compile("content|entry-content|main-content")}) or
            soup.find("article") or
            soup.body
        )
        page_text = main.get_text(separator="\n", strip=True) if main else soup.get_text(separator="\n", strip=True)
        page_text = clean_text(page_text)

        logger.info(f"Content sample from {url}:\n{page_text[:500]}...")
        if not page_text or len(page_text) < 50:
            logger.warning(f"No meaningful text extracted from {url}")
        else:
            logger.info(f"Collected {len(page_text)} characters from {url}")

        links = set()
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            full_url = urljoin(url, href)
            if is_valid_url(full_url):
                links.add(full_url)

        return page_text, links
    except Exception as e:
        logger.error(f"Failed to process {url}: {e}")
        return "", set()

def crawl_site(start_url):
    to_visit = [start_url]
    documents = []

    while to_visit and len(visited_urls) < CRAWL_LIMIT:
        current_url = to_visit.pop(0)
        if current_url in visited_urls:
            continue
        visited_urls.add(current_url)

        text, links = extract_links_and_text(current_url)
        if text:
            documents.append(Document(text=text, metadata={"source": current_url}))
        else:
            logger.warning(f"No content retrieved from {current_url}")
        to_visit.extend(links - visited_urls)
        logger.info(f"Found {len(links)} new links, {len(to_visit)} URLs left to visit")

    logger.info(f"Crawling complete. Collected {len(documents)} documents.")
    return documents

def generate_index():
    if os.path.exists("./storage"):
        logger.info("Loading existing index from ./storage")
        storage_context = StorageContext.from_defaults(persist_dir="./storage")
        return load_index_from_storage(storage_context)
    documents = crawl_site(BASE_URL)
    if not documents:
        logger.error("No documents collected. Index will be empty.")
        return None
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir="./storage")
    logger.info(f"✅ Index created with {len(documents)} documents.")
    return index

def query_index(query, index):
    try:
        query_engine = index.as_query_engine()
        response = query_engine.query(query)
        logger.info(f"\nQuery: {query}")
        logger.info(f"Response: {response.response}")
        for node in response.source_nodes:
            logger.info(f"Source: {node.metadata.get('source', 'Unknown')}")
            logger.info(f"Score: {node.score:.2f}")
            logger.info(f"Snippet: {node.node.get_content()[:300]}...")
        return response.response
    except Exception as e:
        logger.error(f"Query failed: {e}")
        return "Error: No data available"

if __name__ == "__main__":
    index = generate_index()
    queries = [
        "What is the CAMARA Project?",
        "What are the new APIs proposed in 2025?",
        "What is the Spring25 meta-release?"
    ]
    for query in queries:
        query_index(query, index)

import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urlunparse
import logging
import regex as re
import html2text
from typing import Set, List

# Initialize the logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize html2text converter
converter = html2text.HTML2Text()
converter.ignore_links = True
converter.ignore_images = True
converter.ignore_emphasis = True

# Define constants
KEYWORDS = ['Overview', 'Description', 'Model', 'Modelling', 'Experiments',
            'Engineering', 'Results', 'Design', 'Implementation']


def is_valid_url(base: str, url: str) -> bool:
    """
    Check if the URL belongs to the same domain as the base URL.

    Args:
        base (str): The base URL.
        url (str): The URL to validate.

    Returns:
        bool: True if the URL is valid and belongs to the same domain, False otherwise.
    """
    return urlparse(url).netloc == urlparse(base).netloc


def canonicalize_url(url: str) -> str:
    """
    Normalize and canonicalize a URL.

    Args:
        url (str): The URL to canonicalize.

    Returns:
        str: The canonicalized URL.
    """
    parsed = urlparse(url)
    scheme = 'https' if parsed.scheme == 'http' else parsed.scheme  # Force HTTPS
    netloc = parsed.netloc.replace('www.', '')  # Remove 'www'
    normalized_path = parsed.path.rstrip('/') if parsed.path != '/' else '/'
    return urlunparse((scheme, netloc, normalized_path, '', '', ''))


def extract_text_from_html(html_content: str) -> str:
    """
    Extract text from HTML content using html2text.

    Args:
        html_content (str): The HTML content to extract text from.

    Returns:
        str: The extracted text.
    """
    try:
        text = converter.handle(html_content)
        return text
    except Exception as e:
        logger.error(f"Error processing HTML content: {str(e)}")
        return ""


def crawl_and_extract_text(url: str) -> str:
    """
    Crawl the given URL, extract text, and handle links recursively.

    Args:
        url (str): The starting URL to crawl.

    Returns:
        str: The concatenated text extracted from the crawled pages.
    """
    all_text: List[str] = []

    visited: Set[str] = set()
    seen_sentences: Set[str] = set()

    def crawl(url: str) -> None:
        url = canonicalize_url(url)
        initial_base_path = urlparse(url).path.rstrip('/')
        if url in visited:
            logger.info(f"Already visited: {url}")
            return
        logger.info(f"Visiting {url}")
        visited.add(url)
        try:
            response = requests.get(url, timeout=40)
            response.raise_for_status()  # Check for HTTP errors
            html_content = response.text

            # Extract text using html2text
            text = extract_text_from_html(html_content)

            # Split the extracted text into sentences
            sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and sentence not in seen_sentences:
                    seen_sentences.add(sentence)
                    all_text.append(sentence)

            # Find all links and recurse
            soup = BeautifulSoup(html_content, 'html.parser')
            for link_tag in soup.find_all('a', href=True):
                link = urljoin(url, link_tag['href'])
                link = canonicalize_url(link)
                if (any(keyword in link for keyword in KEYWORDS)
                        and is_valid_url(url, link)
                        and urlparse(link).path.startswith(initial_base_path)):
                    if link not in visited:
                        crawl(link)
        except requests.RequestException as e:
            logger.error(f"Failed to retrieve {url}: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to process {url}: {str(e)}")

    crawl(url)
    return ' '.join(all_text)


def clean_text(text: str) -> str:
    """
    Clean the text by removing unwanted characters, normalizing spaces, and removing citations.

    Args:
        text (str): The text to clean.

    Returns:
        str: The cleaned text.
    """
    # Normalize unicode characters to ASCII
    text = text.encode('ascii', 'ignore').decode('ascii')

    # Replace URLs with a placeholder
    text = re.sub(r'http[s]?://\S+', '', text)

    # Replace email addresses with a placeholder
    text = re.sub(r'\S+@\S+', '', text)

    # Remove citations like [1], [1-3], [1,2], or [1-3,5]
    text = re.sub(r'\[\d+(-\d+)?(,\s*\d+(-\d+)?)*\]', '', text)

    # Handling common scientific notations and units, e.g., 1e-4, 2.3mg
    text = re.sub(r'\b\d+\.\d+([eE][-+]?\d+)?\s*mg\b', '', text)
    text = re.sub(r'\b\d+\.\d+([eE][-+]?\d+)?\b', '', text)

    # Remove references to journals and papers often found in academic writing
    text = re.sub(r'\[[Jj]\]\.', '', text)  # Remove [J].
    text = re.sub(r'\b([A-Z][a-z]*\b\.?\s*){1,4}(\d{4}).*?\d+', '', text)  # Remove author lists and year followed by anything ending in numbers.

    # Normalize spacing around punctuation
    text = re.sub(r'\s*([,.!?])\s*', r'\1 ', text)

    # Handle initials and abbreviations followed by periods
    text = re.sub(r'\b([A-Z]\.)\s*', r'\1', text)  # Collapse spaces after initials like J. D. -> J.D.

    # Normalize space usage, collapsing multiple spaces into one
    text = ' '.join(text.split())

    return text


def load_and_process_data(input_csv: str = 'src/data/database.csv', output_excel: str = 'src/data/processed_data.xlsx') -> None:
    """
    Load data from a CSV file, process it, and save it to an Excel file.

    Args:
        input_csv (str): The path to the input CSV file.
        output_excel (str): The path to the output Excel file.
    """
    try:
        data = pd.read_csv(input_csv)
        logger.info(f"Loaded data from {input_csv}")

        data = data[data['Year'] == 2019]
        logger.info("Filtered data for the year 2019")

        data['Abstract'] = data['Abstract'].apply(clean_text)
        logger.info("Cleaned 'Abstract' column")

        # Apply crawling and text extraction to the 'Wiki' column
        data['Wiki_content'] = data['Wiki'].apply(lambda url: clean_text(crawl_and_extract_text(url)))
        logger.info("Processed 'Wiki' column")

        data.to_excel(output_excel, index=False)
        logger.info(f"Saved processed data to {output_excel}")
    except Exception as e:
        logger.error(f"An error occurred during data processing: {str(e)}")


if __name__ == "__main__":
    load_and_process_data()

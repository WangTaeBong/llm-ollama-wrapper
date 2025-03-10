import asyncio
import logging
import re
import time
from typing import List, Dict, Any

from duckduckgo_search import DDGS
from langchain_core.documents import Document
from tenacity import retry, stop_after_attempt, wait_exponential

# Module level logger configuration
logger = logging.getLogger(__name__)


class SearchEngine:
    """
    Class responsible for web search functionality.

    Provides features for performing web searches through search engine (DuckDuckGo)
    and processing the results. Includes synchronous and asynchronous search methods
    and URL handling functionality.
    """

    def __init__(self, settings):
        """
        Constructor for the SearchEngine class.

        Args:
            settings: Settings object containing search configurations
        """
        self.settings = settings

        # Pre-compile URL pattern for performance optimization
        self._url_pattern = re.compile(
            r"https?://(?:www\.)?[-a-zA-Z0-9@:%._+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b[-ㄱ-ㅎ가-힣a-zA-Z0-9@:%_+.~#?&/=]*")

        logger.debug("SearchEngine instance has been initialized")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=4))
    def websearch_duckduckgo(self, query: str, sleep_duration: float = 1.0) -> List[Document]:
        """
        Performs DuckDuckGo search and returns relevant documents.

        Retries up to 3 times in case of failure.

        Args:
            query (str): Query string to search for
            sleep_duration (float): Wait time in seconds after search

        Returns:
            List[Document]: List of Document objects containing search results
        """
        if not query or not query.strip():
            logger.warning("Empty search query provided")
            return []

        start_time = time.time()

        try:
            with DDGS() as ddgs:
                # Get search settings
                region = getattr(self.settings.web_search, 'region', 'wt-wt')
                max_results = getattr(self.settings.web_search, 'max_results', 10)

                # Execute search
                results = list(ddgs.text(query, region=region, max_results=max_results))

                # Convert results to Document objects
                documents = self._convert_results_to_documents(results)

            # Add interval between requests to prevent server overload
            if sleep_duration > 0:
                time.sleep(sleep_duration)

            # Performance logging
            elapsed_time = time.time() - start_time
            if documents:
                logger.info(
                    f"Search '{query[:30]}...' returned {len(documents)} results, "
                    f"processing time: {elapsed_time:.2f} seconds")
            else:
                logger.warning(
                    f"No results found for search '{query[:30]}...' (processing time: {elapsed_time:.2f} seconds)")

            return documents

        except Exception as e:
            logger.error(f"Error occurred during DuckDuckGo search (query='{query[:30]}...'): {str(e)}")
            return []  # Return empty list to allow continued execution even in case of failure

    async def websearch_duckduckgo_async(self, query: str, sleep_duration: float = 1.0) -> List[Document]:
        """
        Performs DuckDuckGo search asynchronously.

        Args:
            query (str): Query string to search for
            sleep_duration (float): Wait time in seconds after search

        Returns:
            List[Document]: List of Document objects containing search results
        """
        if not query or not query.strip():
            return []

        start_time = time.time()

        try:
            # DuckDuckGo search is actually synchronous, but this allows it to be used in async contexts
            with DDGS() as ddgs:
                # Get search settings
                region = getattr(self.settings.web_search, 'region', 'wt-wt')
                max_results = getattr(self.settings.web_search, 'max_results', 10)

                # Execute search
                results = list(ddgs.text(query, region=region, max_results=max_results))

                # Convert results to Document objects
                documents = self._convert_results_to_documents(results)

            # Perform asynchronous wait
            if sleep_duration > 0:
                await asyncio.sleep(sleep_duration)

            # Performance logging
            elapsed_time = time.time() - start_time
            if documents:
                logger.info(
                    f"Async search '{query[:30]}...' returned {len(documents)} results, "
                    f"processing time: {elapsed_time:.2f} seconds")
            else:
                logger.warning(
                    f"No results found for async search '{query[:30]}...' "
                    f"(processing time: {elapsed_time:.2f} seconds)")

            return documents

        except Exception as e:
            logger.error(f"Error occurred during async DuckDuckGo search (query='{query[:30]}...'): {str(e)}")
            return []

    def replace_urls_with_links(self, query_answer: str) -> str:
        """
        Replaces URLs in text with hyperlinks.

        Args:
            query_answer (str): Original text that may contain URLs

        Returns:
            str: Text with URLs replaced by hyperlinks
        """
        if not query_answer:
            return ""

        try:
            # Find URLs using the compiled regex
            matches = self._url_pattern.findall(query_answer)

            # Replace each discovered URL with a hyperlink
            for url in matches:
                query_answer = query_answer.replace(url, f'<a href="{url}" target="_blank">{url}</a>')

            return query_answer

        except Exception as e:
            logger.error(f"Error converting URLs to hyperlinks: {str(e)}")
            return query_answer  # Return original text in case of error

    @classmethod
    def _convert_results_to_documents(cls, results: List[Dict[str, Any]]) -> List[Document]:
        """
        Converts search results to a list of Document objects.

        Args:
            results (List[Dict[str, Any]]): List of search result dictionaries

        Returns:
            List[Document]: List of converted Document objects
        """
        documents = []
        invalid_count = 0

        for result in results:
            # Check if result is valid
            if not result or not isinstance(result, dict):
                invalid_count += 1
                continue

            # Check if required fields exist
            body = result.get('body', '')
            if not body:
                logger.debug("Skipping search result with no content")
                continue

            # Create Document object
            doc = Document(
                page_content=body,
                metadata={
                    'source': result.get('title', 'Unknown'),
                    'doc_page': result.get('href', '#'),
                }
            )
            documents.append(doc)

        # Debug log for invalid results
        if invalid_count > 0:
            logger.debug(f"Skipped {invalid_count} invalid search results")

        return documents

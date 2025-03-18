import asyncio
import hashlib
import json
import logging
import time
from collections import defaultdict
from typing import List, Optional, Callable, Set, Dict, Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel, Field, field_validator

from src.common.config_loader import ConfigLoader
from src.common.query_check_dict import QueryCheckDict
from src.common.restclient import rc
from src.schema.retriever_req import RetrieverRequest
from src.services.response_generator import ResponseGenerator
from src.services.search_engine import SearchEngine

# Logging configuration
logging.getLogger("aiohttp").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)

# Load settings
config_loader = ConfigLoader()
settings = config_loader.get_settings()

# Logger setup
logger = logging.getLogger(__name__)


class CacheManager:
    """
    Cache manager for API responses and search results

    Integrates various caching strategies for optimal performance.
    """

    def __init__(self, ttl: int = 3600):
        """
        Initialize the cache manager

        Args:
            ttl: Cache Time-To-Live in seconds
        """
        self.cache = {}
        self.cache_timestamps = {}
        self.ttl = ttl

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve a value from cache

        Args:
            key: Cache key

        Returns:
            Cached value or None (on cache miss or expiration)
        """
        current_time = time.time()

        if key in self.cache:
            # Check if cache entry is still valid
            if current_time - self.cache_timestamps.get(key, 0) < self.ttl:
                logger.debug(f"Cache hit: {key}")
                return self.cache[key]
            else:
                # Remove expired cache entry
                self._remove(key)

        return None

    def set(self, key: str, value: Any) -> None:
        """
        Store an item in the cache

        Args:
            key: Cache key
            value: Value to store
        """
        self.cache[key] = value
        self.cache_timestamps[key] = time.time()

    def _remove(self, key: str) -> None:
        """
        Remove an item from the cache

        Args:
            key: Key to remove
        """
        if key in self.cache:
            del self.cache[key]
        if key in self.cache_timestamps:
            del self.cache_timestamps[key]

    def clear(self) -> None:
        """Clear all cache entries"""
        self.cache.clear()
        self.cache_timestamps.clear()

    @staticmethod
    def create_key(data: Any) -> str:
        """
        Utility method to create cache keys

        Args:
            data: Data to generate a key for

        Returns:
            Generated hash key
        """
        # Generate hash based on data type
        if isinstance(data, str):
            return hashlib.md5(data.encode('utf-8')).hexdigest()
        elif isinstance(data, dict):
            # Convert dictionary to sorted JSON string for consistent hashing
            return hashlib.md5(json.dumps(data, sort_keys=True).encode('utf-8')).hexdigest()
        else:
            # Convert other data types to string
            return hashlib.md5(str(data).encode('utf-8')).hexdigest()


class DocumentStore:
    """
    Document storage class

    Manages storage and retrieval of Document objects with deduplication.
    """

    def __init__(self):
        """Initialize document store"""
        self._documents: Dict[str, List[Document]] = defaultdict(list)
        self._document_keys: Set[str] = set()

    def add(self, document: Document) -> bool:
        """
        Add a single document

        Args:
            document: Document object to add

        Returns:
            bool: True if document was added, False if duplicate
        """
        doc_key = self._create_document_key(document)

        # Check for duplicates
        if doc_key in self._document_keys:
            return False

        # Add document
        doc_name = document.metadata.get("doc_name", "default")
        self._documents[doc_name].append(document)
        self._document_keys.add(doc_key)
        return True

    def add_batch(self, documents: List[Document]) -> int:
        """
        Add a batch of documents

        Args:
            documents: List of Document objects to add

        Returns:
            int: Count of successfully added documents
        """
        added_count = 0
        for doc in documents:
            if self.add(doc):
                added_count += 1
        return added_count

    def get_all(self) -> List[Document]:
        """
        Get all stored documents

        Returns:
            List[Document]: All stored Document objects
        """
        return [doc for docs in self._documents.values() for doc in docs]

    def get_by_doc_name(self, doc_name: str) -> List[Document]:
        """
        Get documents by name

        Args:
            doc_name: Document name to filter by

        Returns:
            List[Document]: Document objects with the specified name
        """
        return self._documents.get(doc_name, [])

    def clear(self) -> None:
        """Clear the document store"""
        self._documents.clear()
        self._document_keys.clear()

    @classmethod
    def _create_document_key(cls, document: Document) -> str:
        """
        Create a unique key for a document

        Args:
            document: Document object

        Returns:
            str: Unique document key
        """
        return f"{document.metadata.get('doc_name', 'default')}_{document.page_content}"


class APIClient:
    """
    API communication client class

    Handles API requests with caching, timeout and error handling.
    """

    def __init__(self, url: str, headers: Dict[str, str], cache_manager: CacheManager):
        """
        Initialize API client

        Args:
            url: API endpoint URL
            headers: HTTP headers for API requests
            cache_manager: Cache manager instance
        """
        self.url = url
        self.headers = headers
        self.cache_manager = cache_manager
        self.timeout = 60.0  # API call timeout in seconds

    async def fetch_documents(self, request_data: RetrieverRequest, query: str) -> Dict[str, Any]:
        """
        Fetch documents by calling the API

        Args:
            request_data: API request data
            query: Search query

        Returns:
            Dict[str, Any]: API response data
        """
        # Copy request data and update query
        request_data_copy = request_data.copy()
        request_data_copy.chat.user = query

        # Extract session_id for logging
        session_id = request_data_copy.meta.session_id

        # Create cache key
        cache_key = self.cache_manager.create_key({
            'url': self.url,
            'request': request_data_copy.dict()
        })

        # Check cache
        cached_response = self.cache_manager.get(cache_key)
        if cached_response:
            logger.debug(f"[{session_id}] Using API cache for query: {query}")
            return cached_response

        start_time = time.time()
        logger.debug(f"[{session_id}] API call started for query: {query}")

        try:
            # API call with timeout
            response = await asyncio.wait_for(
                rc.restapi_post_async(self.url, request_data_copy.dict()),
                timeout=self.timeout
            )

            # Check response status
            if response.get("status", 200) != 200:
                logger.error(f"[{session_id}] API call failed with status code {response.get('status')}")
                raise Exception(f"API call failed: {response.get('status')}")

            # Store response in cache
            self.cache_manager.set(cache_key, response)

            api_time = time.time() - start_time
            logger.debug(f"[{session_id}] API call completed: {api_time:.4f}s elapsed")

            return response

        except asyncio.TimeoutError:
            logger.warning(f"[{session_id}] API call timeout (> {self.timeout}s) for query: {query}")
            return {"status": 408, "message": "Request Timeout", "chat": {"payload": []}}
        except Exception as e:
            logger.error(f"[{session_id}] Error during API call: {e}")
            return {"status": 500, "message": str(e), "chat": {"payload": []}}


class WebSearchProvider:
    """
    Web search provider class

    Handles web search operations with caching and optimized performance.
    """

    def __init__(self, api_settings, cache_manager: CacheManager):
        """
        Initialize web search provider

        Args:
            api_settings: Application settings
            cache_manager: Cache manager instance
        """
        self.settings = api_settings
        self.cache_manager = cache_manager
        self.timeout = 5.0  # Web search timeout in seconds

        query_check_dict = QueryCheckDict(self.settings.prompt.llm_prompt_path)
        self.response_generator = ResponseGenerator(self.settings, query_check_dict)
        self.search_engine = SearchEngine(self.settings)

    async def search_async(self, query: str, rag_sys_info: str = None, session_id: str = None) -> List[Document]:
        """
        Perform web search asynchronously

        Args:
            query: Search query
            rag_sys_info: Chatbot system information
            session_id: Session identifier for logging

        Returns:
            List[Document]: Search result documents
        """
        # Return empty list if web search is disabled or for FAQ chatbots
        if (not self.settings.web_search.use_flag or
                (rag_sys_info and self.response_generator.is_faq_type_chatbot(rag_sys_info)) or
                self.response_generator.is_voc_type_chatbot(rag_sys_info)):
            return []

        # Create cache key
        cache_key = self.cache_manager.create_key(f"websearch_{query}_{rag_sys_info}")

        # Check cache
        cached_results = self.cache_manager.get(cache_key)
        if cached_results is not None:
            logger.debug(f"[{session_id}] Web search cache hit for query: {query}")
            return cached_results

        # Perform web search asynchronously
        try:
            start_time = time.time()
            logger.debug(f"[{session_id}] Web search started for query: {query}")

            # Apply timeout
            web_results = await asyncio.wait_for(
                self._perform_web_search(query),
                timeout=self.timeout
            )

            search_time = time.time() - start_time
            logger.debug(f"[{session_id}] Web search completed: {search_time:.4f}s elapsed, {len(web_results)} results")

            # Cache results
            self.cache_manager.set(cache_key, web_results)

            return web_results

        except asyncio.TimeoutError:
            logger.warning(f"[{session_id}] Web search timeout (> {self.timeout}s) for query: {query}")
            return []
        except Exception as e:
            logger.error(f"[{session_id}] Error during web search: {e}")
            return []

    async def _perform_web_search(self, query: str) -> List[Document]:
        """
        Internal method to perform the actual web search

        Args:
            query: Search query

        Returns:
            List[Document]: Search results
        """
        # Determine if DuckDuckGo search is async
        is_async = asyncio.iscoroutinefunction(self.search_engine.websearch_duckduckgo_async)

        if is_async:
            try:
                # Temporarily disable debug logs from DuckDuckGo
                original_level = logging.getLogger().level
                logging.getLogger().setLevel(logging.WARNING)

                try:
                    results = await self.search_engine.websearch_duckduckgo_async(query)
                    return results
                finally:
                    # Restore original logging level
                    logging.getLogger().setLevel(original_level)
            except Exception as e:
                logger.error(f"Error during async web search: {e}")
                return []
        else:
            # Run synchronous function in a separate thread
            return await asyncio.to_thread(self.search_engine.websearch_duckduckgo, query)


class CustomRetriever(BaseRetriever, BaseModel):
    """
    Custom retriever that fetches documents from external API and web search

    Inherits from LangChain's BaseRetriever and Pydantic's BaseModel to support
    both synchronous and asynchronous document retrieval with caching and parallelism.
    """
    request_data: RetrieverRequest = Field(..., description="Request data structure for API calls")
    url: str = Field(default=settings.api.retrival_api, description="External API endpoint URL")
    headers: Dict[str, str] = Field(
        default={"content-type": "application/json;charset=utf-8"},
        description="HTTP headers for API requests"
    )
    page_content_key: str = Field(default="content", description="Key to extract content from API response")
    metadata_key: List[str] = Field(
        default=["doc_name", "doc_page"],
        description="Keys to extract metadata from API response"
    )
    query_callback: Optional[Callable[[str], None]] = Field(
        default=None, description="Optional callback for query rewriting"
    )

    # Performance settings
    max_concurrent_requests: int = Field(default=5, description="Maximum number of concurrent requests")
    cache_ttl: int = Field(default=3600, description="Cache time-to-live in seconds")

    # Pydantic private fields (not initialized during class initialization)
    _document_store: DocumentStore = None
    _api_client: APIClient = None
    _web_search: WebSearchProvider = None
    _cache_manager: CacheManager = None
    _semaphore: asyncio.Semaphore = None

    # Pydantic field validator (V2 style)
    @field_validator('request_data')
    def validate_request_data(cls, v):
        """Validate request data"""
        if v is None:
            raise ValueError("request_data cannot be None")
        return v

    def __init__(self, **data):
        """
        Initialize CustomRetriever
        """
        super().__init__(**data)

        # Initialize shared cache manager
        self._cache_manager = CacheManager(ttl=self.cache_ttl)

        # Initialize dependencies
        self._document_store = DocumentStore()
        self._api_client = APIClient(self.url, self.headers, self._cache_manager)
        self._web_search = WebSearchProvider(settings, self._cache_manager)

        # Semaphore for concurrency control
        self._semaphore = asyncio.Semaphore(self.max_concurrent_requests)

    def add_documents(self, documents: List[Document], batch_size: int = 50) -> None:
        """
        Add documents to the retriever's local store

        Args:
            documents: List of Document objects to store
            batch_size: Number of documents to process in each batch for better performance
        """
        if not documents:
            logger.warning("No documents provided to add_documents")
            return

        try:
            # Process documents in batches
            added_count = 0
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                added_count += self._document_store.add_batch(batch)

            if added_count > 0:
                logger.info(f"Successfully added {added_count} new documents")
        except Exception as e:
            logger.exception(f"Error adding documents: {e}")

    async def add_documents_async(self, documents: List[Document], batch_size: int = 50) -> None:
        """
        Add documents to the retriever's local store asynchronously

        Args:
            documents: List of Document objects to store
            batch_size: Number of documents to process in each batch
        """
        if not documents:
            logger.warning("No documents provided to add_documents_async")
            return

        try:
            # Process documents in batches (asynchronously)
            added_count = 0
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                # Yield control to allow other async operations to progress
                await asyncio.sleep(0)
                added_count += self._document_store.add_batch(batch)

            if added_count > 0:
                logger.info(f"Successfully added {added_count} new documents asynchronously")
        except Exception as e:
            logger.exception(f"Error adding documents asynchronously: {e}")

    def get_all_documents(self) -> List[Document]:
        """
        Retrieve all stored documents

        Returns:
            List of all stored Document objects
        """
        all_docs = self._document_store.get_all()
        if all_docs:
            logger.debug(f"Retrieved {len(all_docs)} documents in total")
        return all_docs

    def _get_relevant_documents(
            self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """
        Retrieve documents synchronously

        Args:
            query: Query string for document retrieval
            run_manager: Optional callback manager

        Returns:
            List of retrieved Document objects
        """
        session_id = self.request_data.meta.session_id
        logger.debug(f"[{session_id}] Retrieving documents for query: {query}")

        try:
            # Use asyncio.run() for safe async execution
            return asyncio.run(self._fetch_and_process_documents(query))
        except RuntimeError as e:
            logger.error(f"[{session_id}] Event loop error: {e}")
            # Fallback method for when event loop is already running
            try:
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(self._fetch_and_process_documents(query))
            except Exception as inner_e:
                logger.exception(f"[{session_id}] Error in fallback event loop: {inner_e}")
                return []
        except Exception as e:
            logger.exception(f"[{session_id}] Error in synchronous document retrieval: {e}")
            return []

    async def _ainvoke(
            self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None, **kwargs
    ) -> List[Document]:
        """
        Retrieve documents asynchronously (LangChain 0.1.46+ compatible method)

        Args:
            query: Query string for document retrieval
            run_manager: Optional callback manager
            **kwargs: Additional parameters

        Returns:
            List of retrieved Document objects
        """
        return await self._fetch_and_process_documents(query)

    async def _aget_relevant_documents(
            self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """
        Retrieve documents asynchronously (deprecated, use _ainvoke instead)

        Args:
            query: Query string for document retrieval
            run_manager: Optional callback manager

        Returns:
            List of retrieved Document objects
        """
        return await self._ainvoke(query, run_manager=run_manager)

    async def _fetch_and_process_documents(self, query: str) -> List[Document]:
        """
        Integrated method to fetch and process documents from API and web search

        Args:
            query: Query string for document retrieval

        Returns:
            List[Document]: Retrieved document objects
        """
        session_id = self.request_data.meta.session_id

        try:
            start_time = time.time()
            logger.debug(f"[{session_id}] Document retrieval process started for query: {query}")

            # Acquire semaphore (limit concurrent requests)
            async with self._semaphore:
                # Process query callback if provided
                if self.query_callback:
                    self.query_callback(query)

                # 웹 검색 설정 확인
                web_search_enabled = getattr(settings.web_search, 'use_flag', False)
                document_add_type = getattr(settings.web_search, 'document_add_type', 1)

                # 작업 목록 초기화
                tasks = []
                task_types = []

                # web_search.use_flag가 true이고 document_add_type이 0이면 API 문서 검색을 건너뜀
                if not (web_search_enabled and document_add_type == 0):
                    # API 문서 검색 작업 추가
                    tasks.append(self._api_client.fetch_documents(self.request_data, query))
                    task_types.append("api")
                    logger.debug(f"[{session_id}] API 문서 검색 작업 추가됨")
                else:
                    logger.debug(
                        f"[{session_id}] 웹 검색 설정에 따라 API 문서 검색을 건너뜁니다 (use_flag={web_search_enabled}, document_add_type={document_add_type})")

                # 웹 검색이 활성화된 경우 웹 검색 작업 추가
                if web_search_enabled:
                    tasks.append(self._web_search.search_async(query, self.request_data.meta.rag_sys_info, session_id))
                    task_types.append("web")
                    logger.debug(f"[{session_id}] 웹 검색 작업 추가됨")

                # 모든 작업 동시 실행
                results = await asyncio.gather(*tasks)

                # 결과 처리
                rag_documents = []

                # 작업 유형에 따라 결과 처리
                for i, task_type in enumerate(task_types):
                    if task_type == "api":
                        api_response = results[i]
                        rag_documents = self._extract_documents_from_response(api_response, session_id)
                    elif task_type == "web" and results[i]:
                        web_results = results[i]
                        # 웹 검색 결과 통합 방식 결정
                        if web_search_enabled:
                            if document_add_type == 1 and rag_documents:
                                # 기존 문서에 웹 검색 결과 추가
                                rag_documents.extend(web_results)
                            elif document_add_type == 0:
                                # 웹 검색 결과만 반환
                                rag_documents = web_results

                total_time = time.time() - start_time
                logger.debug(
                    f"[{session_id}] Document retrieval completed: {total_time:.4f}s elapsed, "
                    f"{len(rag_documents)} documents found")

                return rag_documents

        except Exception as e:
            logger.exception(f"[{session_id}] Error fetching and processing documents: {e}")
            return []

    def _extract_documents_from_response(self, response: Dict[str, Any], session_id: str = None) -> List[Document]:
        """
        Extract documents from API response and convert to Document objects

        Args:
            response: API response data
            session_id: Session identifier for logging

        Returns:
            List[Document]: Converted Document objects
        """
        documents = []

        try:
            start_time = time.time()

            # Extract payload
            payload = response.get("chat", {}).get("payload", [])

            for item in payload:
                # Check if necessary data exists
                content = item.get(self.page_content_key)
                if not content:
                    continue

                # Extract metadata
                metadata = {
                    key: item.get(key, "")
                    for key in self.metadata_key
                    if key in item
                }

                # Create and add Document object
                documents.append(Document(
                    page_content=content,
                    metadata=metadata
                ))

            process_time = time.time() - start_time
            logger.debug(f"[{session_id}] Response processing completed: {process_time:.4f}s elapsed")

            return documents

        except KeyError as e:
            logger.error(f"[{session_id}] Key error during response processing: {e}")
            return []
        except Exception as e:
            logger.exception(f"[{session_id}] Unexpected error during response processing: {e}")
            return []

    def clear_cache(self) -> None:
        """Clear the cache"""
        self._cache_manager.clear()

    # Property accessor for backward compatibility
    @property
    def stored_documents(self) -> Dict[str, List[Document]]:
        """Return dictionary of stored documents (for backward compatibility)"""
        result = defaultdict(list)
        for doc in self._document_store.get_all():
            doc_name = doc.metadata.get("doc_name", "default")
            result[doc_name].append(doc)
        return result

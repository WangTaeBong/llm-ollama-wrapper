import logging
from datetime import datetime
from functools import lru_cache
from time import localtime
from typing import Dict, List, Tuple

from langchain_core.documents import Document

# Module level logger configuration
logger = logging.getLogger(__name__)


# Helper function to add session ID to log messages
def log_with_session_id(logger_func, message, request=None):
    """
    Adds session ID to log messages if available in the request.

    Args:
        logger_func: The logger function to use (debug, info, warning, error)
        message (str): The log message
        request: The request object that might contain session_id
    """
    session_id = None
    if request and hasattr(request, 'meta') and hasattr(request.meta, 'session_id'):
        session_id = request.meta.session_id

    if session_id:
        logger_func(f"[{session_id}] {message}")
    else:
        logger_func(message)


class ResponseGenerator:
    """
    Class responsible for response generation functionalities.

    This class handles the response generation logic for the chat system, providing
    features such as prompt retrieval, response formatting, and reference information addition.
    """

    def __init__(self, settings, llm_prompt_json_dict):
        """
        Constructor for the ResponseGenerator class.

        Args:
            settings: System settings object
            llm_prompt_json_dict: Dictionary object containing prompt information
        """
        self.settings = settings
        self.llm_prompt_json_dict = llm_prompt_json_dict

        # Caching frequently used setting values
        self._cached_settings = {
            'source_rag_target': None,
            'none_source_rag_target': None,
            'faq_category_rag_target_list': None
        }

        # Pre-defined language metadata
        self._language_data = {
            "ko": ("Korean", "한국어", "[참고문헌]"),
            "en": ("English", "영어", "[References]"),
            "jp": ("Japanese", "일본어", "[参考文献]"),
            "cn": ("Chinese", "중국어", "[参考文献]"),
        }

        # Pre-defined day names
        self._day_names = ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일']

        # Load initial settings
        self._load_cached_settings()

    def _load_cached_settings(self) -> None:
        """
        Preloads and caches frequently used setting values.

        Loads values that are frequently used from the settings object to reduce repeated parsing.
        """
        try:
            # Load source type settings
            self._cached_settings['source_rag_target'] = self.settings.prompt.source_type.split(',')
            self._cached_settings['none_source_rag_target'] = self.settings.prompt.none_source_type.split(',')
            self._cached_settings['faq_category_rag_target_list'] = self.settings.prompt.faq_type.split(',')

            logger.debug("Successfully loaded cached settings.")
        except AttributeError as e:
            logger.warning(f"Attribute error while loading settings: {e}")
        except Exception as e:
            logger.error(f"Unexpected error while caching settings: {e}")
            # Initialize with default values
            for key in self._cached_settings:
                self._cached_settings[key] = []

    def _get_cached_setting(self, setting_name: str) -> List[str]:
        """
        Returns a cached setting value.

        Args:
            setting_name (str): Name of the setting to retrieve

        Returns:
            List[str]: List of setting values, or empty list if not found
        """
        # Check value in cache
        cached_value = self._cached_settings.get(setting_name)

        # Return cached value if it exists
        if cached_value is not None:
            return cached_value  # type: ignore

        # Try to load from settings if not in cache
        try:
            if setting_name == 'source_rag_target':
                value = self.settings.prompt.source_type.split(',')
            elif setting_name == 'none_source_rag_target':
                value = self.settings.prompt.none_source_type.split(',')
            elif setting_name == 'faq_category_rag_target_list':
                value = self.settings.prompt.faq_type.split(',')
            else:
                return []

            # Update cache
            self._cached_settings[setting_name] = value
            return value
        except Exception:
            # Return empty list on error
            return []

    def is_faq_type_chatbot(self, current_rag_sys_info: str) -> bool:
        """
        Checks if the current RAG system information is of FAQ type.

        Args:
            current_rag_sys_info (str): Current RAG system information

        Returns:
            bool: True if it's an FAQ type, False otherwise
        """
        faq_targets = self._get_cached_setting('faq_category_rag_target_list')
        return current_rag_sys_info in faq_targets

    @lru_cache(maxsize=64)
    def get_rag_qa_prompt(self, rag_sys_info: str) -> str:
        """
        Retrieves the appropriate prompt based on RAG system information.

        Args:
            rag_sys_info (str): RAG system information

        Returns:
            str: Retrieved prompt, or empty string if not found
        """
        # Get cached settings
        source_rag_target = self._get_cached_setting('source_rag_target')
        none_source_rag_target = self._get_cached_setting('none_source_rag_target')

        try:
            # Determine prompt type
            if rag_sys_info in source_rag_target:
                prompt_type = "with-source-prompt"
            elif rag_sys_info in none_source_rag_target:
                prompt_type = "without-source-prompt"
            else:
                # Determine based on default priority
                prompt_type = "with-source-prompt" if self.settings.prompt.source_priority else "without-source-prompt"

            # Determine prompt key
            prompt_key = (rag_sys_info
                          if rag_sys_info in source_rag_target + none_source_rag_target
                          else "common-prompt")

            # Get prompt
            return self.llm_prompt_json_dict.get_prompt_data("prompts", prompt_type, prompt_key) or ""
        except Exception as e:
            logger.error(f"Error occurred during prompt retrieval: {e}")
            return ""

    def get_translation_language_word(self, lang: str) -> Tuple[str, str, str]:
        """
        Returns language name, translated name, and reference notation based on language code.

        Args:
            lang (str): Language code (ko, en, jp, cn)

        Returns:
            Tuple[str, str, str]: (English name, Local name, Reference notation)
        """
        # Handle invalid language code
        if not lang or not isinstance(lang, str) or lang not in self._language_data:
            return self._language_data["ko"]  # Return Korean as default

        return self._language_data[lang]

    def get_today(self) -> str:
        """
        Returns the current date and day of the week in Korean format.

        Returns:
            str: String in the format "YYYY년 MM월 DD일 요일 HH시 MM분"
        """
        try:
            today = datetime.now()
            weekday = self._day_names[localtime().tm_wday]
            return f"{today.strftime('%Y년 %m월 %d일')} {weekday} {today.strftime('%H시 %M분')}입니다."
        except Exception as e:
            logger.warning(f"Error during date formatting: {e}")
            # Fallback to simpler format
            return datetime.now().strftime('%Y년 %m월 %d일 %H시 %M분')

    def make_answer_reference(self, query_answer: str, rag_sys_info: str,
                              reference_word: str, retriever_documents: List[Document], request=None) -> str:
        """
        Adds reference document information to the answer.

        Args:
            query_answer (str): Original answer text
            rag_sys_info (str): RAG system information
            reference_word (str): Reference section indicator (e.g., "[References]")
            retriever_documents (List[Document]): List of documents to reference
            request (Optional[str]): Request data (default: None)

        Returns:
            str: Answer text with added reference information
        """
        # Return original if no answer or no reference documents
        if not query_answer or not retriever_documents:
            return query_answer

        # Check source type settings
        source_rag_target = self._get_cached_setting('source_rag_target')
        if rag_sys_info not in source_rag_target:
            return query_answer

        try:
            # Collect deduplicated document source information
            docs_source: Dict[str, str] = {}
            for doc in retriever_documents:
                # Extract document name and page
                doc_name = doc.metadata.get("doc_name") or doc.metadata.get("source", "Unknown Document")
                if isinstance(doc_name, str) and "," in doc_name:
                    doc_name = doc_name.split(",")[0].strip()

                doc_page = doc.metadata.get("doc_page", "N/A")

                # Remove leading '/' from path
                if isinstance(doc_name, str) and doc_name.startswith('/'):
                    doc_name = doc_name[1:]

                # Remove duplicate sources (reference each document only once)
                if doc_name not in docs_source:
                    docs_source[doc_name] = doc_page

            # Add reference information
            if docs_source:
                # Maximum number of sources to display (from settings or default)
                max_sources = min(
                    getattr(self.settings.prompt, 'source_count', len(docs_source)),
                    len(docs_source)
                )

                # Create reference section
                reference_section = f"\n\n---------\n{reference_word}"
                for i, (doc_name, doc_page) in enumerate(docs_source.items()):
                    if i >= max_sources:
                        break
                    reference_section += f"\n- {doc_name} (Page: {doc_page})"

                # Add reference section to original answer
                query_answer += reference_section

                # Log important information
                if len(docs_source) > 0:
                    log_with_session_id(logger.debug, f"{len(docs_source)} reference documents added to the answer",
                                        request)

            return query_answer

        except Exception as e:
            logger.warning(f"Error while adding reference information: {e}")
            return query_answer  # Return original answer on error

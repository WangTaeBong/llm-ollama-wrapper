import logging
import random
import re
from typing import Optional, Set, List

from src.schema.chat_req import ChatRequest

# Logger configuration
logger = logging.getLogger(__name__)


class QueryProcessor:
    """
    Class responsible for query processing functionalities.

    Handles the cleaning, filtering, pattern checking, and FAQ query composition of user input queries.
    """

    def __init__(self, settings, query_check_json_dict):
        """
        Constructor for the QueryProcessor class.

        Args:
            settings: Object containing configuration settings
            query_check_json_dict: Dictionary containing query patterns and responses
        """
        self.settings = settings
        self.query_check_json_dict = query_check_json_dict

        # Compiled regular expression patterns (pre-compiled in constructor for performance optimization)
        self._clean_query_pattern = re.compile(r'[~!@#$%^&*()=+\[\]{}:?,<>/\-_.]')
        self._ko_jamo_pattern = re.compile(r'([ㄱ-ㅎㅏ-ㅣ\s]+)')
        self._arabia_num_pattern = re.compile(r'([0-9]+)')
        self._wild_char_pattern = re.compile(r'([^\w\s]+)')

        # Caching frequently used data for performance improvement
        self._faq_category_rag_targets: List[str] = self.settings.prompt.faq_type.split(',')
        self._excluded_categories: Set[str] = {"담당자 메일 문의", "AI 직접 질문", "챗봇 문의"}

    def clean_query(self, query: str) -> str:
        """
        Cleans the user input query by removing unnecessary special characters and symbols.

        Args:
            query (str): Original query to be cleaned

        Returns:
            str: Cleaned query
        """
        if not query:
            return ""
        return self._clean_query_pattern.sub('', query.lower())

    def filter_query(self, query: str) -> str:
        """
        Applies filters to remove unnecessary characters from the query.

        Args:
            query (str): Original query to be filtered

        Returns:
            str: Filtered query
        """
        if not query:
            return ""

        # Low need for exception handling - only checking for configuration access errors
        try:
            # Apply filtering patterns according to settings
            if self.settings.query_filter.ko_jamo:
                query = self._ko_jamo_pattern.sub('', query)
            if self.settings.query_filter.arabia_num:
                query = self._arabia_num_pattern.sub('', query)
            if self.settings.query_filter.wild_char:
                query = self._wild_char_pattern.sub('', query)
            return query
        except AttributeError as e:
            # Only log errors related to settings access
            logger.warning(f"Settings access error in filter_query: {e}")
            return query

    def check_query_sentence(self, request: ChatRequest) -> Optional[str]:
        """
        Compares the user query with predefined response patterns to generate an appropriate response.
        Recognizes greetings like 'hi', 'hello' and other specific patterns to return appropriate responses.

        Args:
            request (ChatRequest): Chat request object to be processed

        Returns:
            Optional[str]: Response string if there's a match, None otherwise
        """
        # Get settings
        query_lang_key_list = self.settings.lm_check.query_lang_key.split(',')
        query_dict_key_list = self.settings.lm_check.query_dict_key.split(',')

        if not query_lang_key_list or not query_dict_key_list:
            return None

        # User's language code (ko, en, jp, cn, etc.)
        user_lang = request.chat.lang

        # Clean the query
        raw_query = self.clean_query(request.chat.user)

        # Handle queries that are too short or contain only digits
        if len(raw_query) < 2 or raw_query.isdigit():
            farewells_msgs = self.query_check_json_dict.get_dict_data(user_lang, "farewells_msg")
            if farewells_msgs:
                return random.choice(farewells_msgs)
            return None

        # Pattern matching - check for all language keys and all dictionary keys
        try:
            # Initialize cache - pattern dictionary for each language code
            pattern_cache = {}

            # 1. First check patterns that match the user's language code (optimization)
            if user_lang in query_lang_key_list:
                for data_dict in query_dict_key_list:
                    # Get pattern list (greetings, endings, etc.)
                    patterns = self.query_check_json_dict.get_dict_data(user_lang, data_dict)
                    if not patterns:
                        continue

                    # Convert patterns to a set and cache it (performance optimization)
                    pattern_set = set(patterns)
                    pattern_cache[(user_lang, data_dict)] = pattern_set

                    # Check if current query exists in the pattern set
                    if raw_query in pattern_set:
                        # Construct response key (e.g., greetings → greetings_msg)
                        response_key = f"{data_dict}_msg"
                        # Get response messages for the matching language
                        response_messages = self.query_check_json_dict.get_dict_data(user_lang, response_key)
                        if response_messages:
                            return random.choice(response_messages)

            # 2. If not found in the user's language code, check other language codes
            for chat_lang in query_lang_key_list:
                # Skip the user's language that was already checked
                if chat_lang == user_lang:
                    continue

                for data_dict in query_dict_key_list:
                    # Reuse from cache if available, otherwise fetch and cache
                    if (chat_lang, data_dict) in pattern_cache:
                        pattern_set = pattern_cache[(chat_lang, data_dict)]
                    else:
                        patterns = self.query_check_json_dict.get_dict_data(chat_lang, data_dict)
                        if not patterns:
                            continue
                        pattern_set = set(patterns)
                        pattern_cache[(chat_lang, data_dict)] = pattern_set

                    # Pattern matching
                    if raw_query in pattern_set:
                        # Pattern found in another language, but provide response in user's language
                        response_key = f"{data_dict}_msg"
                        response_messages = self.query_check_json_dict.get_dict_data(user_lang, response_key)
                        if response_messages:
                            return random.choice(response_messages)

            # Return None if no pattern matches
            return None

        except Exception as e:
            logger.warning(f"[{request.meta.session_id}] Error in check_query_sentence: {e}")
            # Return None on error
            return None

    def construct_faq_query(self, request: ChatRequest) -> str:
        """
        Creates an optimized LLM query based on FAQ categories.

        Args:
            request (ChatRequest): Chat request object to be processed

        Returns:
            str: Constructed FAQ query or the original query
        """
        # Check if the RAG system info corresponds to an FAQ type
        if request.meta.rag_sys_info not in self._faq_category_rag_targets:
            return request.chat.user

        # Query construction
        query_parts = []

        # Process by category
        category_suffixes = [
            (request.chat.category1, " belongs to this category."),
            (request.chat.category2, " belongs to this category."),
            (request.chat.category3, " is this information.")
        ]

        for category, suffix in category_suffixes:
            if category and category not in self._excluded_categories:
                query_parts.append(f"{category}{suffix}")

        # Combine with the original query
        if query_parts:
            return " ".join(query_parts) + " " + request.chat.user

        return request.chat.user

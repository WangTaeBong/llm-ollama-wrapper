import logging
import re
from functools import lru_cache
from typing import Dict, List, Set, Pattern

# Module level logger configuration
logger = logging.getLogger(__name__)


class VOCLinkProcessor:
    """
    Class responsible for VOC link processing functionality

    Provides validation, processing, and transformation of electronic approval document links.
    Includes regex pattern caching and performance optimization features.
    """

    def __init__(self, settings):
        """
        Constructor for the VOCLinkProcessor class

        Args:
            settings: Object containing configuration information
        """
        self.settings = settings

        # Cache settings values
        self._cached_settings: Dict[str, str] = self._load_cached_settings()

        # Compile and cache regex patterns
        self._compiled_patterns: Dict[str, Pattern] = {}
        self._init_patterns()

        # Define exception patterns
        self._excluded_doc_ids: Set[str] = {"12345678", "00000000"}

        logger.debug("VOCLinkProcessor instance has been initialized")

    def _load_cached_settings(self) -> Dict[str, str]:
        """
        Cache frequently used configuration values.

        Returns:
            Dict[str, str]: Cached configuration values
        """
        try:
            return {
                'url_pattern': getattr(self.settings.voc, 'gw_doc_id_link_url_pattern', ''),
                'correct_pattern': getattr(self.settings.voc, 'gw_doc_id_link_correct_pattern', ''),
                'check_gw_word': getattr(self.settings.voc, 'check_gw_word_link', ''),
                'check_block_line': getattr(self.settings.voc, 'check_block_line', '')
            }
        except AttributeError as e:
            logger.error(f"Error accessing configuration attributes: {e}")
            return {
                'url_pattern': '',
                'correct_pattern': '',
                'check_gw_word': '',
                'check_block_line': ''
            }

    def _init_patterns(self):
        """
        Compile and cache frequently used regex patterns.
        """
        try:
            url_pattern = self._cached_settings.get('url_pattern', '')
            correct_pattern = self._cached_settings.get('correct_pattern', '')

            if url_pattern:
                self._compiled_patterns['url'] = re.compile(url_pattern)
            if correct_pattern:
                self._compiled_patterns['correct'] = re.compile(correct_pattern)

                # Optimized pattern: Directly check if URL ends with 8 digits
                optimized_pattern = fr"{correct_pattern.rstrip('$')}\/(\d{{8}})(?:$|[?#])"
                self._compiled_patterns['optimized'] = re.compile(optimized_pattern)

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"{len(self._compiled_patterns)} regex patterns have been compiled")
        except re.error as e:
            logger.error(f"Regex compilation error: {e}")
        except Exception as e:
            logger.error(f"Error during pattern initialization: {e}")

    def voc_judge_gw_doc_id_link_pattern(self, source: str, pattern_key: str = 'url') -> bool:
        """
        Check if a specific URL pattern exists in a string.

        Args:
            source (str): String to check
            pattern_key (str): Pattern key to use (default: 'url')

        Returns:
            bool: True if pattern exists, False otherwise
        """
        if not source or not isinstance(source, str):
            return False

        # Get compiled pattern
        pattern = self._compiled_patterns.get(pattern_key)
        if not pattern:
            # Use original configuration if pattern not found
            pattern_str = self._cached_settings.get(f'{pattern_key}_pattern', '')
            if not pattern_str:
                return False

            try:
                return bool(re.search(pattern_str, source))
            except re.error as e:
                logger.error(f"Invalid regex pattern: {pattern_str}, error: {e}")
                return False

        try:
            return bool(pattern.search(source))
        except Exception as e:
            logger.error(f"Error occurred during pattern search: {e}")
            return False

    @lru_cache(maxsize=64)
    def _is_valid_doc_id(self, doc_id: str) -> bool:
        """
        Check if document ID is valid. (Caching applied)

        Args:
            doc_id (str): Document ID to check

        Returns:
            bool: True if ID is valid, False otherwise
        """
        return (doc_id.isdigit() and
                len(doc_id) == 8 and
                doc_id not in self._excluded_doc_ids)

    def voc_judge_gw_doc_id_link_correct_pattern(self, source: str, pattern_key: str = 'correct') -> bool:
        """
        Check if URL pattern ends with 8 digits and excludes specific exceptions.

        URL must end with 8 digits and not be an exception value (12345678, 00000000).
        Uses optimized regex for efficient checking.

        Args:
            source (str): String to check
            pattern_key (str): Pattern key to use (default: 'correct')

        Returns:
            bool: True if valid pattern exists, False otherwise
        """
        if not source or not isinstance(source, str):
            return False

        try:
            # Try using optimized pattern
            optimized_pattern = self._compiled_patterns.get('optimized')
            if optimized_pattern:
                matches = optimized_pattern.findall(source)
                return any(self._is_valid_doc_id(doc_id) for doc_id in matches)

            # Use original method if optimized pattern is not available
            pattern = self._compiled_patterns.get(pattern_key)
            if not pattern:
                pattern_str = self._cached_settings.get(f'{pattern_key}_pattern', '')
                if not pattern_str:
                    return False
                matches = re.findall(pattern_str, source)
            else:
                matches = pattern.findall(source)

            # Validate each match
            for match in matches:
                # Extract the last part of the URL
                end_path = match.split('/')[-1]

                # Validity check (caching applied)
                if self._is_valid_doc_id(end_path):
                    return True

            return False

        except Exception as e:
            logger.error(f"Error occurred during pattern validation: {e}")
            return False

    def make_komico_voc_groupware_docid_url(self, query_answer: str) -> str:
        """
        Validate electronic approval links in chatbot responses and remove unnecessary parts.

        Args:
            query_answer (str): Original response text to process

        Returns:
            str: Processed response text
        """
        if not query_answer:
            return ""

        try:
            # Get patterns from settings
            url_pattern = self._cached_settings.get('url_pattern', '')
            correct_pattern = self._cached_settings.get('correct_pattern', '')

            if not url_pattern or not correct_pattern:
                if logger.isEnabledFor(logging.WARNING):
                    logger.warning("URL pattern settings not found. Returning original text.")
                return query_answer

            # Process response text line by line
            lines = query_answer.split('\n')
            processed_lines: List[str] = []
            is_link_exist = False

            for line in lines:
                # Add empty lines as is
                if not line:
                    processed_lines.append(line)
                    continue

                # Check if URL pattern exists
                if self.voc_judge_gw_doc_id_link_pattern(line):
                    # Add if it's a correct pattern
                    if self.voc_judge_gw_doc_id_link_correct_pattern(line):
                        processed_lines.append(line)
                        is_link_exist = True
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f"Found valid document link: {line[:50]}...")
                    else:
                        # Exclude incorrect patterns
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f"Excluded incorrect link pattern: {line[:50]}...")
                else:
                    # Add normal text without URL patterns as is
                    processed_lines.append(line)

            # Remove specific words/lines if no links found
            if not is_link_exist:
                check_gw_word = self._cached_settings.get('check_gw_word', '')
                check_block_line = self._cached_settings.get('check_block_line', '')

                if check_gw_word or check_block_line:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug("Removing related guidance text because no document links were found")

                    # Filter in one iteration
                    filtered_lines: List[str] = []
                    for line in processed_lines:
                        has_gw_word = check_gw_word and check_gw_word in line
                        has_block_line = check_block_line and check_block_line in line
                        if not (has_gw_word or has_block_line):
                            filtered_lines.append(line)

                    processed_lines = filtered_lines

            # Combine results
            result = '\n'.join(processed_lines).strip()

            return result

        except AttributeError as e:
            logger.error(f"Error accessing configuration attributes: {e}")
            return query_answer  # Return original in case of configuration error
        except Exception as e:
            logger.error(f"Error occurred during link processing: {e}")
            return query_answer  # Return original in case of other errors

    def reload_settings(self) -> bool:
        """
        Reload settings and patterns.
        Call this when settings have changed dynamically.

        Returns:
            bool: Success status of reload operation
        """
        try:
            # Reload cached settings
            self._cached_settings = self._load_cached_settings()

            # Clear lru_cache
            self._is_valid_doc_id.cache_clear()

            # Recompile patterns
            self._compiled_patterns.clear()
            self._init_patterns()

            logger.info("VOCLinkProcessor settings have been reloaded")
            return True
        except Exception as e:
            logger.error(f"Error occurred during settings reload: {e}")
            return False

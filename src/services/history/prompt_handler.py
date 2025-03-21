# services/history/prompt_handler.py
"""
Prompt handling for history-aware systems.

This module provides specialized prompt building and management
for conversation history integration.
"""

import logging
import re
from typing import Dict, Any

from src.common.config_loader import ConfigLoader

# Load settings
settings = ConfigLoader().get_settings()
logger = logging.getLogger(__name__)


class HistoryPromptHandler:
    """
    Builds and manages prompts for history-aware conversation systems.
    """

    @staticmethod
    def build_system_prompt(template: str, context: Dict[str, Any]) -> str:
        """
        Build a system prompt by applying context variables to a template.

        Args:
            template: The prompt template
            context: The context variables

        Returns:
            str: The formatted prompt
        """
        try:
            return template.format(**context)
        except KeyError as e:
            # Handle missing keys gracefully
            missing_key = str(e).strip("'")
            logger.warning(f"Missing key in system prompt template: {missing_key}, setting to empty string")
            context[missing_key] = ""
            return template.format(**context)
        except Exception as e:
            logger.error(f"Error formatting system prompt: {e}")
            # Fallback to basic prompt if formatting fails completely
            return f"Answer the following question based on the context: {context.get('input', 'No input provided')}"

    @staticmethod
    def build_system_prompt_gemma(template: str, context: Dict[str, Any]) -> str:
        """
        Build a system prompt in Gemma format.

        Args:
            template: The prompt template
            context: The context variables

        Returns:
            str: Gemma-formatted prompt
        """
        try:
            # Generate regular prompt first
            raw_prompt = HistoryPromptHandler.build_system_prompt(template, context)

            # Convert to Gemma format
            formatted_prompt = "<start_of_turn>user\n"
            formatted_prompt += raw_prompt
            formatted_prompt += "\n<end_of_turn>\n<start_of_turn>model\n"

            return formatted_prompt

        except KeyError as e:
            # Handle missing keys
            missing_key = str(e).strip("'")
            logger.warning(f"Missing key in Gemma prompt template: {missing_key}, setting to empty string")
            context[missing_key] = ""
            return HistoryPromptHandler.build_system_prompt_gemma(template, context)
        except Exception as e:
            # Handle other errors
            logger.error(f"Error formatting Gemma prompt: {e}")
            # Fallback to basic Gemma prompt
            basic_prompt = f"<start_of_turn>user\nPlease answer the following question: {context.get('input', 'No question provided')}\n<end_of_turn>\n<start_of_turn>model\n"
            return basic_prompt

    @staticmethod
    def build_improved_system_prompt(template: str, context: Dict[str, Any]) -> str:
        """
        Build an improved system prompt with rewritten question integration.

        Args:
            template: The prompt template
            context: The context variables including rewritten_question

        Returns:
            str: The improved formatted prompt
        """
        try:
            # Add rewritten question instruction if not already in template
            if "rewritten_question" in context and "{rewritten_question}" not in template:
                insert_point = template.find("{input}")
                if insert_point > 0:
                    instruction = "\n\n# Rewritten Question\nThe following is a clarified version of the question based on conversation context. Consider this when generating your response:\n{rewritten_question}\n\n# Original Question\n"
                    template = template[:insert_point] + instruction + template[insert_point:]

            # Check for required keys
            required_keys = set()
            for match in re.finditer(r"{(\w+)}", template):
                required_keys.add(match.group(1))

            # Set missing keys to empty strings
            for key in required_keys:
                if key not in context:
                    logger.warning(f"Missing key in template: {key}, setting to empty string")
                    context[key] = ""

            # Format the template
            return template.format(**context)

        except KeyError as e:
            # Handle missing keys
            missing_key = str(e).strip("'")
            logger.warning(f"Missing key in improved prompt template: {missing_key}, setting to empty string")
            context[missing_key] = ""
            return template.format(**context)
        except Exception as e:
            # Handle other errors
            logger.error(f"Error formatting improved prompt: {e}")
            # Fallback to basic prompt
            return f"Answer the following question based on the provided context:\n\nContext: {context.get('context', '')}\n\nQuestion: {context.get('input', 'No question provided')}"

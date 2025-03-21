"""
History formatting utilities.

This module provides functions for formatting conversation history in various formats
suitable for different LLM models and use cases.
"""

import logging

from langchain_community.chat_message_histories import ChatMessageHistory

logger = logging.getLogger(__name__)


class HistoryFormatter:
    """
    Formats conversation history for different contexts and models.
    """

    @classmethod
    def format_for_prompt(cls, session_history: ChatMessageHistory, max_turns: int = 5) -> str:
        """
        Format conversation history for inclusion in prompts.
        Uses an improved format to effectively capture context from previous conversations.

        Args:
            session_history: Chat message history
            max_turns: Maximum number of conversation turns to include (default: 5)

        Returns:
            str: Formatted conversation history string
        """
        try:
            # Parameter validation
            if not session_history or not hasattr(session_history, 'messages'):
                logger.warning("Invalid session_history object provided")
                return ""

            messages = session_history.messages
            if not messages:
                return ""

            # Extract most recent turns up to max_turns
            if len(messages) > max_turns * 2:  # Each turn includes user message and system response
                messages = messages[-(max_turns * 2):]

            formatted_history = []

            # Add prompt header
            formatted_history.append("# Previous Conversation")

            # Process conversation turns
            turns = []
            current_turn = {"user": None, "assistant": None}

            for msg in messages:
                # Type checking
                if hasattr(msg, '__class__') and hasattr(msg.__class__, '__name__'):
                    msg_type = msg.__class__.__name__
                else:
                    msg_type = str(type(msg))

                # Handle user message
                if isinstance(msg, HumanMessage) or "HumanMessage" in msg_type:
                    # Save completed turn if exists
                    if current_turn["user"] is not None and current_turn["assistant"] is not None:
                        turns.append(current_turn)
                        current_turn = {"user": None, "assistant": None}

                    # Store current user message
                    if hasattr(msg, 'content'):
                        current_turn["user"] = msg.content
                    else:
                        current_turn["user"] = str(msg)

                # Handle assistant message
                elif isinstance(msg, AIMessage) or "AIMessage" in msg_type:
                    if hasattr(msg, 'content'):
                        current_turn["assistant"] = msg.content
                    else:
                        current_turn["assistant"] = str(msg)

            # Include the final turn
            if current_turn["user"] is not None:
                turns.append(current_turn)

            # Limit to max_turns if needed
            if len(turns) > max_turns:
                turns = turns[-max_turns:]

            # Format each turn
            for i, turn in enumerate(turns):
                formatted_history.append(f"\n## Conversation {i + 1}")

                if turn["user"]:
                    formatted_history.append(f"User: {turn['user']}")

                if turn["assistant"]:
                    formatted_history.append(f"Assistant: {turn['assistant']}")

            # Add guidance for using history
            formatted_history.append("\n# Consider the above conversation when answering the current question.")

            return "\n".join(formatted_history)

        except Exception as e:
            logger.error(f"Error formatting conversation history: {str(e)}")
            return ""

    @classmethod
    def format_for_gemma(cls, session_history: ChatMessageHistory, max_turns: int = 5) -> str:
        """
        Format conversation history for Gemma models.
        Uses Gemma's <start_of_turn>user/<start_of_turn>model format.

        Args:
            session_history: Chat message history
            max_turns: Maximum number of conversation turns to include (default: 5)

        Returns:
            str: Gemma-formatted conversation history string
        """
        try:
            # Parameter validation
            if not session_history or not hasattr(session_history, 'messages'):
                logger.warning("Invalid session_history object provided")
                return ""

            messages = session_history.messages
            if not messages:
                return ""

            # Extract most recent turns up to max_turns
            if len(messages) > max_turns * 2:
                messages = messages[-(max_turns * 2):]

            formatted_history = []

            # Process conversation pairs into Gemma format
            for i in range(0, len(messages), 2):
                # User message
                if i < len(messages):
                    user_msg = messages[i]
                    if hasattr(user_msg, 'content'):
                        formatted_history.append(f"<start_of_turn>user\n{user_msg.content}<end_of_turn>")

                # Assistant message
                if i + 1 < len(messages):
                    sys_msg = messages[i + 1]
                    if hasattr(sys_msg, 'content'):
                        formatted_history.append(f"<start_of_turn>model\n{sys_msg.content}<end_of_turn>")

            return "\n".join(formatted_history)

        except Exception as e:
            logger.error(f"Error formatting Gemma conversation history: {str(e)}")
            return ""

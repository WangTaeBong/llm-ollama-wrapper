import json
import logging

from langchain_core.prompts import PromptTemplate
from src.common.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


class PromptManager:
    """Manages prompt templates for the application."""

    _prompt_data = None
    _history_prompt_data = None

    @classmethod
    def _load_history_prompt_data(cls):
        """Load history prompt data from JSON file."""
        try:
            settings = ConfigLoader().get_settings()

            history_prompt_path = settings.prompt.history_prompt_path
            with open(history_prompt_path, 'r', encoding='utf-8') as file:
                cls._history_prompt_data = json.load(file)
                logger.debug(f"History prompt data loaded from {history_prompt_path}")
        except Exception as e:
            logger.error(f"Error loading history prompt data: {e}")
            # Initialize with empty data if loading fails
            cls._history_prompt_data = {"prompts": {}}

    @classmethod
    def get_contextualize_q_prompt(cls):
        """Get the contextualize question prompt template."""
        if cls._history_prompt_data is None:
            cls._load_history_prompt_data()

        prompt_text = cls._history_prompt_data.get("prompts", {}).get(
            "contextualize_q_system_prompt",
            "System prompt for contextualizing questions could not be loaded."
        )

        # return prompt_text
        return PromptTemplate(
            template=prompt_text,
            input_variables=["input", "chat_history"]
        )

    @classmethod
    def get_rewrite_prompt_template(cls):
        """Get the rewrite prompt template."""
        if cls._history_prompt_data is None:
            cls._load_history_prompt_data()

        prompt_text = cls._history_prompt_data.get("prompts", {}).get(
            "rewrite_prompt_template",
            "당신은 대화 컨텍스트를 고려하여 사용자의 질문을 명확하고 완전한 형태로 재작성하는 AI 도우미입니다.\n이전 대화 내용과 현재 질문을 고려하여, 대화 맥락이 충분히 반영된 독립적인 질문으로 재작성해주세요.\n다음 정보를 고려하세요:\n1. 현재 질문에서 생략된 맥락을 이전 대화에서 찾아 보완하세요.\n2. 대명사(이것, 그것, 저것 등)는 실제 지칭하는 대상으로 바꿔주세요.\n3. 간결하면서도 정확한 질문으로 재작성하세요.\n4. 재작성된 질문만 출력하세요. 설명이나 다른 텍스트는 포함하지 마세요.\n\n{history}\n\n현재 질문: {input}\n\n재작성된 질문:"
        )

        return prompt_text

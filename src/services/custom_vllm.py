import uuid
from typing import Any, List, Optional, Dict, AsyncGenerator

import aiohttp
import asyncio
from langchain.callbacks.manager import AsyncCallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.schema import Generation, LLMResult
import logging


class CustomVLLM(LLM):
    api_url: str

    async def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
            **kwargs: Any
    ) -> str:
        stream = kwargs.get("stream", False)

        if stream:
            return "".join([chunk async for chunk in self._stream_call(prompt, stop, run_manager, **kwargs)])
        else:
            return await self._non_stream_call(prompt, stop, run_manager, **kwargs)

    async def _non_stream_call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
            **kwargs: Any
    ) -> str:
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.api_url}/generate_full", json={
                "request_id": str(kwargs.get("request_id", uuid.uuid4())),
                "query": prompt,
            }) as response:
                result = await response.json()
                if "response" not in result:
                    logging.error(f"Invalid API response format: {result}")
                    return {"answer": "Error: Invalid LLM response", "context": []}
                return {"answer": result["response"], "context": []}  # 응답 형식 명확히 지정

    async def _stream_call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
            **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                    f"{self.api_url}/generate",
                    json={
                        "request_id": str(kwargs.get("request_id", uuid.uuid4())),
                        "query": prompt,
                    },
                    chunked=True
            ) as response:
                async for chunk in response.content.iter_any():
                    if chunk:  # 빈 청크 무시
                        text = chunk.decode('utf-8')
                        yield text
                        if run_manager:
                            await run_manager.on_llm_new_token(text)

    async def _agenerate(
            self,
            prompts: List[str],
            stop: Optional[List[str]] = None,
            run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
            **kwargs: Any
    ) -> LLMResult:
        generations = []
        for prompt in prompts:
            if kwargs.get("stream", False):
                text = "".join([chunk async for chunk in self._stream_call(prompt, stop, run_manager, **kwargs)])
            else:
                text = await self._non_stream_call(prompt, stop, run_manager, **kwargs)
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations)

    @property
    def _llm_type(self) -> str:
        return "async_custom_vllm"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"api_url": self.api_url}

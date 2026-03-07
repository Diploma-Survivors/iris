"""
BackendLLM Plugin

Routes LLM inference through sfinx-backend's LangChain/Gemini pipeline so
voice and chat share the same model, prompt, and LangSmith tracing.
"""

import asyncio
import contextlib
import json
import logging
import uuid
from typing import Any

import httpx
from livekit.agents.llm import ChatContext
from livekit.agents.llm.llm import (
    LLM,
    ChatChunk,
    ChoiceDelta,
    LLMStream,
)
from livekit.agents.llm.tool_context import Tool, ToolChoice
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)

from backend_client import backend_client as _backend_client

logger = logging.getLogger("backend_llm")

# Persistent client — avoids TCP handshake overhead on every voice turn.
# Keepalive connections mean the first token arrives much faster.
_http_client = httpx.AsyncClient(
    timeout=httpx.Timeout(connect=5.0, read=60.0, write=10.0, pool=5.0),
)


class BackendLLMStream(LLMStream):
    def __init__(
        self,
        owner: "BackendLLM",
        *,
        chat_ctx: ChatContext,
        tools: list[Tool],
        conn_options: APIConnectOptions,
        interview_id: str,
    ) -> None:
        super().__init__(
            owner, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options
        )
        self._interview_id = interview_id
        # Holds the active streaming response so aclose() can abort it instantly
        self._active_response: httpx.Response | None = None

    async def aclose(self) -> None:
        """Abort any in-flight HTTP stream immediately when LiveKit interrupts."""
        if self._active_response is not None:
            with contextlib.suppress(Exception):
                await self._active_response.aclose()
            self._active_response = None
        await super().aclose()

    async def _run(self) -> None:
        # Find the last user message in the conversation
        last_user_content: str | None = None
        for msg in reversed(self._chat_ctx.messages()):
            if msg.role == "user":
                last_user_content = msg.text_content
                break

        if last_user_content is None:
            # Greeting path
            text = await _backend_client.voice_start(self._interview_id)
            await self._event_ch.send(
                ChatChunk(
                    id=str(uuid.uuid4()),
                    delta=ChoiceDelta(role="assistant", content=text),
                )
            )
        else:
            # Voice message: stream tokens so TTS starts on first sentence
            await self._stream_voice_message(last_user_content)

    async def _stream_voice_message(self, content: str) -> None:
        url = (
            f"{_backend_client.base_url}/internal/ai-interviews"
            f"/{self._interview_id}/voice-message/stream"
        )
        received_any = False
        try:
            async with _http_client.stream(
                "POST",
                url,
                headers=_backend_client._headers(),
                json={"content": content},
                timeout=30.0,
            ) as response:
                self._active_response = response
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    try:
                        data = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue

                    if data.get("token"):
                        received_any = True
                        await self._event_ch.send(
                            ChatChunk(
                                id=str(uuid.uuid4()),
                                delta=ChoiceDelta(
                                    role="assistant", content=data["token"]
                                ),
                            )
                        )
                    elif data.get("done") or data.get("error"):
                        break
        except asyncio.CancelledError:
            raise  # Let the framework handle interruption cleanly
        except Exception as e:
            logger.exception(f"Error streaming voice message: {e}")
        finally:
            self._active_response = None

        if not received_any:
            # Fallback: single-chunk non-streaming call
            text = await _backend_client.voice_message(self._interview_id, content)
            await self._event_ch.send(
                ChatChunk(
                    id=str(uuid.uuid4()),
                    delta=ChoiceDelta(role="assistant", content=text),
                )
            )


class BackendLLM(LLM):
    def __init__(self, interview_id: str) -> None:
        super().__init__()
        self._interview_id = interview_id

    @property
    def model(self) -> str:
        return "sfinx-backend/gemini"

    @property
    def provider(self) -> str:
        return "sfinx-backend"

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> LLMStream:
        return BackendLLMStream(
            self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
            interview_id=self._interview_id,
        )

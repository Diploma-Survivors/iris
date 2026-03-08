"""
Iris - Voice Interview Agent
"""

import asyncio
import json
import logging
import time

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    cli,
    room_io,
)
from livekit.plugins import deepgram, elevenlabs, noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from backend_client import backend_client
from backend_llm import BackendLLM
from interviewer import InterviewerAgent

logger = logging.getLogger("agent")
load_dotenv(".env.local")

SUPPORTED_LANGUAGES = {"en", "vi"}

server = AgentServer()
_background_tasks: set[asyncio.Task] = set()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load(
        min_speech_duration=0.1,  # Min speech to register (filter brief noise)
        min_silence_duration=0.5,  # How long user must be quiet before VAD fires
    )


server.setup_fnc = prewarm


async def send_to_frontend(room: rtc.Room, msg_type: str, data: dict):
    """Send data channel message to frontend."""
    try:
        payload = json.dumps({"type": msg_type, **data})
        await room.local_participant.publish_data(
            payload=payload.encode("utf-8"),
            reliable=True,
        )
    except Exception as e:
        logger.error(f"Failed to send to frontend: {e}")


def _create_stt(language: str):
    """Create STT plugin configured for the given language."""
    if language == "vi":
        return deepgram.STT(model="nova-3", language="vi")
    return deepgram.STT(
        model="nova-3",
        language="en-US",
        keyterm=[
            "pointer",
            "node",
            "linked list",
            "algorithm",
            "recursion",
            "iteration",
            "array",
            "hash map",
            "binary search",
            "time complexity",
            "space complexity",
            "O(n)",
            "O(log n)",
            "traverse",
            "reverse",
            "iterate",
            "recursive",
        ],
    )


def _create_tts(language: str):
    """Create TTS plugin configured for the given language."""
    if language == "vi":
        return elevenlabs.TTS(model="eleven_turbo_v2_5")
    return deepgram.TTS(model="aura-2-thalia-en")


def _detect_language(context: dict | None) -> str:
    """Extract language from backend context, defaulting to English."""
    if not context:
        return "en"
    lang = context.get("language", "en")
    if lang not in SUPPORTED_LANGUAGES:
        logger.warning(f"Unsupported language '{lang}', falling back to English")
        return "en"
    return lang


@server.rtc_session()
async def interview_agent(ctx: JobContext):
    """Main agent session."""
    t_start = time.monotonic()
    logger.info(f"[TIMING] Agent joining room: {ctx.room.name}")

    interview_id = backend_client.extract_interview_id(ctx.room.name)

    # Fetch interview context to determine language before creating the session
    context = await backend_client.get_interview_context(interview_id) if interview_id else None
    language = _detect_language(context)
    logger.info(f"Interview {interview_id}: language={language}")

    streaming_msg_id: str | None = None
    is_streaming = False
    background_tasks = set()

    session = AgentSession(
        stt=_create_stt(language),
        llm=BackendLLM(interview_id=interview_id or ""),
        tts=_create_tts(language),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=False,
        min_endpointing_delay=0.4,
        max_endpointing_delay=3.0,
    )

    @session.on("conversation_item_added")
    def on_conversation_item_added(event):
        """Send transcripts to frontend for live display."""
        if not interview_id:
            return

        item = event.item
        if not hasattr(item, "role") or not hasattr(item, "content"):
            return

        role = "user" if item.role == "user" else "assistant"

        # Extract content
        content = ""
        if isinstance(item.content, str):
            content = item.content
        elif isinstance(item.content, list):
            for part in item.content:
                if hasattr(part, "text"):
                    content += part.text
                elif isinstance(part, str):
                    content += part

        if not content.strip():
            return

        logger.info(f"{role} message: {content[:60]}...")

        async def send_to_ui():
            msg_id = f"{role}-{asyncio.get_event_loop().time()}"
            await send_to_frontend(
                ctx.room,
                "transcript",
                {
                    "role": role,
                    "content": content,
                    "messageId": msg_id,
                    "timestamp": asyncio.get_event_loop().time(),
                },
            )

        task = asyncio.create_task(send_to_ui())
        background_tasks.add(task)
        task.add_done_callback(background_tasks.discard)

    @session.on("generation_chunk")
    def on_generation_chunk(event):
        """Stream LLM token chunks to frontend for live text display."""
        nonlocal streaming_msg_id, is_streaming

        text = ""
        if hasattr(event, "chunk") and event.chunk:
            chunk = event.chunk
            # ChatChunk stores text in delta.content
            if hasattr(chunk, "delta") and chunk.delta and chunk.delta.content:
                text = chunk.delta.content
            else:
                text = getattr(chunk, "text", "") or getattr(chunk, "content", "")
        elif hasattr(event, "delta"):
            text = event.delta
        elif hasattr(event, "text"):
            text = event.text

        if not text:
            return

        if not is_streaming:
            is_streaming = True
            streaming_msg_id = f"stream-{asyncio.get_event_loop().time()}"

            t = asyncio.create_task(
                send_to_frontend(ctx.room, "agent_status", {"status": "typing_start"})
            )
            background_tasks.add(t)
            t.add_done_callback(background_tasks.discard)

        async def send_delta():
            await send_to_frontend(
                ctx.room,
                "transcript_delta",
                {"role": "assistant", "delta": text, "messageId": streaming_msg_id},
            )

        task = asyncio.create_task(send_delta())
        background_tasks.add(task)
        task.add_done_callback(background_tasks.discard)

    @session.on("generation_done")
    def on_generation_done(event):
        """Signal generation complete to frontend (typing indicator off)."""
        nonlocal is_streaming, streaming_msg_id

        if is_streaming:

            async def send_done():
                await send_to_frontend(
                    ctx.room, "agent_status", {"status": "typing_end"}
                )

            task = asyncio.create_task(send_done())
            background_tasks.add(task)
            task.add_done_callback(background_tasks.discard)

            is_streaming = False
            streaming_msg_id = None

    # Create and start agent
    t_before_agent = time.monotonic()
    interviewer = InterviewerAgent(interview_id=interview_id, language=language)
    t_after_agent = time.monotonic()
    logger.info(
        f"[TIMING] InterviewerAgent init: {(t_after_agent - t_before_agent):.2f}s "
        f"(total: {(t_after_agent - t_start):.2f}s)"
    )

    t_before_start = time.monotonic()
    await session.start(
        agent=interviewer,
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: noise_cancellation.BVCTelephony()
                if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                else noise_cancellation.BVC(),
            ),
        ),
    )
    t_after_start = time.monotonic()
    logger.info(
        f"[TIMING] session.start (incl. on_enter greeting): "
        f"{(t_after_start - t_before_start):.2f}s (total: {(t_after_start - t_start):.2f}s)"
    )

    await ctx.connect()

    # Send ready status
    await send_to_frontend(ctx.room, "agent_status", {"status": "ready"})
    logger.info(
        f"[TIMING] Agent fully ready: {(time.monotonic() - t_start):.2f}s total | "
        f"room: {ctx.room.name}"
    )


if __name__ == "__main__":
    cli.run_app(server)

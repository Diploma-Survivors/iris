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
    inference,
    room_io,
)
from livekit.plugins import deepgram, noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from backend_client import backend_client
from interviewer import InterviewerAgent

logger = logging.getLogger("agent")
load_dotenv(".env.local")

server = AgentServer()
_background_tasks: set[asyncio.Task] = set()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load(
        min_speech_duration=0.1,   # Min speech to register (filter brief noise)
        min_silence_duration=1.3,  # Longer pause before end-of-turn (reduces interruptions)
    )


server.setup_fnc = prewarm


async def send_to_frontend(room: rtc.Room, msg_type: str, data: dict):
    """Send data channel message to frontend."""
    try:
        payload = json.dumps({"type": msg_type, **data})
        await room.local_participant.publish_data(
            payload=payload.encode('utf-8'),
            reliable=True,
        )
    except Exception as e:
        logger.error(f"Failed to send to frontend: {e}")


async def store_transcript(interview_id: str, role: str, content: str):
    """Store transcript to backend."""
    if interview_id and content:
        success = await backend_client.store_transcript(interview_id, role, content)
        if success:
            logger.info(f"Stored {role}: {content[:50]}...")


@server.rtc_session()
async def interview_agent(ctx: JobContext):
    """Main agent session."""
    t_start = time.monotonic()
    logger.info(f"[TIMING] Agent joining room: {ctx.room.name}")

    interview_id = backend_client.extract_interview_id(ctx.room.name)
    interview_context = None

    if interview_id:
        t_before_context = time.monotonic()
        interview_context = await backend_client.get_interview_context(interview_id)
        t_after_context = time.monotonic()
        logger.info(
            f"[TIMING] Context fetch: {(t_after_context - t_before_context):.2f}s "
            f"(total: {(t_after_context - t_start):.2f}s)"
        )
        logger.info(f"Context loaded for interview: {interview_id}")

    # Track streaming state
    streaming_msg_id = None
    streaming_content = ""
    is_streaming = False
    background_tasks = set()

    session = AgentSession(
        stt=deepgram.STT(
            model="nova-3",
            language="en-US",
            keyterm=[
                "pointer", "node", "linked list", "algorithm", "recursion", "iteration",
                "array", "hash map", "binary search", "time complexity", "space complexity",
                "O(n)", "O(log n)", "traverse", "reverse", "iterate", "recursive",
            ],
        ),
        llm=inference.LLM(model="openai/gpt-4.1-mini"),
        tts=deepgram.TTS(model="aura-2-thalia-en"),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=False,  # Wait for user to finish before generating (reduces cut-offs)
        min_endpointing_delay=1.2,   # Wait longer after user stops before responding
        max_endpointing_delay=5.0,   # Allow longer pauses when user might continue
    )

    @session.on("conversation_item_added")
    def on_conversation_item_added(event):
        """Handle both user and agent messages."""
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

        # Send to frontend via data channel
        async def send_and_store():
            msg_id = f"{role}-{asyncio.get_event_loop().time()}"
            await send_to_frontend(
                ctx.room,
                "transcript",
                {
                    "role": role,
                    "content": content,
                    "messageId": msg_id,
                    "timestamp": asyncio.get_event_loop().time(),
                }
            )
            await store_transcript(interview_id, role, content)

        task = asyncio.create_task(send_and_store())
        background_tasks.add(task)
        task.add_done_callback(background_tasks.discard)

    @session.on("generation_chunk")
    def on_generation_chunk(event):
        """Stream LLM chunks to frontend."""
        nonlocal streaming_msg_id, streaming_content, is_streaming

        # Extract text from event
        text = ""
        if hasattr(event, 'chunk') and event.chunk:
            text = getattr(event.chunk, 'text', str(event.chunk))
        elif hasattr(event, 'delta'):
            text = event.delta
        elif hasattr(event, 'text'):
            text = event.text

        if not text:
            return

        # Start new streaming message
        if not is_streaming:
            is_streaming = True
            streaming_msg_id = f"stream-{asyncio.get_event_loop().time()}"
            streaming_content = ""

            # Send typing indicator
            _t = asyncio.create_task(send_to_frontend(
                ctx.room, "agent_status", {"status": "typing_start"}
            ))
            _background_tasks.add(_t)
            _t.add_done_callback(_background_tasks.discard)

        streaming_content += text

        # Send chunk
        async def send_chunk():
            await send_to_frontend(
                ctx.room,
                "transcript_delta",
                {
                    "role": "assistant",
                    "delta": text,
                    "messageId": streaming_msg_id,
                }
            )

        task = asyncio.create_task(send_chunk())
        background_tasks.add(task)
        task.add_done_callback(background_tasks.discard)

    @session.on("generation_done")
    def on_generation_done(event):
        """Handle generation completion."""
        nonlocal is_streaming, streaming_content, streaming_msg_id

        if is_streaming:
            # Send final message
            async def send_final():
                await send_to_frontend(
                    ctx.room, "agent_status", {"status": "typing_end"}
                )
                if streaming_content:
                    await send_to_frontend(
                        ctx.room,
                        "transcript",
                        {
                            "role": "assistant",
                            "content": streaming_content,
                            "messageId": streaming_msg_id,
                            "isFinal": True,
                        }
                    )

            task = asyncio.create_task(send_final())
            background_tasks.add(task)
            task.add_done_callback(background_tasks.discard)

            # Reset state
            is_streaming = False
            streaming_content = ""
            streaming_msg_id = None

    # Create and start agent
    t_before_agent = time.monotonic()
    interviewer = InterviewerAgent(
        interview_context=interview_context,
        interview_id=interview_id,
    )
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

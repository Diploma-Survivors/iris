"""
Iris - Voice Interview Agent

PURPOSE:
Main entry point for the LiveKit voice agent that conducts AI coding interviews.
This module sets up the voice pipeline and handles the agent lifecycle.

HOW IT WORKS:
1. Agent server registers with LiveKit Cloud
2. When a user starts a voice interview, frontend requests a token from sfinx-backend
3. User joins the LiveKit room using the token
4. LiveKit dispatches this agent to join the same room
5. Agent fetches interview context from backend using room metadata
6. Voice conversation flows: User Speech → STT → LLM → TTS → Agent Speech
7. Transcripts are stored to backend for evaluation

COMPONENTS:
- STT (Speech-to-Text): AssemblyAI - Converts user speech to text
- LLM (Language Model): OpenAI GPT-4.1-mini - Generates responses
- TTS (Text-to-Speech): Cartesia Sonic-3 - Converts responses to speech
- VAD (Voice Activity Detection): Silero - Detects when user is speaking
- Turn Detector: Multilingual model - Determines when user finished speaking

ARCHITECTURE:
┌─────────────────────────────────────────────────────────────────┐
│                        LiveKit Room                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐    ┌─────────────────────────────────────────┐    │
│  │ Frontend │◄──►│              Iris Agent                 │    │
│  │  (User)  │    │  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐    │    │
│  └──────────┘    │  │ STT │→ │ LLM │→ │ TTS │→ │Audio│    │    │
│                  │  └─────┘  └─────┘  └─────┘  └─────┘    │    │
│                  └─────────────────────────────────────────┘    │
│                                  │                               │
└──────────────────────────────────┼───────────────────────────────┘
                                   ▼
                         ┌──────────────────┐
                         │  sfinx-backend   │
                         │  - Context API   │
                         │  - Transcript    │
                         └──────────────────┘
"""
import asyncio
import logging

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
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from backend_client import backend_client
from interviewer import InterviewerAgent

logger = logging.getLogger("agent")

load_dotenv(".env.local")


server = AgentServer()


def prewarm(proc: JobProcess):
    """
    Prewarm function - called when agent process starts.

    Loads heavy models (like VAD) once and reuses them across sessions.
    This reduces latency when a new interview starts.
    """
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


async def store_transcript_callback(interview_id: str, role: str, content: str):
    """
    Callback to store transcript messages to the backend.

    Called after each turn to persist the conversation for:
    - Chat history display
    - Final interview evaluation
    - Seamless text/voice mode switching
    """
    if interview_id and content:
        success = await backend_client.store_transcript(interview_id, role, content)
        if success:
            logger.info(f"Transcript stored: {role} - {content[:30]}...")
        else:
            logger.error(f"Failed to store transcript: {role}")


@server.rtc_session()
async def interview_agent(ctx: JobContext):
    """
    Main agent session handler.

    This function is called when a new room needs an agent.
    It sets up the interview-specific agent and voice pipeline.
    """
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    logger.info(f"Agent joining room: {ctx.room.name}")

    interview_id = backend_client.extract_interview_id(ctx.room.name)
    interview_context = None

    if interview_id:
        logger.info(f"Fetching context for interview: {interview_id}")
        interview_context = await backend_client.get_interview_context(interview_id)

        if interview_context:
            logger.info(
                f"Context loaded - Problem: {interview_context.get('problemSnapshot', {}).get('title', 'Unknown')}"
            )
        else:
            logger.warning(f"Failed to fetch context for interview: {interview_id}")
    else:
        logger.warning(f"Could not extract interview ID from room: {ctx.room.name}")

    session = AgentSession(
        stt=inference.STT(model="assemblyai/universal-streaming", language="en"),
        llm=inference.LLM(model="openai/gpt-4.1-mini"),
        tts=inference.TTS(
            model="cartesia/sonic-3",
            voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    background_tasks: set = set()

    @session.on("conversation_item_added")
    def on_conversation_item_added(event):
        """
        Store conversation items (both user and agent) to backend.
        This event fires when a message is committed to the chat history.
        """
        if not interview_id:
            return

        item = event.item
        if hasattr(item, "role") and hasattr(item, "content"):
            role = "user" if item.role == "user" else "assistant"
            content = ""

            if isinstance(item.content, str):
                content = item.content
            elif isinstance(item.content, list):
                for part in item.content:
                    if hasattr(part, "text"):
                        content += part.text
                    elif isinstance(part, str):
                        content += part

            if content:
                logger.info(f"Storing {role} transcript: {content[:50]}...")
                task = asyncio.create_task(
                    store_transcript_callback(interview_id, role, content)
                )
                background_tasks.add(task)
                task.add_done_callback(background_tasks.discard)

    interviewer = InterviewerAgent(
        interview_context=interview_context,
        interview_id=interview_id,
    )

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

    await ctx.connect()

    logger.info(f"Interview agent started for room: {ctx.room.name}")


if __name__ == "__main__":
    cli.run_app(server)

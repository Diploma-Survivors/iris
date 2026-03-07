"""
Interviewer Agent

PURPOSE:
The core voice AI agent that conducts coding interviews.
Prompts and LLM inference are now fully owned by sfinx-backend — this agent
handles only the LiveKit session layer (STT, TTS, turn-taking, tools).

HOW IT WORKS:
1. BackendLLM routes all inference through sfinx-backend's LangChain/Gemini pipeline
2. on_enter calls voice-start endpoint (via BackendLLM) to get the opening greeting
3. Subsequent turns call voice-message endpoint with the user's speech
4. Transcripts are stored in the DB by the backend endpoints

VOICE INTERVIEW FLOW:
┌─────────────────┐
│ User joins room │
└────────┬────────┘
         ▼
┌─────────────────────────────┐
│ Extract interview ID from   │
│ room name (interview-{uuid})│
└────────────┬────────────────┘
             ▼
┌─────────────────────────────┐
│ on_enter → BackendLLM calls │
│ voice-start → greeting TTS  │
└────────────┬────────────────┘
             ▼
┌─────────────────────────────┐
│ Each user turn → BackendLLM │
│ calls voice-message → TTS   │
└─────────────────────────────┘
"""

import logging
import time

from livekit.agents import Agent, RunContext, function_tool

logger = logging.getLogger("interviewer")


class InterviewerAgent(Agent):
    """
    Voice AI agent for conducting coding interviews.

    System prompt and LLM inference are delegated to sfinx-backend via BackendLLM.
    This agent owns the LiveKit session layer only.
    """

    def __init__(self, interview_id: str | None = None) -> None:
        self.interview_id = interview_id
        # Backend owns all prompts — empty instructions here
        super().__init__(instructions="")
        logger.info(
            f"InterviewerAgent initialized for interview: {interview_id or 'unknown'}"
        )

    async def on_enter(self) -> None:
        """Speak the initial greeting when the agent joins the session."""
        t0 = time.monotonic()
        await self.session.generate_reply(
            instructions="Greet the candidate warmly and ask if they are ready to begin the interview. Keep it brief and friendly — one or two sentences. Do NOT mention the problem yet."
        )
        elapsed = time.monotonic() - t0
        logger.info(f"[TIMING] on_enter generate_reply (LLM+TTS+audio): {elapsed:.2f}s")

    @function_tool
    async def provide_hint(
        self,
        context: RunContext,
        hint_level: str,
    ):
        """
        Provide a progressive hint to the candidate.

        Use this when the candidate is stuck and needs guidance.
        Choose the appropriate hint level based on how stuck they are.

        Args:
            hint_level: Level of hint - 'gentle' (nudge in right direction),
                       'moderate' (more specific guidance), or
                       'strong' (nearly giving away the approach)
        """
        logger.info(f"Providing {hint_level} hint for interview {self.interview_id}")

        hint_guidance = {
            "gentle": "Give a subtle nudge without revealing the solution approach",
            "moderate": "Provide more specific guidance about the approach to consider",
            "strong": "Give substantial help but still let them implement it",
        }

        guidance = hint_guidance.get(hint_level, hint_guidance["gentle"])
        return f"[Internal: {guidance}]"

    @function_tool
    async def summarize_progress(self, context: RunContext):
        """
        Summarize the candidate's progress so far.

        Use this to give the candidate feedback on what they've accomplished
        and what remains to be done.
        """
        logger.info(f"Summarizing progress for interview {self.interview_id}")
        return (
            "[Internal: Summarize what the candidate has accomplished and what's left]"
        )

    @function_tool
    async def request_code_review(
        self,
        context: RunContext,
        focus_area: str,
    ):
        """
        Ask the candidate to walk through their code.

        Use this to understand their implementation and evaluate code quality.

        Args:
            focus_area: Specific aspect to focus on - 'logic', 'edge_cases',
                       'time_complexity', 'space_complexity', or 'readability'
        """
        logger.info(
            f"Requesting code review for {focus_area} in interview {self.interview_id}"
        )
        return (
            f"[Internal: Ask candidate to explain their code focusing on {focus_area}]"
        )

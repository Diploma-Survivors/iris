"""
Interviewer Agent

PURPOSE:
The core voice AI agent that conducts coding interviews.
This replaces the generic assistant with an interview-specific agent that:
1. Understands the problem context
2. Guides candidates through coding problems
3. Provides hints without giving away solutions
4. Evaluates communication and problem-solving

HOW IT WORKS:
1. When user joins room, agent fetches interview context from backend
2. Agent is initialized with interview-specific system prompt
3. During conversation, transcripts are stored to backend
4. Agent uses tools to help with interview tasks

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
│ Fetch context from backend  │
│ (problem, existing messages)│
└────────────┬────────────────┘
             ▼
┌─────────────────────────────┐
│ Initialize agent with       │
│ interview system prompt     │
└────────────┬────────────────┘
             ▼
┌─────────────────────────────┐
│ Conduct voice interview     │
│ (STT → LLM → TTS)           │
└────────────┬────────────────┘
             ▼
┌─────────────────────────────┐
│ Store transcripts to backend│
└─────────────────────────────┘
"""
import logging

from livekit.agents import Agent, RunContext, function_tool

logger = logging.getLogger("interviewer")

# Default prompt used when backend context is unavailable
DEFAULT_INTERVIEW_PROMPT = """You are a senior software engineer conducting a coding interview.

Your role:
- Guide the candidate through the problem
- Ask clarifying questions to understand their approach
- Provide hints when they're stuck (but don't give away the solution)
- Evaluate their communication and problem-solving process

Rules:
- Be encouraging but professional
- Focus on understanding their thought process
- If they ask for help, give progressive hints
- Keep responses concise and conversational
- Remember this is a VOICE conversation, so speak naturally
- Avoid complex formatting, code blocks, or special characters
"""

VOICE_ADAPTATION_PROMPT = """
IMPORTANT - Voice Interview Guidelines:
- You are speaking, not writing. Keep responses SHORT and conversational.
- Avoid saying code syntax literally (don't say "curly brace" or "semicolon")
- When discussing code, describe the logic conceptually
- Use natural pauses and transitions
- If the candidate shares code, summarize what you see rather than reading it
- Ask one question at a time
- Be encouraging when they make progress
"""


class InterviewerAgent(Agent):
    """
    Voice AI agent for conducting coding interviews.

    This agent is customized for each interview session based on:
    - The specific problem being solved
    - Any existing conversation history (from text chat)
    - The candidate's progress so far
    """

    def __init__(
        self,
        interview_context: dict | None = None,
        interview_id: str | None = None,
    ) -> None:
        """
        Initialize the interviewer agent.

        Args:
            interview_context: Context fetched from backend containing:
                - systemPrompt: Pre-formatted interview prompt
                - problemSnapshot: Problem details
                - existingMessages: Previous chat history
            interview_id: UUID of the interview (for transcript storage)
        """
        self.interview_id = interview_id
        self.interview_context = interview_context

        if interview_context and "systemPrompt" in interview_context:
            base_prompt = interview_context["systemPrompt"]
            instructions = base_prompt + VOICE_ADAPTATION_PROMPT
        else:
            instructions = DEFAULT_INTERVIEW_PROMPT + VOICE_ADAPTATION_PROMPT

        super().__init__(instructions=instructions)

        logger.info(
            f"InterviewerAgent initialized for interview: {interview_id or 'unknown'}"
        )

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
        return "[Internal: Summarize what the candidate has accomplished and what's left]"

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
        return f"[Internal: Ask candidate to explain their code focusing on {focus_area}]"

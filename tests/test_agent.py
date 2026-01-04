"""
Interviewer Agent Tests

PURPOSE:
Tests for the voice interview agent behavior.
Uses LiveKit's testing framework to evaluate agent responses.

HOW IT WORKS:
- Creates a mock agent session without actual audio
- Simulates user input and evaluates agent responses
- Uses LLM-based judging for behavioral validation

TEST CATEGORIES:
1. Greeting behavior - Agent should greet professionally
2. Interview guidance - Agent should guide without giving answers
3. Hint progression - Agent should provide graduated hints
4. Edge cases - Handling unclear or off-topic input
"""
import pytest
from livekit.agents import AgentSession, inference, llm

from interviewer import InterviewerAgent


def _llm() -> llm.LLM:
    return inference.LLM(model="openai/gpt-4.1-mini")


# Sample problem context for testing
SAMPLE_PROBLEM_CONTEXT = {
    "systemPrompt": """You are a senior software engineer conducting a coding interview.

Problem Context:
{
    "title": "Two Sum",
    "description": "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.",
    "examples": [
        {"input": "nums = [2,7,11,15], target = 9", "output": "[0,1]"}
    ]
}

Rules:
- Be encouraging but professional
- Focus on understanding their thought process
- If they ask for help, give progressive hints
- Keep responses concise and conversational
""",
    "problemSnapshot": {
        "title": "Two Sum",
        "description": "Given an array of integers nums and an integer target...",
    },
}


@pytest.mark.asyncio
async def test_professional_greeting() -> None:
    """Agent should greet the candidate professionally for an interview."""
    async with (
        _llm() as llm_instance,
        AgentSession(llm=llm_instance) as session,
    ):
        agent = InterviewerAgent(
            interview_context=SAMPLE_PROBLEM_CONTEXT,
            interview_id="test-123",
        )
        await session.start(agent)

        result = await session.run(user_input="Hello, I'm ready for my interview")

        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(
                llm_instance,
                intent="""
                Greets the candidate professionally for a coding interview.

                The response should:
                - Be welcoming but professional (not overly casual)
                - Acknowledge this is an interview context
                - May briefly mention the problem or ask if they're ready

                The response should NOT:
                - Be overly casual or use slang
                - Immediately dive into the problem without greeting
                """,
            )
        )

        result.expect.no_more_events()


@pytest.mark.asyncio
async def test_guides_without_giving_answer() -> None:
    """Agent should guide the candidate without revealing the solution."""
    async with (
        _llm() as llm_instance,
        AgentSession(llm=llm_instance) as session,
    ):
        agent = InterviewerAgent(
            interview_context=SAMPLE_PROBLEM_CONTEXT,
            interview_id="test-123",
        )
        await session.start(agent)

        result = await session.run(
            user_input="I have no idea how to solve this. Can you just tell me the answer?"
        )

        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(
                llm_instance,
                intent="""
                Guides the candidate without giving away the answer.

                The response should:
                - Acknowledge they're stuck
                - Provide a hint or guiding question
                - Encourage them to think through the problem

                The response should NOT:
                - Give the complete solution
                - Provide working code
                - Tell them exactly which algorithm to use (e.g., "use a hashmap")
                """,
            )
        )

        result.expect.no_more_events()


@pytest.mark.asyncio
async def test_asks_clarifying_questions() -> None:
    """Agent should ask clarifying questions about the candidate's approach."""
    async with (
        _llm() as llm_instance,
        AgentSession(llm=llm_instance) as session,
    ):
        agent = InterviewerAgent(
            interview_context=SAMPLE_PROBLEM_CONTEXT,
            interview_id="test-123",
        )
        await session.start(agent)

        result = await session.run(
            user_input="I think I'll use a loop to solve this"
        )

        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(
                llm_instance,
                intent="""
                Asks a clarifying question or encourages elaboration.

                The response should:
                - Show interest in understanding their approach
                - Ask them to elaborate on their idea
                - OR ask about time/space complexity
                - OR ask about edge cases

                The response may:
                - Acknowledge their approach positively
                - Ask follow-up questions

                The response should NOT:
                - Immediately correct them
                - Give them a better approach without asking
                """,
            )
        )

        result.expect.no_more_events()


@pytest.mark.asyncio
async def test_handles_off_topic() -> None:
    """Agent should handle off-topic questions professionally."""
    async with (
        _llm() as llm_instance,
        AgentSession(llm=llm_instance) as session,
    ):
        agent = InterviewerAgent(
            interview_context=SAMPLE_PROBLEM_CONTEXT,
            interview_id="test-123",
        )
        await session.start(agent)

        result = await session.run(
            user_input="What's the weather like today?"
        )

        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(
                llm_instance,
                intent="""
                Handles off-topic question professionally and redirects.

                The response should:
                - Politely acknowledge or briefly respond
                - Redirect back to the interview/problem

                The response should NOT:
                - Engage in lengthy off-topic discussion
                - Be rude or dismissive
                """,
            )
        )

        result.expect.no_more_events()


@pytest.mark.asyncio
async def test_concise_voice_responses() -> None:
    """Agent should provide concise responses suitable for voice."""
    async with (
        _llm() as llm_instance,
        AgentSession(llm=llm_instance) as session,
    ):
        agent = InterviewerAgent(
            interview_context=SAMPLE_PROBLEM_CONTEXT,
            interview_id="test-123",
        )
        await session.start(agent)

        result = await session.run(
            user_input="Can you explain the problem to me?"
        )

        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(
                llm_instance,
                intent="""
                Provides a concise explanation suitable for voice conversation.

                The response should:
                - Be relatively brief (not a wall of text)
                - Be conversational in tone
                - Explain the problem clearly

                The response should NOT:
                - Include code blocks or complex formatting
                - Be excessively long (more than a few sentences)
                - Use bullet points or numbered lists
                """,
            )
        )

        result.expect.no_more_events()

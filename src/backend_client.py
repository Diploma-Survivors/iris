"""
Backend API Client

PURPOSE:
Provides a client for Iris to communicate with sfinx-backend.
This enables the voice agent to:
1. Fetch interview context (problem details, system prompt)
2. Store voice transcripts for evaluation
3. Sync voice conversations with chat history

HOW IT WORKS:
- Uses HTTP requests with API key authentication
- Extracts interview ID from LiveKit room name (format: interview-{uuid})
- Caches interview context to minimize API calls during conversation

AUTHENTICATION:
- Uses x-api-key header with INTERNAL_API_KEY
- Must match the INTERNAL_API_KEY configured in sfinx-backend
"""
import logging
import os
from typing import Any

import httpx

logger = logging.getLogger("backend_client")


class BackendClient:
    """Client for communicating with sfinx-backend API"""

    def __init__(self):
        self.base_url = os.getenv("BACKEND_URL", "http://localhost:3000")
        self.api_key = os.getenv("INTERNAL_API_KEY", "")
        self._context_cache: dict[str, dict] = {}

    def _headers(self) -> dict[str, str]:
        """Generate request headers with API key"""
        return {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
        }

    def extract_interview_id(self, room_name: str) -> str | None:
        """
        Extract interview ID from room name.
        Room names follow format: interview-{uuid}
        """
        if room_name.startswith("interview-"):
            return room_name.replace("interview-", "")
        return None

    async def get_interview_context(self, interview_id: str) -> dict[str, Any] | None:
        """
        Fetch interview context from backend.

        Returns:
            - problemSnapshot: Problem details (title, description, examples)
            - systemPrompt: Pre-formatted prompt for the interviewer
            - existingMessages: Previous chat history to maintain context
            - status: Current interview status

        The response is cached to avoid repeated API calls during a session.
        """
        if interview_id in self._context_cache:
            logger.info(f"Using cached context for interview: {interview_id}")
            return self._context_cache[interview_id]

        url = f"{self.base_url}/internal/ai-interviews/{interview_id}/context"
        logger.info(f"Fetching context from: {url}")

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    headers=self._headers(),
                    timeout=10.0,
                )

                logger.info(f"Context response status: {response.status_code}")

                if response.status_code == 200:
                    raw_response = response.json()
                    # Unwrap { data: {...} } wrapper from NestJS response
                    data = raw_response.get("data", raw_response)
                    if "error" not in data:
                        self._context_cache[interview_id] = data
                        logger.info(
                            f"Context loaded - Problem: {data.get('problemSnapshot', {}).get('title', 'Unknown')}"
                        )
                        return data
                    logger.error(f"Backend error: {data.get('error')}")
                else:
                    logger.error(
                        f"Failed to fetch context: {response.status_code} - {response.text}"
                    )

        except Exception as e:
            logger.exception(f"Error fetching interview context: {e}")

        return None

    async def store_transcript(
        self, interview_id: str, role: str, content: str
    ) -> bool:
        """
        Store a single transcript message to the backend.

        Args:
            interview_id: UUID of the interview
            role: 'user' or 'assistant'
            content: The transcribed text

        Returns:
            True if stored successfully, False otherwise

        This ensures voice conversations are persisted for:
        - Viewing in chat history
        - Final interview evaluation
        - Continuity if user switches to text mode
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/internal/ai-interviews/{interview_id}/transcript",
                    headers=self._headers(),
                    json={
                        "role": role,
                        "content": content,
                    },
                    timeout=10.0,
                )

                if response.status_code == 200 or response.status_code == 201:
                    return True
                else:
                    logger.error(
                        f"Failed to store transcript: {response.status_code} - {response.text}"
                    )

        except Exception as e:
            logger.exception(f"Error storing transcript: {e}")

        return False

    def clear_cache(self, interview_id: str | None = None):
        """Clear cached context (useful when interview ends)"""
        if interview_id:
            self._context_cache.pop(interview_id, None)
        else:
            self._context_cache.clear()


# Singleton instance
backend_client = BackendClient()

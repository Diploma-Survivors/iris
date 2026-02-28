# Iris - Real-time Voice Interview Agent

Iris is the real-time voice I/O layer of the Sfinx system. It acts as a senior software engineer conducting coding interviews through high-performance, low-latency voice interaction.

## 🌟 Overview

Iris is built on the **LiveKit Agents SDK** and integrates seamlessly with the broader Sfinx ecosystem. It transforms a standard coding problem into an interactive voice experience by handling speech-to-text, natural language processing, and text-to-speech in a unified pipeline.

## 🏗 System Architecture

The following diagram illustrates how Iris interacts with other components:

```mermaid
sequence_diagram
    participant U as User (Frontend)
    participant LK as LiveKit Cloud
    participant I as Iris Agent
    participant B as Sfinx Backend

    U->>B: Request Interview Session
    B->>U: Return LiveKit Token & Room ID (interview-{uuid})
    U->>LK: Join Room
    LK->>I: Dispatch Agent to Room
    I->>B: Fetch Interview Context (Problem, Prompt, History)
    B->>I: Return Context Data
    I->>U: "Hello! I'm your interviewer today..."
    Note over U,I: Voice Conversation Starts
    I->>B: Periodic Transcript Sync
```

## 🎙 Voice Pipeline

Iris uses a state-of-the-art voice pipeline optimized for low latency (< 500ms glass-to-glass):

| Component | Provider | Purpose |
| :--- | :--- | :--- |
| **VAD** | Silero | Detects when the user is speaking to trigger processing. |
| **STT** | AssemblyAI | Converts user audio streams into text in real-time. |
| **LLM** | OpenAI (GPT-4o-mini) | Reasons about the candidate's responses and generates guidance. |
| **TTS** | Cartesia (Sonic-3) | Generates high-quality, expressive human-like speech. |
| **Turn Detection** | Multilingual Model | Identifies natural pauses and end-of-turns. |
| **Noise Cancellation** | BVC (LiveKit) | Filters background noise for crystal clear audio. |

## 🧩 Key Modules

### 1. `src/agent.py` (Entry Point)
The core orchestrator that manages the LiveKit `AgentServer`.
- **Prewarming**: Loads heavy models (VAD) during process startup to eliminate cold-start latency.
- **Session Handling**: Extracts `interview_id` from room names and initializes the voice pipeline.
- **Transcript Management**: Captures conversation events and syncs them back to the backend.

### 2. `src/backend_client.py` (Integration)
A specialized HTTP client for communicating with the `sfinx-backend`.
- **Authentication**: Uses a secure `INTERNAL_API_KEY` shared with the backend.
- **Context Fetching**: Retrieves the problem snapshot, system prompts, and existing chat history.
- **Persistence**: Exports transcripts to ensure the voice interview is visible in the web UI.

### 3. `src/interviewer.py` (AI Logic)
Contains the `InterviewerAgent` class which defines the "personality" of the interviewer.
- **Voice Adaptation**: Modifies standard LLM prompts to be more suitable for spoken conversation (shorter, no markdown, natural transitions).
- **Function Tools**:
    - `provide_hint`: Delivers progressive hints (gentle/moderate/strong).
    - `summarize_progress`: Recaps what the candidate has achieved.
    - `request_code_review`: Prompts the candidate to explain specific parts of their code.

### 4. `tests/test_agent.py` (Validation)
A comprehensive test suite using the LiveKit testing framework.
- Uses **LLM-based judging** to verify agent behavior (e.g., "Was the greeting professional?", "Did the agent avoid giving the answer?").

## 🚀 Getting Started

### Prerequisites
- Python 3.13+
- `uv` package manager
- LiveKit Cloud account & API keys

### Installation
```bash
cd iris
uv sync
```

### Environment Configuration (`.env.local`)
```env
LIVEKIT_URL=...
LIVEKIT_API_KEY=...
LIVEKIT_API_SECRET=...
BACKEND_URL=http://localhost:3000
INTERNAL_API_KEY=your-shared-secret
OPENAI_API_KEY=...
CARTESIA_API_KEY=...
ASSEMBLYAI_API_KEY=...
```

### Running the Agent
```bash
# Start in development mode (auto-reload)
uv run python src/agent.py dev

# Run in console mode for testing without a frontend
uv run python src/agent.py console
```

### Running Tests
```bash
uv run pytest
```

## 🛠 Development Guidelines

1. **Latency is King**: Always prefer `uv run python src/agent.py download-files` to cache models locally.
2. **Conversation Flow**: Keep LLM responses short. Use the `VOICE_ADAPTATION_PROMPT` in `interviewer.py` to refine speech patterns.
3. **Consistency**: Ensure any transcript stored via Iris matches the schema expected by the `sfinx-backend`.

# Iris Voice Interview Tuning Guide

This guide covers how to improve the voice interview experience by tuning models, turn detection, and other AgentSession options.

---

## Current Configuration

| Component | Current Setting | Purpose |
|-----------|-----------------|---------|
| **STT** | `deepgram/nova-3` + keyterms | Speech-to-text (technical vocabulary for coding interviews) |
| **LLM** | `openai/gpt-4.1-mini` | Response generation |
| **TTS** | `deepgram` aura-2-thalia-en | Text-to-speech |
| **Turn detection** | `MultilingualModel()` | End-of-turn detection |
| **VAD** | Silero (prewarmed) | Voice activity detection |
| **preemptive_generation** | `False` | Wait for user to finish before generating (reduces interruptions) |

---

## 1. Model Alternatives

### STT (Speech-to-Text)

| Model | Latency | Accuracy | Use Case |
|-------|---------|----------|----------|
| `deepgram.STT(nova-3, keyterm=[...])` | **Lower** | **Excellent** | **Current** — keyterms fix technical terms (pointer, node, etc.) |
| `openai/gpt-4o-transcribe` | Medium | Good | No vocabulary hints; mishears "pointer"→"minot" |
| `deepgram/nova-3` (no keyterms) | Lower | Excellent | General speech |
| `assemblyai/universal-streaming` | Medium | Good | Streaming, English only |

**Current:** Deepgram Nova-3 with `keyterm` — improves recognition of coding vocabulary (pointer, node, linked list, algorithm, recursion, etc.).

```python
# Deepgram STT with keyterm prompting (Nova-3)
from livekit.plugins import deepgram
stt=deepgram.STT(
    model="nova-3",
    language="en-US",
    keyterm=["pointer", "node", "linked list", "algorithm", "recursion", ...],
),
```

### LLM (Language Model)

| Model | Latency | Quality | Cost |
|-------|---------|---------|------|
| `openai/gpt-4.1-mini` | Fast | Good | Low |
| `openai/gpt-4.1-nano` | **Fastest** | Adequate | Lowest |
| `openai/gpt-4.1` | Slower | Best | High |
| `openai/gpt-4o-mini` | Fast | Good | Low |

**Recommendation:** For interviews, `gpt-4.1-mini` is a good balance. Use `gpt-4.1-nano` if you need faster replies and can accept slightly lower quality.

### TTS (Text-to-Speech)

| Model | Latency | Quality | Notes |
|-------|---------|---------|-------|
| `cartesia/sonic-3` | Low | Excellent | Current |
| `cartesia/sonic-2` | Lower | Good | Slightly faster |
| `deepgram/aura-2` | Low | Good | Alternative |
| `elevenlabs/eleven_turbo_v2_5` | Medium | Very natural | Higher cost |

**Suggested voices (Cartesia):**
- `9626c31c-bec5-4cca-baa8-f8ba9e84c8bc` — Jacqueline (current, confident female)
- `a167e0f3-df7e-4d52-a9c3-f949145efdab` — Blake (energetic male)
- `f31cc6a7-c1e8-4764-980c-60a361443dd1` — Robyn (neutral Australian female)

---

## 2. Turn Detection Tuning

Controls when the agent decides the user has finished speaking.

| Parameter | Default | Effect | Tuning |
|-----------|---------|--------|--------|
| `min_endpointing_delay` | 0.5s | Min wait after last speech before responding | **Lower (0.3)** = faster but may cut off pauses |
| `max_endpointing_delay` | 3.0s | Max wait when turn detector says "user will continue" | **Lower (2.0)** = faster for slow thinkers |

**For snappier interviews** (user speaks in short bursts):
```python
session = AgentSession(
    stt=...,
    llm=...,
    tts=...,
    turn_detection=MultilingualModel(),
    vad=ctx.proc.userdata["vad"],
    preemptive_generation=True,
    min_endpointing_delay=0.3,   # Respond faster
    max_endpointing_delay=2.0,   # Don't wait too long
)
```

**For thoughtful interviews** (user needs time to think):
```python
session = AgentSession(
    ...
    min_endpointing_delay=0.6,   # Give user more time
    max_endpointing_delay=4.0,   # Allow longer pauses
)
```

---

## 3. Interruption & Barge-in

| Parameter | Default | Effect |
|-----------|---------|--------|
| `allow_interruptions` | True | User can interrupt when agent is speaking |
| `min_interruption_duration` | 0.5s | Min speech duration to trigger interrupt |
| `false_interruption_timeout` | 2.0s | Time before resuming if interrupt was accidental |
| `resume_false_interruption` | True | Resume agent speech after false interrupt |

**For interview flow:** Keep `allow_interruptions=True` so the candidate can interrupt to clarify.

---

## 4. VAD Tuning (Silero)

**Current config (reduces interruptions):**

```python
def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load(
        min_speech_duration=0.1,   # Min speech to register (filter brief noise)
        min_silence_duration=1.3,  # Longer pause before end-of-turn (reduces interruptions)
    )
```

- **min_silence_duration** (default 0.55s): How long silence must last before the system treats it as end-of-turn. **Higher = fewer interruptions** but slightly slower turn-taking.
- **min_speech_duration** (default 0.05s): Minimum speech duration to start a chunk. Slightly higher (0.1) filters brief noise.

---

## 5. Preemptive Generation

`preemptive_generation=False` (current) — agent waits for the user to fully finish before generating. Reduces cut-offs and fragmented responses.

**Trade-off:** Set `True` for snappier feel; use `False` if you get too many interruptions.

---

## 6. Endpointing (Turn Detection Timing)

| Parameter | Default | Current | Effect |
|-----------|---------|---------|--------|
| `min_endpointing_delay` | 0.5s | 1.2s | Wait longer after user stops before responding (reduces interruptions) |
| `max_endpointing_delay` | 3.0s | 5.0s | Allow longer pauses when user might continue |

---

## 7. Quick Wins (Low Effort)

1. **Pin LiveKit region** — Use Singapore URL in `.env` for consistent latency
2. **If too slow:** Reduce `min_endpointing_delay` to 0.6s (trade-off: more interruptions)
3. **Try Deepgram STT** — Often lower latency than AssemblyAI
4. **Run in production mode** — `uv run python src/agent.py start` for prewarmed workers
5. **Connect before session** — Call `ctx.connect()` before `session.start()` (see SYSTEM_FLOW_ARCHITECTURE.md)

---

## 8. Environment Variables

Ensure these are set for optimal performance:

```env
# Iris .env.local
LIVEKIT_URL=https://thesis-5fn0o0su.osingapore1b.production.livekit.cloud  # Pin to Singapore
BACKEND_URL=http://localhost:3000  # Or your backend URL

# For Deepgram STT (if switching)
DEEPGRAM_API_KEY=...

# For AssemblyAI (current)
ASSEMBLYAI_API_KEY=...
```

---

## 9. Model Reference (LiveKit Inference)

Full list: [LiveKit Models](https://docs.livekit.io/agents/models/)

- **STT:** `assemblyai/universal-streaming`, `deepgram/nova-3`, `deepgram/nova-2`
- **LLM:** `openai/gpt-4.1-mini`, `openai/gpt-4.1-nano`, `openai/gpt-4o-mini`
- **TTS:** `deepgram` plugin: `aura-2-thalia-en` (current), `aura-2-asteria-en`, etc. | Inference: `cartesia/sonic-3`, `deepgram/aura-2`, `elevenlabs/eleven_turbo_v2_5`

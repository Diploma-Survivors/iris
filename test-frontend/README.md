# Voice Interview Test Frontend

Simple HTML page to test the LiveKit voice interview feature.

## Prerequisites

1. **Backend running** with LiveKit credentials configured:
   ```bash
   cd sfinx-backend
   npm run start:dev
   ```

2. **Iris agent running**:
   ```bash
   cd iris
   uv run python src/agent.py dev
   ```

3. **Active interview session** - You need an existing interview ID from the chat mode.

## Usage

1. Open `index.html` in a browser (or serve it):
   ```bash
   # Simple HTTP server
   python -m http.server 8080
   # Then open http://localhost:8080
   ```

2. Fill in the configuration:
   - **Backend URL**: Your sfinx-backend URL (default: `http://localhost:3000`)
   - **Interview ID**: UUID of an active interview session
   - **JWT Token**: Your authentication token (get from login)

3. Click **Connect** to join the voice interview

4. Speak to the AI interviewer!

## Getting Test Values

### Interview ID
Create an interview via API:
```bash
curl -X POST http://localhost:3000/ai-interviews \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"problemId": 1}'
```

### JWT Token
Login and extract the token from the response.

## Troubleshooting

- **"Token request failed"**: Check JWT token is valid and interview exists
- **"Agent not connecting"**: Ensure Iris agent is running with `uv run python src/agent.py dev`
- **No audio**: Check browser microphone permissions
- **CORS errors**: Ensure backend allows your origin

## Architecture

```
┌─────────────────┐     1. Get Token      ┌─────────────────┐
│   This Page     │ ───────────────────►  │  sfinx-backend  │
│   (Browser)     │ ◄───────────────────  │  /livekit/token │
└────────┬────────┘     2. Token          └─────────────────┘
         │
         │ 3. Connect with Token
         ▼
┌─────────────────┐                       ┌─────────────────┐
│  LiveKit Cloud  │ ◄──────────────────►  │   Iris Agent    │
│     (Room)      │    WebRTC Audio       │  (Python)       │
└─────────────────┘                       └─────────────────┘
```

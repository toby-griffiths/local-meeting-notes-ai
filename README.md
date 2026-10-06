# Notepad — Local AI Meeting Notes

A privacy-first, fully local alternative to tools like Granola or Otter.ai. It listens to your meetings (mic + your computer's system audio), transcribes them on your own machine, and turns the raw transcript into clean, structured notes — all without sending anything to the cloud.

## What it does

- **Live meeting capture** — hit "Start Meeting" when your call begins (Zoom, Meet, Teams, anything), and it records both your microphone and your computer's system audio (the other participants) together, using Windows WASAPI loopback capture. No bot joins your call, no visible recording indicator to other participants — it works quietly in the background, the same way Granola itself works.
- **Local transcription** — powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper) running entirely on your CPU. No audio ever leaves your machine.
- **Local AI note structuring** — a locally-run LLM via [Ollama](https://ollama.com) turns the raw transcript into a Summary, Key Decisions, Action Items, and Follow-up Tasks.
- **Meeting history** — every meeting is saved locally and browsable from a sidebar.
- **Ask your meeting anything** — a simple Q&A feature lets you ask questions about a specific meeting's transcript and get answers grounded only in what was actually said.
- **Fallback file upload** — if you'd rather transcribe a pre-recorded audio file instead of live-capturing, that works too.

## Tech stack

- **Backend:** FastAPI (Python)
- **Transcription:** faster-whisper (`small` model)
- **Note structuring / Q&A:** Ollama, running `llama3.2:3b` locally
- **Audio capture:** PyAudioWPatch (WASAPI loopback support on Windows)
- **Frontend:** Plain HTML/CSS/JS, no build step required

## Requirements

- Windows (loopback audio capture is Windows-specific in this version)
- Python 3.10+
- [Ollama](https://ollama.com) installed and running locally

## Setup

1. **Clone the repo**
```bash
   git clone <your-repo-url>
   cd granola-clone
```

2. **Create and activate a virtual environment**
```bash
   python -m venv venv
   venv\Scripts\activate
```

3. **Install dependencies**
```bash
   pip install fastapi uvicorn python-dotenv requests faster-whisper PyAudioWPatch numpy scipy
```

4. **Install Ollama and pull the model**
   Download Ollama from [ollama.com](https://ollama.com), then:
```bash
   ollama pull llama3.2:3b
```

5. **Set up environment variables**
   Create a `.env` file inside `app/` with a long random API token, for example:
```bash
   python -c "import secrets; print('API_TOKEN=' + secrets.token_urlsafe(32))" > app/.env
```
   Every API request must send `Authorization: Bearer <API_TOKEN>`. If `API_TOKEN` is not set, a temporary token is generated on each start and printed in the logs.

   Optional: `ALLOWED_HOSTS` (comma-separated, default `localhost,127.0.0.1`) limits which `Host` headers are accepted.

6. **Run the backend**
```bash
   cd app
   uvicorn main:app --host 127.0.0.1
```

7. **Open the frontend**
   The backend serves the UI. Open the URL printed at startup (`http://localhost:8000/#token=...`). The token is saved in your browser and removed from the address bar. Don't open `index.html` from disk; it won't be able to reach the API.

## How to use it

1. Join your call as usual, in whatever app you already use (Zoom, Meet, Teams).
2. Switch to the Notepad tab and click **Start Meeting**.
3. Talk normally — nothing to babysit.
4. When the call ends, click **Stop & Process**. Notepad transcribes the recording and generates structured notes automatically.
5. Browse past meetings from the sidebar, or use the **Ask** tab to ask questions about any meeting's transcript.

## Known limitations

- **Windows only** — the system audio loopback capture relies on WASAPI, which is Windows-specific. It won't run as-is on macOS or Linux.
- **Transcription accuracy** — using the `small` Whisper model with beam search and VAD filtering for a balance of speed and accuracy on CPU. Accuracy on noisy or heavily overlapping conversation is still a work in progress.
- **No delete option yet** — meeting history is stored in `meetings.json` inside `app/`; removing old meetings currently means editing that file directly.
- **Long meetings aren't optimized** — very long transcripts may hit the local LLM's context window or processing time limits. Works well for typical meeting lengths (15-30+ min); not yet built for multi-hour sessions.
- **Single-user, local-only** — this isn't designed for multi-device sync or shared team access; it's a personal, on-device tool.

## Roadmap

- Live streaming transcript (see it update in real time during the meeting, not just after)
- Semantic search across all past meetings (ask "what did we decide about X" across your whole history, not just one meeting)
- Delete/manage meeting history from the UI

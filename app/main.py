"""
Granola-style local AI meeting notes app — backend.
"""

import requests
import math
import secrets
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware
from faster_whisper import WhisperModel
import tempfile
import shutil
import os
import json
import uuid
import threading
from datetime import datetime

# Safe import — MeetingRecorder needs PyAudioWPatch, which is Windows-only.
try:
    from audio_capture import MeetingRecorder
    recorder = MeetingRecorder()
    LIVE_RECORDING_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    recorder = None
    LIVE_RECORDING_AVAILABLE = False
    print(f"WARNING: Live meeting recording disabled — {e}")

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_EMBED_URL = OLLAMA_URL.replace("/api/generate", "/api/embeddings")
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")
# Largest request body accepted (uploads included), in MB.
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "500")) * 1024 * 1024
EMBEDDING_MODEL = "nomic-embed-text"

# --- Access control -------------------------------------------------------
# Every API call must send `Authorization: Bearer <API_TOKEN>`. If no token is
# configured, a random one is generated for this run (and printed below) so the
# API is never left open.
API_TOKEN = os.getenv("API_TOKEN", "").strip()
_API_TOKEN_GENERATED = not API_TOKEN
if _API_TOKEN_GENERATED:
    API_TOKEN = secrets.token_urlsafe(32)

# Host headers we accept. Blocks DNS-rebinding attacks from malicious websites.
ALLOWED_HOSTS = [
    h.strip() for h in os.getenv("ALLOWED_HOSTS", "localhost,127.0.0.1").split(",") if h.strip()
]

# The frontend is served by this backend (same origin), so no CORS is needed.
UI_FILE = Path(os.getenv("UI_FILE", Path(__file__).resolve().parent.parent / "index.html"))
PUBLIC_GET_PATHS = {"/"}  # only the UI page itself is served without a token

# --- Storage --------------------------------------------------------------
# Transcripts, notes and temporary audio all live under DATA_DIR (default
# app/data/), created owner-only (0700). Keep it on an encrypted disk
# (FileVault / BitLocker); the data itself is not encrypted by the app.
DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).resolve().parent / "data"))
HISTORY_FILE = DATA_DIR / "meetings.json"
TMP_DIR = DATA_DIR / "tmp"
_history_lock = threading.Lock()


def _init_storage():
    for d in (DATA_DIR, TMP_DIR):
        d.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(d, 0o700)
    if HISTORY_FILE.exists():
        os.chmod(HISTORY_FILE, 0o600)
    # Recordings/uploads are written here instead of the system temp dir, so
    # they stay in the protected data dir. Anything left over from a crash is
    # raw meeting audio, so remove it on startup.
    for leftover in TMP_DIR.iterdir():
        if leftover.is_file():
            leftover.unlink()
    tempfile.tempdir = str(TMP_DIR)


_init_storage()

TRANSCRIBE_OPTIONS = dict(
    language="en",
    beam_size=5,
    vad_filter=True,
    vad_parameters=dict(min_silence_duration_ms=500),
)


def load_history():
    with _history_lock:
        if not HISTORY_FILE.exists():
            return []
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)


def save_history(meetings):
    """Atomic, owner-only write: a crash mid-write can't corrupt the history."""
    with _history_lock:
        tmp_path = HISTORY_FILE.with_name(HISTORY_FILE.name + ".tmp")
        fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(meetings, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, HISTORY_FILE)
        except BaseException:
            if tmp_path.exists():
                tmp_path.unlink()
            raise


def get_embedding(text: str) -> list:
    """Converts text into an embedding using a local Ollama embedding model."""
    try:
        response = requests.post(
            OLLAMA_EMBED_URL,
            json={"model": EMBEDDING_MODEL, "prompt": text},
            timeout=60,
        )
        response.raise_for_status()
        return response.json().get("embedding", [])
    except Exception as e:
        print(f"[embedding] Failed to generate embedding: {e}")
        return []


def cosine_similarity(vec1: list, vec2: list) -> float:
    """Measures similarity between two embeddings (0 = unrelated, 1 = identical meaning)."""
    if not vec1 or not vec2:
        return 0.0
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    return dot_product / (magnitude1 * magnitude2)


def extract_action_items(structured_notes: str) -> list:
    """Shared helper: pulls just the action items out of a meeting's structured notes."""
    extract_prompt = f"""Extract ONLY the action items listed below, as a simple list, one per line.
If there are none or it says "None noted", return exactly: NONE

{structured_notes}
"""
    response = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": extract_prompt, "stream": False},
        timeout=60,
    )
    response.raise_for_status()
    text = response.json().get("response", "").strip()

    if text.upper() == "NONE" or not text:
        return []
    return [line.strip("- ").strip() for line in text.split("\n") if line.strip()]


app = FastAPI(title="Local Meeting Notes AI")


class APITokenMiddleware:
    """Plain ASGI middleware: rejects any HTTP request without a valid bearer token,
    except GET on PUBLIC_GET_PATHS (the UI page)."""

    def __init__(self, app, token: str, public_get_paths=frozenset()):
        self.app = app
        self.token = token.encode()
        self.public_get_paths = public_get_paths

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        if scope["method"] == "GET" and scope["path"] in self.public_get_paths:
            return await self.app(scope, receive, send)
        auth = b""
        for name, value in scope.get("headers", []):
            if name.lower() == b"authorization":
                auth = value
                break
        scheme, _, token = auth.partition(b" ")
        if scheme.lower() == b"bearer" and secrets.compare_digest(token.strip(), self.token):
            return await self.app(scope, receive, send)
        body = b'{"detail":"Missing or invalid API token"}'
        await send({
            "type": "http.response.start",
            "status": 401,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
                (b"www-authenticate", b"Bearer"),
            ],
        })
        await send({"type": "http.response.body", "body": body})


class RequestTooLarge(HTTPException):
    def __init__(self):
        super().__init__(status_code=413, detail="Request body too large")


class BodySizeLimitMiddleware:
    """Plain ASGI middleware: rejects bodies over max_bytes, using
    Content-Length up front and counting bytes as they stream in (covers
    chunked uploads with no Content-Length)."""

    def __init__(self, app, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def _reject(self, send):
        body = b'{"detail":"Request body too large"}'
        await send({
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        })
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        for name, value in scope.get("headers", []):
            if name.lower() == b"content-length":
                try:
                    too_big = int(value) > self.max_bytes
                except ValueError:
                    too_big = True
                if too_big:
                    return await self._reject(send)

        received = 0
        response_started = False

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise RequestTooLarge()
            return message

        async def tracking_send(message):
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except RequestTooLarge:
            if response_started:
                raise
            await self._reject(send)


# Inner to the token check, so unauthenticated requests are rejected first.
app.add_middleware(BodySizeLimitMiddleware, max_bytes=MAX_UPLOAD_BYTES)
app.add_middleware(APITokenMiddleware, token=API_TOKEN, public_get_paths=PUBLIC_GET_PATHS)
# Added last so it runs first: reject unexpected Host headers before anything else.
app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)

if _API_TOKEN_GENERATED:
    print("API_TOKEN not set; generated a temporary token for this run.")
    print("Set API_TOKEN in app/.env (or your environment) to keep it stable across restarts.")
print(f"Open the UI at: http://localhost:8000/#token={API_TOKEN}")

print(f"Loading Whisper model ({WHISPER_MODEL_SIZE})...")
whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
print("Whisper model loaded.")

recording_active = False


@app.get("/", include_in_schema=False)
def ui():
    return FileResponse(
        UI_FILE,
        media_type="text/html",
        headers={
            "Cache-Control": "no-store",
            "X-Frame-Options": "DENY",
            "Content-Security-Policy": "frame-ancestors 'none'",
            "Referrer-Policy": "no-referrer",
        },
    )


@app.get("/health")
def health_check():
    return {
        "status": "running",
        "ollama_model": OLLAMA_MODEL,
        "whisper_model": WHISPER_MODEL_SIZE,
        "live_recording_available": LIVE_RECORDING_AVAILABLE,
    }


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    # Stream to disk in 1 MB chunks instead of loading the whole file into memory.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        shutil.copyfileobj(file.file, tmp, 1024 * 1024)
        tmp_path = tmp.name

    try:
        segments, info = whisper_model.transcribe(tmp_path, **TRANSCRIBE_OPTIONS)
        full_text = " ".join(segment.text.strip() for segment in segments)
        return {"transcript": full_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    finally:
        os.remove(tmp_path)


@app.post("/structure-notes")
async def structure_notes(payload: dict):
    raw_text = payload.get("text", "")
    if not raw_text.strip():
        raise HTTPException(status_code=400, detail="No text provided")

    prompt = f"""You are a meeting notes assistant. Below is a raw, messy transcript from a meeting.
Turn it into clean, structured notes with exactly this format:

SUMMARY:
(2-4 sentences summarizing what the meeting was about)

KEY DECISIONS:
- (concrete decisions that were made or agreed on during the meeting)

ACTION ITEMS:
- (specific tasks assigned, with who's responsible if mentioned)

FOLLOW-UP TASKS:
- (things that need to happen next but weren't fully assigned, or need checking later)

If a section has nothing relevant, write "None noted" under it instead of leaving it blank.

Raw transcript:
\"\"\"
{raw_text}
\"\"\"
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
        result = response.json()
        return {"structured_notes": result.get("response", "").strip()}
    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Could not connect to Ollama. Make sure it's running (`ollama serve`).",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Note structuring failed: {str(e)}")


@app.post("/process-meeting")
async def process_meeting(file: UploadFile = File(...)):
    transcript_result = await transcribe_audio(file)
    notes_result = await structure_notes({"text": transcript_result["transcript"]})

    meetings = load_history()
    meeting_record = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "transcript": transcript_result["transcript"],
        "structured_notes": notes_result["structured_notes"],
        "embedding": get_embedding(transcript_result["transcript"]),
    }
    meetings.insert(0, meeting_record)
    save_history(meetings)

    return meeting_record


@app.post("/start-meeting")
def start_meeting():
    global recording_active
    if not LIVE_RECORDING_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Live recording isn't available in this environment (requires Windows). Use /process-meeting with an uploaded file instead.",
        )
    if recording_active:
        raise HTTPException(status_code=400, detail="A meeting is already being recorded.")
    recorder.start()
    recording_active = True
    return {"status": "recording_started"}


@app.post("/stop-meeting")
async def stop_meeting():
    global recording_active
    if not LIVE_RECORDING_AVAILABLE:
        raise HTTPException(status_code=501, detail="Live recording isn't available in this environment.")
    if not recording_active:
        raise HTTPException(status_code=400, detail="No meeting is currently being recorded.")

    wav_path = recorder.stop()
    recording_active = False

    try:
        segments, info = whisper_model.transcribe(wav_path, **TRANSCRIBE_OPTIONS)
        full_text = " ".join(segment.text.strip() for segment in segments)
        notes_result = await structure_notes({"text": full_text})

        meetings = load_history()
        meeting_record = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "transcript": full_text,
            "structured_notes": notes_result["structured_notes"],
            "embedding": get_embedding(full_text),
        }
        meetings.insert(0, meeting_record)
        save_history(meetings)

        return meeting_record
    finally:
        os.remove(wav_path)


@app.get("/meetings")
def list_meetings():
    meetings = load_history()
    return [
        {
            "id": m["id"],
            "timestamp": m["timestamp"],
            "preview": m["structured_notes"][:80] + "...",
        }
        for m in meetings
    ]


@app.get("/meetings/{meeting_id}")
def get_meeting(meeting_id: str):
    meetings = load_history()
    for m in meetings:
        if m["id"] == meeting_id:
            return m
    raise HTTPException(status_code=404, detail="Meeting not found")


@app.delete("/meetings/{meeting_id}")
def delete_meeting(meeting_id: str):
    meetings = load_history()
    remaining = [m for m in meetings if m["id"] != meeting_id]
    if len(remaining) == len(meetings):
        raise HTTPException(status_code=404, detail="Meeting not found")
    save_history(remaining)
    return {"status": "deleted", "id": meeting_id}


@app.post("/search-meetings")
async def search_meetings(payload: dict):
    """Semantic search (RAG) across ALL past meetings."""
    question = payload.get("question", "")
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question is required")

    meetings = load_history()
    if not meetings:
        return {"answer": "No meetings recorded yet.", "sources": []}

    question_embedding = get_embedding(question)

    scored = []
    for m in meetings:
        meeting_embedding = m.get("embedding", [])
        score = cosine_similarity(question_embedding, meeting_embedding)
        scored.append((score, m))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_matches = [m for score, m in scored[:3] if score > 0]

    if not top_matches:
        return {"answer": "Couldn't find any relevant past meetings for that question.", "sources": []}

    context = "\n\n---\n\n".join(
        f"Meeting from {m['timestamp']}:\n{m['transcript']}" for m in top_matches
    )

    prompt = f"""You are answering a question using notes from the most relevant past meetings below.
If the answer isn't in these meetings, say so clearly — do not make anything up.

{context}

Question: {question}

Answer concisely, and mention which meeting(s) the answer came from:"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
        result = response.json()
        return {
            "answer": result.get("response", "").strip(),
            "sources": [{"id": m["id"], "timestamp": m["timestamp"]} for m in top_matches],
        }
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail="Could not connect to Ollama.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/meeting-followup/{meeting_id}")
async def generate_followup(meeting_id: str):
    """
    Agentic follow-up: reviews each action item, cross-references it against
    past meetings via RAG, classifies it (new/recurring), and drafts a
    prioritized follow-up summary.
    """
    meetings = load_history()
    current_meeting = next((m for m in meetings if m["id"] == meeting_id), None)
    if not current_meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")

    action_items = extract_action_items(current_meeting["structured_notes"])
    if not action_items:
        return {"meeting_id": meeting_id, "followup": "No action items to follow up on.", "items_reviewed": []}

    other_meetings = [m for m in meetings if m["id"] != meeting_id and m.get("embedding")]
    reviewed_items = []

    for item in action_items:
        item_embedding = get_embedding(item)
        best_match = None
        best_score = 0.0

        for past_meeting in other_meetings:
            score = cosine_similarity(item_embedding, past_meeting.get("embedding", []))
            if score > best_score:
                best_score = score
                best_match = past_meeting

        if best_match and best_score > 0.75:
            status = "recurring"
            note = f"Similar item found in meeting from {best_match['timestamp']}"
        else:
            status = "new"
            note = "No similar item found in past meetings"

        reviewed_items.append({"item": item, "status": status, "note": note})

    items_summary = "\n".join(
        f"- [{r['status'].upper()}] {r['item']} ({r['note']})" for r in reviewed_items
    )
    synthesis_prompt = f"""You are a helpful assistant drafting a follow-up message for a team, based on this action item review:

{items_summary}

Write a short, prioritized follow-up message. Flag "recurring" items as needing attention since they've come up before without resolution. Keep it concise and actionable."""

    synthesis_response = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": synthesis_prompt, "stream": False},
        timeout=60,
    )
    synthesis_response.raise_for_status()
    followup_text = synthesis_response.json().get("response", "").strip()

    return {
        "meeting_id": meeting_id,
        "followup": followup_text,
        "items_reviewed": reviewed_items,
    }


@app.post("/generate-actions/{meeting_id}")
async def generate_actions(meeting_id: str):
    """
    Agentic action-execution layer: classifies each action item by the
    real-world action it requires (email/calendar/task) and drafts the
    actual deliverable content, ready for human approval.
    """
    meetings = load_history()
    current_meeting = next((m for m in meetings if m["id"] == meeting_id), None)
    if not current_meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")

    action_items = extract_action_items(current_meeting["structured_notes"])
    if not action_items:
        return {"meeting_id": meeting_id, "drafted_actions": []}

    drafted_actions = []

    for item in action_items:
        draft_prompt = f"""Analyze this action item from a meeting and respond in EXACTLY this format:

TYPE: (one of: email, calendar_event, task, none)
TITLE: (short title, one line)
CONTENT: (the content, can span multiple lines)

Rules for CONTENT:
- if TYPE is email: write a short professional email body (2-4 sentences)
- if TYPE is calendar_event: write a one-line event description
- if TYPE is task: write a clear one-line task description
- if TYPE is none: write "N/A"

Do not add any text before TYPE: or after the CONTENT section.

Action item: "{item}"
"""
        draft_response = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": draft_prompt, "stream": False},
            timeout=60,
        )
        draft_response.raise_for_status()
        raw = draft_response.json().get("response", "").strip()

        # Multi-line-safe parsing: CONTENT can span several lines, so we
        # track which field we're currently inside rather than only
        # reading whatever's on the same line as the "CONTENT:" label.
        action_type, title = "task", item
        content_lines = []
        current_field = None

        for line in raw.split("\n"):
            stripped = line.strip()
            upper = stripped.upper()
            if upper.startswith("TYPE:"):
                action_type = stripped.split(":", 1)[1].strip().lower()
                current_field = None
            elif upper.startswith("TITLE:"):
                title = stripped.split(":", 1)[1].strip()
                current_field = None
            elif upper.startswith("CONTENT:"):
                first_bit = stripped.split(":", 1)[1].strip()
                if first_bit:
                    content_lines.append(first_bit)
                current_field = "content"
            elif current_field == "content" and stripped:
                content_lines.append(stripped)

        content = "\n".join(content_lines).strip() or item

        drafted_actions.append({
            "original_item": item,
            "type": action_type,
            "title": title,
            "content": content,
        })

    return {"meeting_id": meeting_id, "drafted_actions": drafted_actions}


@app.post("/ask")
async def ask_question(payload: dict):
    transcript = payload.get("transcript", "")
    question = payload.get("question", "")

    if not transcript.strip() or not question.strip():
        raise HTTPException(status_code=400, detail="Both transcript and question are required")

    prompt = f"""You are answering questions about a meeting, based only on the transcript below.
If the answer isn't in the transcript, say clearly that it wasn't discussed — do not make anything up.

Transcript:
\"\"\"
{transcript}
\"\"\"

Question: {question}

Answer concisely, in 1-3 sentences:"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
        result = response.json()
        return {"answer": result.get("response", "").strip()}
    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Could not connect to Ollama. Make sure it's running.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Question answering failed: {str(e)}")
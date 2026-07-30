"""
Granola-style local AI meeting notes app — backend.

Flow:
1. Audio file comes in -> Whisper transcribes it locally (no cloud).
2. Raw transcript text -> sent to a local Ollama model -> clean structured notes.
3. Meetings are saved to local history. /ask lets you chat with a transcript.
4. /start-meeting and /stop-meeting record mic + system audio (loopback) live,
   so you don't need to upload a file — just hit start when your call begins.

Run locally with: uvicorn main:app --reload
Requires Ollama running locally with a model pulled (e.g. `ollama pull llama3.2:3b`)
"""

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
from audio_capture import MeetingRecorder
import tempfile
import os
import json
import uuid
from datetime import datetime

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")

HISTORY_FILE = "meetings.json"

# Shared Whisper transcription settings — tuned for accuracy over speed:
# - language="en": skip language auto-detection guesswork
# - beam_size=5: consider more candidate transcriptions per segment before picking the best
# - vad_filter: strip silence/non-speech so Whisper doesn't "hallucinate" words in quiet gaps
TRANSCRIBE_OPTIONS = dict(
    language="en",
    beam_size=5,
    vad_filter=True,
    vad_parameters=dict(min_silence_duration_ms=500),
)


def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_history(meetings):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(meetings, f, indent=2)


app = FastAPI(title="Local Meeting Notes AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print(f"Loading Whisper model ({WHISPER_MODEL_SIZE})...")
whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
print("Whisper model loaded.")

recorder = MeetingRecorder()
recording_active = False


@app.get("/")
def health_check():
    return {"status": "running", "ollama_model": OLLAMA_MODEL, "whisper_model": WHISPER_MODEL_SIZE}


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        contents = await file.read()
        tmp.write(contents)
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
    }
    meetings.insert(0, meeting_record)
    save_history(meetings)

    return meeting_record


@app.post("/start-meeting")
def start_meeting():
    global recording_active
    if recording_active:
        raise HTTPException(status_code=400, detail="A meeting is already being recorded.")
    recorder.start()
    recording_active = True
    return {"status": "recording_started"}


@app.post("/stop-meeting")
async def stop_meeting():
    global recording_active
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
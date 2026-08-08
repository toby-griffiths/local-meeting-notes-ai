"""
Live meeting audio capture: mic + system loopback, mixed into one stream.
Uses PyAudioWPatch for WASAPI loopback support on Windows.
"""

try:
    import pyaudiowpatch as pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False

import numpy as np
from scipy.signal import resample
import wave
import threading
import tempfile

CHUNK = 1024
FORMAT = pyaudio.paInt16 if PYAUDIO_AVAILABLE else None
TARGET_RATE = 16000  # Whisper wants 16kHz


class MeetingRecorder:
    def __init__(self):
        if not PYAUDIO_AVAILABLE:
            raise RuntimeError("Live audio capture requires PyAudioWPatch (Windows-only).")
        self.p = pyaudio.PyAudio()
        self.recording = False
        self.mic_frames = []
        self.system_frames = []
        self.system_rate = None
        self.system_channels = None
        self.mic_thread = None
        self.system_thread = None
        self.mic_error = None
        self.system_error = None

    def _get_default_loopback_device(self):
        wasapi_info = self.p.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_speakers = self.p.get_device_info_by_index(
            wasapi_info["defaultOutputDevice"]
        )
        if not default_speakers.get("isLoopbackDevice"):
            for loopback in self.p.get_loopback_device_info_generator():
                if default_speakers["name"] in loopback["name"]:
                    return loopback
            raise RuntimeError("No loopback device found matching default output.")
        return default_speakers

    def _record_mic(self):
        try:
            mic_device = self.p.get_default_input_device_info()
            channels = min(int(mic_device["maxInputChannels"]), 1) or 1
            print(f"[mic] Using device: {mic_device['name']}")
            stream = self.p.open(
                format=FORMAT,
                channels=1,
                rate=TARGET_RATE,
                input=True,
                input_device_index=mic_device["index"],
                frames_per_buffer=CHUNK,
            )
            print("[mic] Stream opened, recording...")
            while self.recording:
                data = stream.read(CHUNK, exception_on_overflow=False)
                self.mic_frames.append(data)
            stream.stop_stream()
            stream.close()
            print(f"[mic] Stopped. Captured {len(self.mic_frames)} chunks.")
        except Exception as e:
            print(f"[mic] ERROR: {e}")
            self.mic_error = str(e)

    def _record_system(self):
        try:
            loopback_device = self._get_default_loopback_device()
            self.system_rate = int(loopback_device["defaultSampleRate"])
            self.system_channels = int(loopback_device["maxInputChannels"])  # usually 2
            print(f"[system audio] Using device: {loopback_device['name']} "
                  f"@ {self.system_rate}Hz, {self.system_channels}ch")
            stream = self.p.open(
                format=FORMAT,
                channels=self.system_channels,
                rate=self.system_rate,
                input=True,
                input_device_index=loopback_device["index"],
                frames_per_buffer=CHUNK,
            )
            print("[system audio] Stream opened, recording...")
            while self.recording:
                data = stream.read(CHUNK, exception_on_overflow=False)
                self.system_frames.append(data)
            stream.stop_stream()
            stream.close()
            print(f"[system audio] Stopped. Captured {len(self.system_frames)} chunks.")
        except Exception as e:
            print(f"[system audio] ERROR: {e}")
            self.system_error = str(e)

    def start(self):
        if self.recording:
            raise RuntimeError("Recording already in progress.")
        self.mic_frames = []
        self.system_frames = []
        self.mic_error = None
        self.system_error = None
        self.recording = True
        self.mic_thread = threading.Thread(target=self._record_mic, daemon=True)
        self.system_thread = threading.Thread(target=self._record_system, daemon=True)
        self.mic_thread.start()
        self.system_thread.start()

    def stop(self) -> str:
        if not self.recording:
            raise RuntimeError("No recording in progress.")
        self.recording = False
        self.mic_thread.join(timeout=5)
        self.system_thread.join(timeout=5)

        if self.mic_error:
            print(f"[warning] Mic capture failed: {self.mic_error}")
        if self.system_error:
            print(f"[warning] System audio capture failed: {self.system_error}")

        return self._mix_and_save()

    def _downmix_to_mono(self, raw_bytes: bytes, channels: int) -> np.ndarray:
        """Convert interleaved multi-channel int16 audio to mono by averaging channels."""
        audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
        if channels > 1:
            usable_len = (len(audio) // channels) * channels
            audio = audio[:usable_len].reshape(-1, channels).mean(axis=1)
        return audio

    def _mix_and_save(self) -> str:
        mic_audio = self._downmix_to_mono(b"".join(self.mic_frames), 1)

        sys_audio = np.array([], dtype=np.float32)
        if self.system_frames:
            sys_audio_raw = self._downmix_to_mono(
                b"".join(self.system_frames), self.system_channels or 1
            )
            if self.system_rate and self.system_rate != TARGET_RATE and sys_audio_raw.size:
                new_len = int(len(sys_audio_raw) * TARGET_RATE / self.system_rate)
                sys_audio = resample(sys_audio_raw, new_len)
            else:
                sys_audio = sys_audio_raw

        min_len = min(len(mic_audio), len(sys_audio)) if mic_audio.size and sys_audio.size else 0
        if min_len == 0:
            mixed = mic_audio if mic_audio.size else sys_audio
        else:
            mixed = mic_audio[:min_len] + sys_audio[:min_len]

        if mixed.size == 0:
            raise RuntimeError(
                "No audio captured from either mic or system audio. "
                f"Mic error: {self.mic_error}. System error: {self.system_error}."
            )

        mixed = np.clip(mixed, -32768, 32767).astype(np.int16)

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        with wave.open(tmp.name, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(TARGET_RATE)
            wf.writeframes(mixed.tobytes())

        return tmp.name

    def close(self):
        self.p.terminate()
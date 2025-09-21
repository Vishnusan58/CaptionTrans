"""FastAPI service for converting audio/video files to SRT subtitles via OpenAI Whisper."""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Iterable, List, Sequence

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import HTMLResponse, Response

from openai import OpenAI
from openai import OpenAIError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is required to start the application.")

client = OpenAI()

ALLOWED_EXTS = {".mp3", ".wav", ".m4a", ".mp4", ".mkv"}
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

app = FastAPI(title="CaptionTrans", docs_url=None, redoc_url=None)


def format_timestamp(seconds: float) -> str:
    """Return an SRT-formatted timestamp for the given number of seconds."""
    if seconds < 0:
        seconds = 0
    milliseconds = int(round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def build_srt_from_segments(segments: Sequence[dict]) -> str:
    """Construct an SRT string from Whisper verbose JSON segments."""
    if not segments:
        return ""

    srt_lines: List[str] = []
    for idx, segment in enumerate(segments, start=1):
        start_raw = segment.get("start", 0.0)
        end_raw = segment.get("end", start_raw)
        text_raw = segment.get("text", "")

        start_ts = format_timestamp(float(start_raw))
        end_ts = format_timestamp(float(end_raw))
        text = str(text_raw).replace("-->", "→").strip()

        srt_lines.append(f"{idx}\n{start_ts} --> {end_ts}\n{text}\n")

    return "\n".join(srt_lines).strip() + "\n"


INDEX_HTML = """<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"utf-8\" />
    <title>CaptionTrans &mdash; Whisper to SRT</title>
    <style>
        :root { font-family: system-ui, -apple-system, BlinkMacSystemFont, \"Segoe UI\", sans-serif; color: #111827; background: #f9fafb; }
        body { margin: 0; padding: 0; display: flex; align-items: center; justify-content: center; min-height: 100vh; }
        main { background: #ffffff; padding: 2.5rem; border-radius: 12px; box-shadow: 0 10px 30px rgba(15, 23, 42, 0.1); max-width: 480px; width: 100%; }
        h1 { font-size: 1.75rem; margin-bottom: 1.5rem; text-align: center; color: #1f2937; }
        label { display: block; margin-bottom: 0.5rem; font-weight: 600; }
        input[type=\"file\"], input[type=\"text\"], button { width: 100%; box-sizing: border-box; }
        input[type=\"file\"], input[type=\"text\"] { padding: 0.6rem 0.75rem; margin-bottom: 1rem; border-radius: 8px; border: 1px solid #d1d5db; background-color: #f9fafb; }
        input[type=\"text\"]:focus { outline: none; border-color: #2563eb; background: #ffffff; box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.2); }
        .checkbox-row { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem; }
        .checkbox-row label { margin: 0; font-weight: 500; }
        button { padding: 0.75rem; border: none; border-radius: 8px; background: #2563eb; color: #ffffff; font-size: 1rem; font-weight: 600; cursor: pointer; transition: background 0.2s ease; }
        button:hover { background: #1d4ed8; }
        button:disabled { opacity: 0.6; cursor: not-allowed; }
        .status { margin-top: 1rem; min-height: 1.25rem; font-size: 0.95rem; }
        .status.error { color: #b91c1c; }
        .status.success { color: #047857; }
        .spinner { width: 24px; height: 24px; border-radius: 50%; border: 3px solid #d1d5db; border-top-color: #2563eb; animation: spin 0.8s linear infinite; display: none; margin: 0 auto; margin-top: 1rem; }
        .spinner.active { display: block; }
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <main>
        <h1>CaptionTrans</h1>
        <form id=\"transcribe-form\">
            <label for=\"file\">Upload audio or video</label>
            <input type=\"file\" id=\"file\" name=\"file\" accept=\"audio/*,video/*\" required />

            <div class=\"checkbox-row\">
                <input type=\"checkbox\" id=\"translate_to_english\" name=\"translate_to_english\" checked />
                <label for=\"translate_to_english\">Translate to English</label>
            </div>

            <div class=\"checkbox-row\">
                <input type=\"checkbox\" id=\"direct_srt\" name=\"direct_srt\" checked />
                <label for=\"direct_srt\">Direct SRT from Whisper</label>
            </div>

            <label for=\"language_hint\">Language hint (optional)</label>
            <input type=\"text\" id=\"language_hint\" name=\"language_hint\" placeholder=\"e.g., ta\" />

            <button type=\"submit\">Transcribe</button>
            <div id=\"spinner\" class=\"spinner\" aria-hidden=\"true\"></div>
            <div id=\"status\" class=\"status\"></div>
        </form>
    </main>

    <script>
        const form = document.getElementById('transcribe-form');
        const spinner = document.getElementById('spinner');
        const statusBox = document.getElementById('status');

        function setWorking(isWorking) {
            const elements = form.querySelectorAll('input, button');
            elements.forEach((el) => { el.disabled = isWorking; });
            spinner.classList.toggle('active', isWorking);
        }

        form.addEventListener('submit', async (event) => {
            event.preventDefault();
            statusBox.textContent = '';
            statusBox.className = 'status';

            const fileInput = document.getElementById('file');
            if (!fileInput.files || fileInput.files.length === 0) {
                statusBox.textContent = 'Please choose an audio or video file.';
                statusBox.classList.add('error');
                return;
            }

            const formData = new FormData();
            formData.append('file', fileInput.files[0]);

            const translate = document.getElementById('translate_to_english').checked;
            const direct = document.getElementById('direct_srt').checked;
            const language = document.getElementById('language_hint').value.trim();

            formData.append('translate_to_english', translate ? 'true' : 'false');
            formData.append('direct_srt', direct ? 'true' : 'false');
            if (language) {
                formData.append('language_hint', language);
            }

            setWorking(true);

            try {
                const response = await fetch('/transcribe', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    let message = 'Transcription failed.';
                    try {
                        const data = await response.json();
                        if (data && data.detail) {
                            message = data.detail;
                        }
                    } catch (_) {
                        message = await response.text() || message;
                    }
                    throw new Error(message);
                }

                const blob = await response.blob();
                const disposition = response.headers.get('Content-Disposition');
                let filename = 'subtitles.srt';
                if (disposition) {
                    const match = disposition.match(/filename="?([^";]+)"?/i);
                    if (match && match[1]) {
                        filename = match[1];
                    }
                }

                const url = window.URL.createObjectURL(blob);
                const anchor = document.createElement('a');
                anchor.href = url;
                anchor.download = filename;
                document.body.appendChild(anchor);
                anchor.click();
                document.body.removeChild(anchor);
                window.URL.revokeObjectURL(url);

                statusBox.textContent = 'Download started.';
                statusBox.classList.add('success');
            } catch (error) {
                statusBox.textContent = error.message || 'An unexpected error occurred.';
                statusBox.classList.add('error');
            } finally {
                setWorking(false);
            }
        });
    </script>
</body>
</html>
"""


async def save_upload_to_temp(upload: UploadFile, suffix: str) -> str:
    """Persist the uploaded file to a temporary location, enforcing size limits."""
    fd, temp_path = tempfile.mkstemp(suffix=suffix)
    total_bytes = 0
    try:
        await upload.seek(0)
        with os.fdopen(fd, "wb") as buffer:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File exceeds maximum upload size of {MAX_UPLOAD_MB} MB.",
                    )
                buffer.write(chunk)
    except HTTPException:
        os.unlink(temp_path)
        raise
    except Exception as exc:
        os.unlink(temp_path)
        logger.exception("Failed to store uploaded file: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to store the uploaded file.") from exc
    finally:
        await upload.close()

    if total_bytes == 0:
        os.unlink(temp_path)
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    return temp_path


def _prepare_segments(raw_segments: Iterable) -> List[dict]:
    prepared: List[dict] = []
    for item in raw_segments:
        if isinstance(item, dict):
            prepared.append(item)
            continue
        prepared.append(
            {
                "start": getattr(item, "start", 0.0),
                "end": getattr(item, "end", getattr(item, "start", 0.0)),
                "text": getattr(item, "text", ""),
            }
        )
    return prepared


@app.get("/", response_class=HTMLResponse)
async def home() -> str:
    """Serve the upload form."""
    return INDEX_HTML


@app.post("/transcribe", response_class=Response)
async def transcribe(
    file: UploadFile = File(...),
    translate_to_english: bool = Form(True),
    direct_srt: bool = Form(True),
    language_hint: str | None = Form(None),
) -> Response:
    """Handle transcription requests and return an SRT subtitle file."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must include a filename.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTS:
        allowed = ", ".join(sorted(ALLOWED_EXTS))
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed extensions: {allowed}.")

    temp_path = await save_upload_to_temp(file, suffix)

    language = language_hint.strip() if language_hint else None
    if language:
        language = language.lower()
        if len(language) != 2 or not language.isalpha():
            raise HTTPException(status_code=400, detail="Language hint must be a two-letter ISO-639-1 code.")

    try:
        with open(temp_path, "rb") as audio_file:
            if translate_to_english:
                response = client.audio.translations.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="srt" if direct_srt else "verbose_json",
                )
            else:
                transcription_kwargs = {
                    "model": "whisper-1",
                    "file": audio_file,
                    "response_format": "srt" if direct_srt else "verbose_json",
                }
                if language:
                    transcription_kwargs["language"] = language
                response = client.audio.transcriptions.create(**transcription_kwargs)

        if direct_srt:
            srt_content = response if isinstance(response, str) else str(response)
        else:
            raw_segments = getattr(response, "segments", None)
            if raw_segments is None and hasattr(response, "get"):
                raw_segments = response.get("segments")
            if raw_segments is None:
                raise HTTPException(status_code=500, detail="Transcription segments were not returned by the API.")
            segments = _prepare_segments(raw_segments)
            srt_content = build_srt_from_segments(segments)

    except HTTPException:
        raise
    except OpenAIError as exc:
        logger.exception("OpenAI API error: %s", exc)
        raise HTTPException(status_code=500, detail="Transcription service failed. Please try again later.") from exc
    except Exception as exc:
        logger.exception("Unexpected transcription error: %s", exc)
        raise HTTPException(status_code=500, detail="An unexpected error occurred while transcribing the file.") from exc
    finally:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass

    filename_stem = Path(file.filename).stem or "subtitles"
    attachment_name = f"{filename_stem}_en.srt"

    headers = {
        "Content-Disposition": f"attachment; filename=\"{attachment_name}\"",
    }
    return Response(content=srt_content, media_type="text/plain; charset=utf-8", headers=headers)


__all__ = ["app", "format_timestamp", "build_srt_from_segments"]

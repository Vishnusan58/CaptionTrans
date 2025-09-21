# CaptionTrans

Minimal FastAPI service that converts audio or video uploads into SRT subtitle files using OpenAI Whisper.

## Requirements
- Python 3.11+
- An OpenAI API key with access to the Whisper model.

## Setup
1. **Clone and enter the project directory**
   ```bash
   git clone <repo-url>
   cd CaptionTrans
   ```
2. **Create and activate a virtual environment**
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Configure environment variables**
   ```bash
   export OPENAI_API_KEY="your-api-key"
   # Optional overrides
   export MAX_UPLOAD_MB=50
   export HOST=0.0.0.0
   export PORT=8000
   ```

## Running the server
```bash
uvicorn app:app --host ${HOST:-0.0.0.0} --port ${PORT:-8000}
```

Visit `http://localhost:8000` to open the web UI, upload an audio/video file, and download the resulting SRT.

## Optional extras
- **Tests**: `pytest`
- **Docker**: Build the provided Dockerfile and run the container (see file for details).
- **Nginx**: Use the included `nginx.conf` as a template for reverse proxying to the app container.

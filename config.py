import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"

PDF_PATH = DATA_DIR / "Lead Generation & Outreach SOP + CV.pdf"
COLLECTION_NAME = "leadgen_sop_cv"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL_NAME = "openai/gpt-oss-120b"

CHUNK_SIZE = 700
CHUNK_OVERLAP = 120
RRF_K = 60


def resolve_groq_api_key(
    ui_key: str | None = None,
    secrets_obj: object | None = None,
) -> str:
    """Resolve GROQ_API_KEY from UI input, Streamlit secrets, or environment."""
    if ui_key and ui_key.strip():
        return ui_key.strip()

    if secrets_obj is not None:
        try:
            secret_val = secrets_obj.get("GROQ_API_KEY")  # type: ignore[attr-defined]
            if isinstance(secret_val, str) and secret_val.strip():
                return secret_val.strip()
        except Exception:
            pass

    env_val = os.environ.get("GROQ_API_KEY", "").strip()
    if env_val:
        return env_val

    raise RuntimeError(
        "GROQ_API_KEY not found. Add it to Streamlit Secrets, .env, or the sidebar input."
    )

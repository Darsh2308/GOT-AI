# app/config.py

from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# =========================
# LLM CONFIGURATION
# =========================
LLM_MODEL = os.getenv("LLM_MODEL", "llama3")

# =========================
# EMBEDDING MODEL
# =========================
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

# =========================
# PATH CONFIGURATION
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "books")
CHROMA_PATH = os.path.join(BASE_DIR, "db", "chroma_db")
LOG_PATH = os.path.join(BASE_DIR, "logs")

# ✅ ENSURE DIRECTORIES EXIST (IMPORTANT FIX)
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(CHROMA_PATH, exist_ok=True)
os.makedirs(LOG_PATH, exist_ok=True)

# =========================
# LANGSMITH CONFIG
# =========================
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "got-ai")
# Default tracing off for local CLI runs; opt in via .env when needed.
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false")

# =========================
# DEBUG MODE
# =========================
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

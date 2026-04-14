#!/usr/bin/env bash
# =============================================================================
#  GOT-AI  —  One-shot setup & launcher
#  Supports: macOS, Windows (Git Bash), Linux
# =============================================================================
#
#  HOW TO START
#  ─────────────
#  Open a terminal in the project root, then run:
#
#      bash setup.sh                # full setup + guided launch menu
#      bash setup.sh --skip-venv    # skip venv/pip if already installed
#
#  On macOS/Linux you can also make it executable once:
#      chmod +x setup.sh
#      ./setup.sh
#
#  HOW TO STOP
#  ────────────
#  Web UI        :  press  Ctrl + C  in the terminal running Streamlit
#  CLI           :  type   exit      at the "Ask GOT-AI:" prompt
#  This script   :  press  Ctrl + C  at any prompt to abort setup
#
# =============================================================================

set -euo pipefail

# ── Detect OS ────────────────────────────────────────────────────────────────
OS="linux"
case "$(uname -s)" in
    Darwin*)  OS="mac" ;;
    MINGW*|MSYS*|CYGWIN*)  OS="windows" ;;
esac

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}[INFO]${RESET}  $*"; }
success() { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*"; }
header()  { echo -e "\n${BOLD}${CYAN}══  $*  ══${RESET}\n"; }

SKIP_VENV=false
for arg in "$@"; do [[ "$arg" == "--skip-venv" ]] && SKIP_VENV=true; done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# =============================================================================
# 1. PYTHON CHECK
# =============================================================================
header "Step 1 — Python"

PYTHON_BIN=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        ver=$("$cmd" -c "import sys; print(sys.version_info[:2])" 2>/dev/null)
        if "$cmd" -c "import sys; assert sys.version_info >= (3,10)" 2>/dev/null; then
            PYTHON_BIN="$cmd"
            success "Found $cmd  ($ver)"
            break
        else
            warn "$cmd found but version $ver is below 3.10 — skipping"
        fi
    fi
done

if [[ -z "$PYTHON_BIN" ]]; then
    error "Python 3.10+ not found. Install from https://python.org and re-run."
    exit 1
fi

# =============================================================================
# 2. VIRTUAL ENVIRONMENT
# =============================================================================
header "Step 2 — Virtual environment"

VENV_DIR="$SCRIPT_DIR/venv"

if [[ "$SKIP_VENV" == true ]]; then
    warn "--skip-venv passed — skipping venv creation and pip install"
else
    if [[ -d "$VENV_DIR" ]]; then
        info "venv already exists at $VENV_DIR"
    else
        info "Creating venv..."
        "$PYTHON_BIN" -m venv "$VENV_DIR"
        success "venv created"
    fi

    # Activate
    if [[ -f "$VENV_DIR/Scripts/activate" ]]; then
        # Windows Git Bash
        source "$VENV_DIR/Scripts/activate"
    elif [[ -f "$VENV_DIR/bin/activate" ]]; then
        source "$VENV_DIR/bin/activate"
    else
        error "Cannot find venv activate script. Remove venv/ and re-run."
        exit 1
    fi
    success "venv activated  ($(python --version))"

    # Pip install
    info "Installing / updating dependencies from requirements.txt..."
    pip install --upgrade pip --quiet
    pip install -r requirements.txt --quiet
    success "All Python dependencies installed"
fi

# Make sure we can find the venv python even with --skip-venv
if [[ -f "$VENV_DIR/Scripts/python" ]]; then
    VENV_PYTHON="$VENV_DIR/Scripts/python"
elif [[ -f "$VENV_DIR/bin/python" ]]; then
    VENV_PYTHON="$VENV_DIR/bin/python"
else
    VENV_PYTHON="$PYTHON_BIN"
fi

# =============================================================================
# 3. .env FILE
# =============================================================================
header "Step 3 — Environment file (.env)"

ENV_FILE="$SCRIPT_DIR/.env"
ENV_EXAMPLE="$SCRIPT_DIR/.env.example"

if [[ -f "$ENV_FILE" ]]; then
    success ".env already exists — skipping creation"
else
    if [[ -f "$ENV_EXAMPLE" ]]; then
        cp "$ENV_EXAMPLE" "$ENV_FILE"
        info "Created .env from .env.example"
    else
        # Fallback: write minimal .env
        cat > "$ENV_FILE" <<'ENVEOF'
# LangSmith (optional — set LANGCHAIN_TRACING_V2=true to enable)
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=GOT-AI
LANGCHAIN_TRACING_V2=false

# Model overrides (defaults shown)
LLM_MODEL=llama3
EMBED_MODEL=nomic-embed-text

# Debug output
DEBUG=false
ENVEOF
        info "Created minimal .env (no .env.example found)"
    fi

    warn "Please open .env and set LANGCHAIN_API_KEY if you want LangSmith tracing."
fi

# =============================================================================
# 4. OLLAMA CHECK
# =============================================================================
header "Step 4 — Ollama"

if ! command -v ollama &>/dev/null; then
    error "Ollama not found in PATH."
    if [[ "$OS" == "mac" ]]; then
        echo -e "  Install via Homebrew:  ${CYAN}brew install ollama${RESET}"
        echo -e "  Or download from:      ${CYAN}https://ollama.com/download/mac${RESET}"
    elif [[ "$OS" == "windows" ]]; then
        echo -e "  Download from: ${CYAN}https://ollama.com/download/windows${RESET}"
    else
        echo -e "  Install from: ${CYAN}https://ollama.com${RESET}"
    fi
    echo    "  Then re-run this script."
    exit 1
fi
success "Ollama binary found  ($(ollama --version 2>/dev/null | head -1))"

# Check if Ollama server is reachable
OLLAMA_URL="${OLLAMA_HOST:-http://localhost:11434}"
if curl -s --max-time 3 "$OLLAMA_URL" &>/dev/null; then
    success "Ollama server is running at $OLLAMA_URL"
else
    warn "Ollama server not reachable at $OLLAMA_URL"

    if [[ "$OS" == "mac" ]]; then
        # On macOS, Ollama is a menu-bar app — just launch it
        info "Attempting to start Ollama.app on macOS..."
        open -a Ollama 2>/dev/null || ollama serve &>/dev/null &
    else
        info "Attempting to start Ollama server in the background..."
        ollama serve &>/dev/null &
    fi

    info "Waiting up to 10 seconds for the server..."
    for i in {1..10}; do
        sleep 1
        if curl -s --max-time 2 "$OLLAMA_URL" &>/dev/null; then
            success "Ollama server is now running"
            break
        fi
        if [[ $i -eq 10 ]]; then
            error "Could not reach Ollama after 10 seconds."
            if [[ "$OS" == "mac" ]]; then
                echo "  Open the Ollama app from your Applications folder, then re-run."
            else
                echo "  Run  'ollama serve'  in a separate terminal, then re-run this script."
            fi
            exit 1
        fi
    done
fi

# =============================================================================
# 5. REQUIRED MODELS
# =============================================================================
header "Step 5 — Ollama models"

# Read model names from .env (fall back to defaults)
LLM_MODEL_NAME=$(grep -E '^LLM_MODEL=' "$ENV_FILE" 2>/dev/null | cut -d= -f2 | tr -d '"' | tr -d "'" || echo "llama3")
EMBED_MODEL_NAME=$(grep -E '^EMBED_MODEL=' "$ENV_FILE" 2>/dev/null | cut -d= -f2 | tr -d '"' | tr -d "'" || echo "nomic-embed-text")
LLM_MODEL_NAME="${LLM_MODEL_NAME:-llama3}"
EMBED_MODEL_NAME="${EMBED_MODEL_NAME:-nomic-embed-text}"

PULLED_MODELS=$(ollama list 2>/dev/null | tail -n +2 | awk '{print $1}' | sed 's/:latest//')

check_or_pull() {
    local model="$1"
    local label="$2"
    if echo "$PULLED_MODELS" | grep -qxF "$model"; then
        success "$label ($model) — already pulled"
    else
        warn "$label ($model) not found. Pulling now (this may take a few minutes)..."
        if ollama pull "$model"; then
            success "$label ($model) — pulled successfully"
        else
            error "Failed to pull $model. Check your internet connection and try:"
            echo  "    ollama pull $model"
            exit 1
        fi
    fi
}

check_or_pull "$LLM_MODEL_NAME"   "LLM model"
check_or_pull "$EMBED_MODEL_NAME" "Embedding model"

# =============================================================================
# 6. DATA / DB DIRECTORIES
# =============================================================================
header "Step 6 — Directories"

mkdir -p "$SCRIPT_DIR/data/books"
mkdir -p "$SCRIPT_DIR/db/chroma_db"
mkdir -p "$SCRIPT_DIR/logs"
success "data/books, db/chroma_db, logs — ready"

PDF_COUNT=$(find "$SCRIPT_DIR/data/books" -name "*.pdf" 2>/dev/null | wc -l | tr -d ' ')
if [[ "$PDF_COUNT" -eq 0 ]]; then
    warn "No PDF files found in data/books/"
    warn "Add your Game of Thrones PDFs there, then run: python -m app.ingestion"
else
    success "Found $PDF_COUNT PDF file(s) in data/books/"
fi

# =============================================================================
# 7. INGESTION CHECK
# =============================================================================
header "Step 7 — Vector store"

DB_POPULATED=false
if [[ -f "$SCRIPT_DIR/db/chroma_db/chroma.sqlite3" ]]; then
    DB_SIZE=$(du -sh "$SCRIPT_DIR/db/chroma_db/chroma.sqlite3" 2>/dev/null | cut -f1)
    success "ChromaDB exists ($DB_SIZE) — ingestion already done"
    DB_POPULATED=true
else
    if [[ "$PDF_COUNT" -gt 0 ]]; then
        echo ""
        read -rp "  ChromaDB is empty. Run ingestion now? [y/N] " RUN_INGEST
        if [[ "$RUN_INGEST" =~ ^[Yy]$ ]]; then
            info "Running ingestion (this can take several minutes for large PDFs)..."
            "$VENV_PYTHON" -m app.ingestion
            success "Ingestion complete"
            DB_POPULATED=true
        else
            warn "Skipped ingestion. Run it later with:  python -m app.ingestion"
        fi
    else
        warn "Skipped ingestion — no PDFs in data/books/"
    fi
fi

# =============================================================================
# 8. QUICK DIAGNOSTICS
# =============================================================================
header "Step 8 — Diagnostics"

info "Running utils diagnostics (LLM + embeddings ping)..."
if "$VENV_PYTHON" -m app.utils 2>&1 | grep -E "(OK|PASS|success|Error|FAIL)" | head -10; then
    success "Diagnostics passed"
else
    warn "Diagnostics produced no clear output — check manually with: python -m app.utils"
fi

# =============================================================================
# 9. LAUNCH MENU
# =============================================================================
header "Setup complete — How would you like to start GOT-AI?"

if [[ "$DB_POPULATED" == false ]]; then
    warn "Note: ChromaDB is not populated. Book-related questions will return no results."
    warn "      Add PDFs to data/books/ and run: python -m app.ingestion"
fi

echo ""
echo -e "  ${BOLD}1)${RESET}  Web UI     — Streamlit  (recommended)  →  http://localhost:8501"
echo -e "  ${BOLD}2)${RESET}  CLI        — Terminal chat"
echo -e "  ${BOLD}3)${RESET}  Exit       — I'll start it myself later"
echo ""
read -rp "  Enter choice [1/2/3]: " LAUNCH_CHOICE

case "$LAUNCH_CHOICE" in
    1)
        echo ""
        info "Starting Streamlit web UI..."
        echo -e "  ${YELLOW}Press Ctrl+C to stop the server.${RESET}\n"

        # Activate venv if not already active
        if [[ -z "${VIRTUAL_ENV:-}" ]]; then
            [[ -f "$VENV_DIR/Scripts/activate" ]] && source "$VENV_DIR/Scripts/activate" \
                || source "$VENV_DIR/bin/activate"
        fi

        streamlit run app/ui.py
        ;;
    2)
        echo ""
        info "Starting CLI..."
        echo -e "  ${YELLOW}Type 'exit' to quit.${RESET}\n"

        if [[ -z "${VIRTUAL_ENV:-}" ]]; then
            [[ -f "$VENV_DIR/Scripts/activate" ]] && source "$VENV_DIR/Scripts/activate" \
                || source "$VENV_DIR/bin/activate"
        fi

        python -m app.main
        ;;
    3)
        echo ""
        success "All done. Start the app whenever you're ready:"
        echo ""
        echo -e "  ${CYAN}Web UI:${RESET}  streamlit run app/ui.py"
        echo -e "  ${CYAN}CLI:${RESET}     python -m app.main"
        echo -e "  ${CYAN}Ingest:${RESET}  python -m app.ingestion"
        echo ""
        ;;
    *)
        warn "Unrecognised choice. Run 'streamlit run app/ui.py' or 'python -m app.main' manually."
        ;;
esac

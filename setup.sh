#!/bin/bash
# Citewatch — one-shot setup
# Run: chmod +x setup.sh && ./setup.sh

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'
BOLD='\033[1m'

echo ""
echo -e "${CYAN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}${BOLD}  Citewatch — Setup${NC}"
echo -e "${CYAN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

echo -e "${BOLD}[1/5] Checking Python...${NC}"
if command -v python3 &>/dev/null; then
    PY=$(command -v python3)
    PY_VERSION=$($PY --version 2>&1)
    echo -e "  ${GREEN}✅ Found: ${PY_VERSION} at ${PY}${NC}"
else
    echo -e "  ${RED}❌ Python 3 not found. Install it (e.g. brew install python@3.13)${NC}"
    exit 1
fi

PY_MAJOR=$($PY -c "import sys; print(sys.version_info.major)")
PY_MINOR=$($PY -c "import sys; print(sys.version_info.minor)")
if [ "$PY_MAJOR" -lt 3 ] || ([ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 11 ]); then
    echo -e "  ${RED}❌ Python 3.11+ required (found ${PY_MAJOR}.${PY_MINOR})${NC}"
    exit 1
fi

echo ""
echo -e "${BOLD}[2/5] Setting up virtual environment...${NC}"
if [ -d ".venv" ]; then
    echo -e "  ${YELLOW}⚠️  .venv already exists — reusing${NC}"
else
    $PY -m venv .venv
    echo -e "  ${GREEN}✅ Created .venv${NC}"
fi

# shellcheck disable=SC1091
source .venv/bin/activate
echo -e "  ${GREEN}✅ Activated${NC}"

echo ""
echo -e "${BOLD}[3/5] Installing citewatch and dependencies...${NC}"
echo -e "  ${CYAN}(this may take 1-2 minutes on first run)${NC}"
pip install --upgrade pip -q 2>/dev/null
pip install -e . -q 2>&1 | grep -E "^(Successfully|ERROR|Requirement)" || true
echo -e "  ${GREEN}✅ Installed citewatch${NC}"

echo ""
echo -e "${BOLD}[4/5] Configuring API keys...${NC}"

if [ ! -f ".env" ]; then
    cp .env.example .env
    echo -e "  ${YELLOW}📄 Created .env from .env.example${NC}"
fi

set -a
# shellcheck disable=SC1091
source .env 2>/dev/null || true
set +a

if [ -z "$ANTHROPIC_API_KEY" ] || [[ "$ANTHROPIC_API_KEY" == *"xxxxx"* ]]; then
    echo -e "  ${YELLOW}⚠️  ANTHROPIC_API_KEY not set in .env${NC}"
    echo -e "     ${CYAN}https://console.anthropic.com/settings/keys${NC}"
    HAS_ANTHROPIC=false
else
    echo -e "  ${GREEN}✅ ANTHROPIC_API_KEY is set${NC}"
    HAS_ANTHROPIC=true
fi

if [ -z "$GOOGLE_API_KEY" ] || [[ "$GOOGLE_API_KEY" == *"AIzaXXX"* ]]; then
    echo -e "  ${YELLOW}ℹ️  GOOGLE_API_KEY not set (optional — Gemini models)${NC}"
else
    echo -e "  ${GREEN}✅ GOOGLE_API_KEY is set${NC}"
fi

echo ""
echo -e "${BOLD}[5/5] Verifying installation...${NC}"

if citewatch --help &>/dev/null; then
    echo -e "  ${GREEN}✅ CLI installed: citewatch${NC}"
else
    echo -e "  ${RED}❌ CLI failed to load${NC}"
    exit 1
fi

echo -e "  ${CYAN}Testing embeddings...${NC}"

if curl -s http://localhost:11434/api/tags &>/dev/null; then
    echo -e "  ${GREEN}✅ Ollama: running on :11434${NC}"
    if ollama list 2>/dev/null | grep -q "nomic-embed-text"; then
        echo -e "  ${GREEN}✅ Model: nomic-embed-text available${NC}"
    else
        echo -e "  ${YELLOW}⚠️  Pulling nomic-embed-text model (~274MB)...${NC}"
        ollama pull nomic-embed-text
    fi
else
    echo -e "  ${YELLOW}ℹ️  Ollama not running — using built-in embeddings (onnxruntime)${NC}"
fi

EMB_OUTPUT=$(python -c "
from citewatch.embeddings import get_embedding_manager
emb = get_embedding_manager()
emb.health_check()
print('OK')
" 2>&1) && echo -e "  ${GREEN}✅ Embeddings working${NC}" || {
    echo -e "  ${RED}❌ Embeddings failed:${NC}"
    echo -e "  ${RED}   ${EMB_OUTPUT}${NC}" | head -5
}

echo ""
echo -e "${CYAN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}${BOLD}  ✅ Setup Complete${NC}"
echo -e "${CYAN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "  ${BOLD}Activate:${NC}  source .venv/bin/activate"
echo ""
echo -e "  ${BOLD}Next:${NC}"
echo -e "    1. Edit config.yaml (target.domain, competitors)"
echo -e "    2. citewatch status"
echo -e "    3. citewatch ingest"
echo -e "    4. citewatch expand \"your seed query\""
echo ""

if [ "$HAS_ANTHROPIC" = false ]; then
    echo -e "  ${YELLOW}${BOLD}Add ANTHROPIC_API_KEY (and/or GOOGLE_API_KEY) to .env${NC}"
    echo ""
fi

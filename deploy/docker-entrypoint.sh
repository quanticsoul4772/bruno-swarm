#!/bin/bash
set -e

EXPECTED_MODELS="orchestrator frontend backend test security docs devops"

# --- GPU Detection ---
echo "=== Bruno Swarm Ollama Server ==="
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
    echo "[gpu] $GPU_NAME ($GPU_VRAM)"
else
    echo "[warn] No GPU detected — models will run on CPU (very slow)"
fi

# --- Start Ollama ---
echo "[start] ollama serve"
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to be ready (up to 30s)
for i in $(seq 1 30); do
    if curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
        echo "[ready] Ollama is up (${i}s)"
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "[error] Ollama failed to start after 30s"
        exit 1
    fi
    sleep 1
done

# --- Model Health Check ---
echo ""
echo "=== Model Health Check ==="
MISSING=0
for model in $EXPECTED_MODELS; do
    if ollama list 2>/dev/null | grep -q "^${model} "; then
        echo "  $model [OK]"
    else
        echo "  $model [MISSING]"
        MISSING=$((MISSING + 1))
    fi
done
echo ""

if [ "$MISSING" -gt 0 ]; then
    echo "[warn] $MISSING model(s) missing — some agents will not work"
else
    echo "[ok] All 7 models loaded"
fi

# --- Info ---
echo ""
echo "API: http://0.0.0.0:11434"
echo "OLLAMA_MAX_LOADED_MODELS=${OLLAMA_MAX_LOADED_MODELS:-3}"
echo "OLLAMA_KEEP_ALIVE=${OLLAMA_KEEP_ALIVE:-30m}"
echo ""
echo "Connect: bruno-swarm tui -u http://<host>:11434"
echo ""

# --- Graceful Shutdown ---
cleanup() {
    echo "[shutdown] Stopping Ollama..."
    kill "$OLLAMA_PID" 2>/dev/null
    wait "$OLLAMA_PID" 2>/dev/null
    echo "[shutdown] Done"
    exit 0
}
trap cleanup SIGTERM SIGINT

# --- Foreground Wait ---
wait "$OLLAMA_PID"

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_HOST="${HOST:-127.0.0.1}"
BACKEND_PORT="${PORT:-8000}"
FRONTEND_HOST="${FRONTEND_HOST:-127.0.0.1}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"

cleanup() {
  trap - EXIT INT TERM
  if [[ -n "${BACKEND_PID:-}" ]] && kill -0 "$BACKEND_PID" 2>/dev/null; then
    kill "$BACKEND_PID" 2>/dev/null || true
  fi
  if [[ -n "${FRONTEND_PID:-}" ]] && kill -0 "$FRONTEND_PID" 2>/dev/null; then
    kill "$FRONTEND_PID" 2>/dev/null || true
  fi
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "=========================================================="
echo " Starting Control One — VLM Workbench & Auto-Annotator"
echo " Backend API : http://${BACKEND_HOST}:${BACKEND_PORT}"
echo " Frontend UI : http://${FRONTEND_HOST}:${FRONTEND_PORT}"
echo "=========================================================="

HOST="$BACKEND_HOST" PORT="$BACKEND_PORT" "$ROOT_DIR/backend/run_backend.sh" &
BACKEND_PID=$!

npm --prefix "$ROOT_DIR/frontend" run dev -- --hostname "$FRONTEND_HOST" --port "$FRONTEND_PORT" &
FRONTEND_PID=$!

wait "$BACKEND_PID" "$FRONTEND_PID"

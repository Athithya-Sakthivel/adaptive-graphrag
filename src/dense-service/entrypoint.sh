#!/bin/sh
set -eu

: "${DENSE_HOST:=0.0.0.0}"
: "${DENSE_PORT:=8200}"

# Single worker by design: the model is loaded once, and inference is
# dispatched to a thread pool (plus FastEmbed's own parallel workers).
# Multiple uvicorn workers would multiply model memory by N.
exec uvicorn host_dense:app \
  --host "${DENSE_HOST}" \
  --port "${DENSE_PORT}" \
  --lifespan on \
  --loop uvloop \
  --http httptools \
  --workers 1 \
  --timeout-graceful-shutdown 30

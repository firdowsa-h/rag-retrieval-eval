#!/usr/bin/env bash
set -euo pipefail
for program in python3 docker; do
  if ! command -v "$program" >/dev/null; then
    echo "Missing $program"
    exit 1
  fi
done
python3 --version
docker --version
docker compose version
curl --fail --silent --show-error http://localhost:6333/ >/dev/null && echo "Qdrant is reachable on localhost:6333"

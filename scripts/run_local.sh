#!/bin/sh
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
if [ ! -x .venv/bin/python ]; then
  printf '%s\n' 'Create .venv and install VoiceSep first; see README.md.' >&2
  exit 1
fi
if [ ! -d frontend/dist ]; then
  (cd frontend && npm ci && npm run build)
fi
exec .venv/bin/python -m voicesep.cli serve

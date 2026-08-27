#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${PY:-python}"
export PYTHONUNBUFFERED=1
cd "$ROOT"
exec "$PY" -u relpose/eval_custom_pose.py "$@"

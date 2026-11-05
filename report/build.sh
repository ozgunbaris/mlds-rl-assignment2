#!/usr/bin/env bash
# Build report/report.pdf from report/report.md (needs pandoc and xelatex).
set -e
cd "$(dirname "$0")/.."
.venv/bin/python report/build.py 2>/dev/null || python3 report/build.py

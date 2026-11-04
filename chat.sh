#!/bin/bash
cd "$(dirname "$0")" && exec ${PYTHON:-python3} src/chat.py "$@"

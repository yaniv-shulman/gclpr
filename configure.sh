#!/usr/bin/env bash

set -euo pipefail

poetry install --with dev --no-root

cat <<'EOF'
Environment is ready.

Use one of:
  poetry run <command>
  poetry shell
EOF

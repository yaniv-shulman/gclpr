#!/usr/bin/env bash

set -euo pipefail

GCLPR_REPO_DIR=$(git rev-parse --show-toplevel)
TARGETS=("$GCLPR_REPO_DIR/src" "$GCLPR_REPO_DIR/tests")

while getopts ":f" option; do
    case $option in
        f)
            FIX=1
            ;;
        \?)
            echo "Error: Invalid option, use -f to apply autofixes where supported"
            exit 1
            ;;
    esac
done

if [ -z "${FIX:-}" ]; then
    poetry run black --check "${TARGETS[@]}"
    poetry run ruff check "${TARGETS[@]}"
else
    poetry run black "${TARGETS[@]}"
    poetry run ruff check "${TARGETS[@]}" --fix
fi

poetry run mypy "${TARGETS[@]}"

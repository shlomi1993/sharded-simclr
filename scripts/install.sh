#!/usr/bin/env bash
# Create the project virtual environment and install sharded-simclr in editable mode (Linux, macOS, or WSL).
#
# Usage: bash scripts/install.sh [--dev] [--cpu]
#   --dev  Also install development extras (pytest).
#   --cpu  Install CPU-only PyTorch wheels (Linux).
# Set PYTHON to choose the interpreter (default: python3.12).

set -euo pipefail
cd "$(dirname "$0")/.."

VENV_DIR=".sharded-simclr-venv"
PYTHON="${PYTHON:-python3.12}"
TORCH_VERSION="2.9.1"
TORCHVISION_VERSION="0.24.1"
DEV=false
CPU=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h) sed -n '2,7p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        --dev) DEV=true; shift ;;
        --cpu) CPU=true; shift ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

command -v "$PYTHON" >/dev/null || { echo "$PYTHON not found. Install Python 3.12+ or set PYTHON." >&2; exit 1; }
"$PYTHON" -c 'import sys; sys.exit(sys.version_info < (3, 12))' || { echo "$PYTHON is older than 3.12." >&2; exit 1; }

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "Creating $VENV_DIR with $PYTHON"
    "$PYTHON" -m venv "$VENV_DIR" || {
        echo "Could not create $VENV_DIR. On Debian/Ubuntu, install the venv package (e.g. sudo apt install" \
            "python3.12-venv), delete $VENV_DIR, and rerun." >&2
        exit 1
    }
fi
PIP=("$VENV_DIR/bin/python" -m pip)
"${PIP[@]}" install --upgrade pip

if [[ "$CPU" == true ]]; then
    "${PIP[@]}" install "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" \
        --index-url https://download.pytorch.org/whl/cpu
fi

TARGET="."
[[ "$DEV" == false ]] || TARGET=".[dev]"
"${PIP[@]}" install -e "$TARGET"

echo "Done. Activate with: source $VENV_DIR/bin/activate"

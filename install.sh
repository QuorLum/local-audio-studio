#!/usr/bin/env bash
# local-audio-studio installer for Linux, macOS and WSL2.
#
#   ./install.sh                      # create .venv, install PyTorch (GPU build if an NVIDIA GPU is found),
#                                     # install requirements, then offer to download the models
#   ./install.sh --models all         # ...and download every model (about 6 GB) without asking
#   ./install.sh --models tts         # just speech (about 0.4 GB);  also: sfx, music, sfx,music, none
#   ./install.sh --cpu                # force the CPU-only PyTorch build
#   ./install.sh --cuda cu118         # pick the CUDA wheel set (cu118, cu126 default, cu128)
#   ./install.sh --no-venv            # install into the current Python environment
#   ./install.sh --dry-run            # print what would run, change nothing
#
# Needs: Python 3.10-3.12 and git (the official Stable Audio 3 code is installed from GitHub).
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

VENV=".venv"
USE_VENV=1
FORCE_CPU=0
CUDA_TAG="cu126"
MODELS="ask"
DRY_RUN=0
TORCH_VERSION="2.7.1"

usage() { sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
  case "$1" in
    --cpu) FORCE_CPU=1 ;;
    --cuda) CUDA_TAG="${2:?--cuda needs a value such as cu126}"; shift ;;
    --models) MODELS="${2:?--models needs all|tts|sfx|music|none|ask}"; shift ;;
    --venv) VENV="${2:?--venv needs a directory}"; shift ;;
    --no-venv) USE_VENV=0 ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
run() { printf '+ %s\n' "$*"; if [ "$DRY_RUN" = 0 ]; then "$@"; fi; }

# --- 1. Python 3.10 - 3.12 --------------------------------------------------------------------
PY=""
for candidate in python3.12 python3.11 python3.10 python3 python; do
  if command -v "$candidate" >/dev/null 2>&1 &&
     "$candidate" -c 'import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] < (3, 13) else 1)' 2>/dev/null; then
    PY="$candidate"; break
  fi
done
if [ -z "$PY" ]; then
  echo "error: Python 3.10, 3.11 or 3.12 is required (3.13+ is not supported by Kokoro's text front-end yet)." >&2
  echo "       macOS: brew install python@3.12     Ubuntu/WSL: sudo apt install python3.12 python3.12-venv" >&2
  exit 1
fi
say "Using $($PY --version) at $(command -v "$PY")"

if ! command -v git >/dev/null 2>&1; then
  echo "error: git is required (the official Stable Audio 3 package is installed from GitHub)." >&2
  exit 1
fi

# --- 2. virtual environment -----------------------------------------------------------------------
if [ "$USE_VENV" = 1 ]; then
  say "Creating virtual environment in $VENV"
  run "$PY" -m venv "$VENV"
  if [ "$DRY_RUN" = 0 ]; then
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
    PY="python"
  else
    PY="$VENV/bin/python"
  fi
fi
run "$PY" -m pip install --upgrade pip

# --- 3. PyTorch: pick the build for this machine -------------------------------------------------
TORCH_INDEX=""
OS="$(uname -s)"
if [ "$FORCE_CPU" = 1 ]; then
  say "PyTorch: CPU build (forced)"
  [ "$OS" = "Darwin" ] || TORCH_INDEX="https://download.pytorch.org/whl/cpu"
elif [ "$OS" = "Darwin" ]; then
  say "PyTorch: default macOS build (Apple Silicon runs on the CPU here)"
elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  say "NVIDIA GPU detected: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -n1)"
  say "PyTorch: CUDA build ($CUDA_TAG)"
  TORCH_INDEX="https://download.pytorch.org/whl/$CUDA_TAG"
else
  say "No NVIDIA GPU found: installing the CPU-only PyTorch build"
  TORCH_INDEX="https://download.pytorch.org/whl/cpu"
fi
if [ -n "$TORCH_INDEX" ]; then
  run "$PY" -m pip install "torch==$TORCH_VERSION" "torchaudio==$TORCH_VERSION" --index-url "$TORCH_INDEX"
else
  run "$PY" -m pip install "torch==$TORCH_VERSION" "torchaudio==$TORCH_VERSION"
fi

# --- 4. everything else ---------------------------------------------------------------------------
say "Installing requirements"
run "$PY" -m pip install -r requirements.txt
run "$PY" -m pip install -e . --no-deps   # adds the `local-audio` and `local-audio-setup` commands

# --- 5. models -----------------------------------------------------------------------------------------
case "$MODELS" in
  none) say "Skipping model download (run: python scripts/setup_models.py)" ;;
  all)  say "Downloading all models (about 6 GB)"; run "$PY" scripts/setup_models.py --only all --yes ;;
  ask)
    if [ -t 0 ] && [ "$DRY_RUN" = 0 ]; then
      printf '\nDownload models now? [Y/n] '
      read -r answer || answer="n"
      case "${answer:-y}" in
        [nN]*) say "Skipped. Run later: python scripts/setup_models.py" ;;
        *) run "$PY" scripts/setup_models.py ;;
      esac
    else
      say "Skipping model download (run: python scripts/setup_models.py)"
    fi ;;
  *)    say "Downloading: $MODELS"; run "$PY" scripts/setup_models.py --only "$MODELS" --yes ;;
esac

# --- 6. health check -----------------------------------------------------------------------------------
say "Checking the installation"
run "$PY" scripts/generate_audio.py --doctor || true

echo
echo "Done."
if [ "$USE_VENV" = 1 ]; then echo "Activate the environment with:  source $VENV/bin/activate"; fi
cat <<'EOF'

  python scripts/generate_audio.py --mode tts   --text "Hello from my laptop." -o hello.wav
  python scripts/generate_audio.py --mode sfx   --prompt "rain on a tin roof" --duration 8 -o rain.wav
  python scripts/generate_audio.py --mode music --prompt "calm acoustic guitar" --duration 30 -o bed.wav
EOF

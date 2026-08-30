#!/usr/bin/env bash
#
# One-command setup for macOS and Linux.
#
# Checks for Node.js 22.13+, Python 3.13, FFmpeg, and FFprobe, offers to
# install whatever is missing, then runs `npm install`. The optional Python/ML
# environment is installed from the app on first use, or eagerly with
# `--pipeline`. Re-running is safe: anything already present is skipped.
#
#   ./scripts/bootstrap.sh          install missing tools after confirming
#   ./scripts/bootstrap.sh --yes    do not prompt
#   ./scripts/bootstrap.sh --check  only report what is missing
#   ./scripts/bootstrap.sh --pipeline  also install models now

set -euo pipefail

cd "$(dirname "$0")/.."

REQUIRED_NODE_MAJOR=22
REQUIRED_NODE_MINOR=13
REQUIRED_PYTHON="3.13"

ASSUME_YES=0
CHECK_ONLY=0
INSTALL_PIPELINE=0
for argument in "$@"; do
  case "$argument" in
    -y|--yes) ASSUME_YES=1 ;;
    --check) CHECK_ONLY=1 ;;
    --pipeline) INSTALL_PIPELINE=1 ;;
    -h|--help)
      sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "Unknown option: $argument" >&2
      exit 2
      ;;
  esac
done

if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
  BOLD=$'\033[1m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RED=$'\033[31m'; RESET=$'\033[0m'
else
  BOLD=""; GREEN=""; YELLOW=""; RED=""; RESET=""
fi

ok()   { printf '  %s✓%s %s\n' "$GREEN" "$RESET" "$1"; }
miss() { printf '  %s✗%s %s\n' "$RED" "$RESET" "$1"; }
note() { printf '  %s→%s %s\n' "$YELLOW" "$RESET" "$1"; }
step() { printf '\n%s%s%s\n' "$BOLD" "$1" "$RESET"; }

have() { command -v "$1" >/dev/null 2>&1; }

# Sona's setup script accepts the first working `python3` on PATH, which may be
# an older interpreter. Resolve an explicit 3.13 and hand it over via $PYTHON.
PYTHON_BIN=""
find_python() {
  local candidate
  for candidate in "python${REQUIRED_PYTHON}" python3 python; do
    have "$candidate" || continue
    if "$candidate" -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3, 13) else 1)' 2>/dev/null; then
      PYTHON_BIN="$candidate"
      return 0
    fi
  done
  return 1
}

node_ok() {
  have node || return 1
  local version major minor
  version="$(node -v 2>/dev/null | sed 's/^v//')"
  major="${version%%.*}"
  minor="${version#*.}"; minor="${minor%%.*}"
  [ "$major" -gt "$REQUIRED_NODE_MAJOR" ] && return 0
  [ "$major" -eq "$REQUIRED_NODE_MAJOR" ] && [ "$minor" -ge "$REQUIRED_NODE_MINOR" ]
}

missing=()

step "Checking prerequisites"

if node_ok; then
  ok "Node.js $(node -v)"
else
  if have node; then
    miss "Node.js $(node -v) is older than ${REQUIRED_NODE_MAJOR}.${REQUIRED_NODE_MINOR}"
  else
    miss "Node.js ${REQUIRED_NODE_MAJOR}.${REQUIRED_NODE_MINOR}+ is not installed"
  fi
  missing+=(node)
fi

if find_python; then
  ok "Python $("$PYTHON_BIN" -c 'import platform; print(platform.python_version())') ($PYTHON_BIN)"
else
  miss "Python ${REQUIRED_PYTHON}+ is not installed"
  missing+=(python)
fi

for tool in ffmpeg ffprobe; do
  if have "$tool"; then
    ok "$($tool -version 2>/dev/null | head -1 | cut -d' ' -f1-3)"
  else
    miss "$tool is not installed"
    missing+=(ffmpeg)
  fi
done

# ffmpeg and ffprobe ship together; do not queue the same package twice.
if [ ${#missing[@]} -gt 0 ]; then
  mapfile -t missing < <(printf '%s\n' "${missing[@]}" | awk '!seen[$0]++') 2>/dev/null || \
    missing=($(printf '%s\n' "${missing[@]}" | awk '!seen[$0]++'))
fi

if [ "$CHECK_ONLY" -eq 1 ]; then
  [ ${#missing[@]} -eq 0 ] && { step "Everything is installed."; exit 0; }
  step "Missing: ${missing[*]}"
  exit 1
fi

install_macos() {
  if ! have brew; then
    cat >&2 <<'EOF'

Homebrew is required to install the missing tools automatically.
Install it from https://brew.sh and re-run this script, or install
Node.js 22.13+, Python 3.13, and FFmpeg manually.
EOF
    exit 1
  fi
  local packages=()
  for item in "${missing[@]}"; do
    case "$item" in
      node)   packages+=("node@22") ;;
      python) packages+=("python@3.13") ;;
      ffmpeg) packages+=("ffmpeg") ;;
    esac
  done
  note "brew install ${packages[*]}"
  brew install "${packages[@]}"
  # Homebrew keeps versioned formulae out of PATH by default.
  for formula in node@22 python@3.13; do
    if printf '%s\n' "${packages[@]}" | grep -qx "$formula"; then
      local prefix
      prefix="$(brew --prefix "$formula" 2>/dev/null || true)"
      [ -n "$prefix" ] && [ -d "$prefix/bin" ] && export PATH="$prefix/bin:$PATH"
    fi
  done
}

install_linux() {
  local packages=()
  if have apt-get; then
    for item in "${missing[@]}"; do
      case "$item" in
        node)   packages+=("nodejs") ;;
        python) packages+=("python3.13" "python3.13-venv") ;;
        ffmpeg) packages+=("ffmpeg") ;;
      esac
    done
    note "sudo apt-get install -y ${packages[*]}"
    if printf '%s\n' "${missing[@]}" | grep -qx python; then
      cat >&2 <<'EOF'

Note: Python 3.13 is not in the default repositories on older Debian/Ubuntu
releases. If the install below fails, add the deadsnakes PPA first:
  sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt-get update
EOF
    fi
    if printf '%s\n' "${missing[@]}" | grep -qx node; then
      cat >&2 <<'EOF'

Note: distribution Node.js packages are often older than 22.13. If the
installed version is too old, use nvm (https://github.com/nvm-sh/nvm) or
NodeSource instead.
EOF
    fi
    sudo apt-get update && sudo apt-get install -y "${packages[@]}"
  elif have dnf; then
    for item in "${missing[@]}"; do
      case "$item" in
        node)   packages+=("nodejs") ;;
        python) packages+=("python3.13") ;;
        ffmpeg) packages+=("ffmpeg") ;;
      esac
    done
    note "sudo dnf install -y ${packages[*]}"
    sudo dnf install -y "${packages[@]}"
  elif have pacman; then
    for item in "${missing[@]}"; do
      case "$item" in
        node)   packages+=("nodejs" "npm") ;;
        python) packages+=("python") ;;
        ffmpeg) packages+=("ffmpeg") ;;
      esac
    done
    note "sudo pacman -S --needed ${packages[*]}"
    sudo pacman -S --needed "${packages[@]}"
  else
    echo "No supported package manager found (apt-get, dnf, pacman)." >&2
    echo "Install Node.js 22.13+, Python 3.13, and FFmpeg manually, then re-run." >&2
    exit 1
  fi
}

if [ ${#missing[@]} -gt 0 ]; then
  step "Missing: ${missing[*]}"
  if [ "$ASSUME_YES" -eq 0 ]; then
    printf 'Install them now? [y/N] '
    read -r reply </dev/tty || reply=""
    case "$reply" in
      [yY]*) ;;
      *) echo "Aborted. Nothing was installed."; exit 1 ;;
    esac
  fi

  case "$(uname -s)" in
    Darwin) install_macos ;;
    Linux)  install_linux ;;
    *) echo "Unsupported platform: $(uname -s). Use scripts/bootstrap.ps1 on Windows." >&2; exit 1 ;;
  esac

  step "Re-checking"
  node_ok && ok "Node.js $(node -v)" || { miss "Node.js is still missing or too old"; exit 1; }
  find_python && ok "Python $("$PYTHON_BIN" -c 'import platform; print(platform.python_version())')" \
    || { miss "Python ${REQUIRED_PYTHON}+ is still missing"; exit 1; }
  have ffmpeg && have ffprobe && ok "FFmpeg and FFprobe" || { miss "FFmpeg is still missing"; exit 1; }
fi

step "Installing JavaScript dependencies"
npm install

if [ "$INSTALL_PIPELINE" -eq 1 ]; then
  step "Setting up local song processing"
  note "this downloads PyTorch and ~249 MiB of separation models; it can take several minutes"
  PYTHON="$PYTHON_BIN" npm run pipeline:setup
fi

step "Done"
cat <<'EOF'
Start Sona with:

  npm start

That runs the web interface and the local processing companion together.
The first Add a song can install the optional processing tools in the app.
EOF

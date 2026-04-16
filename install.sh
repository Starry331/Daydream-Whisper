#!/bin/zsh
set -euo pipefail

REPO_URL="${DWHISPER_REPO_URL:-${DAYDREAM_REPO_URL:-https://github.com/Starry331/Daydream-Whisper.git}}"
INSTALL_DIR="${DWHISPER_INSTALL_DIR:-${DAYDREAM_INSTALL_DIR:-$HOME/Daydream-Whisper}}"
DEFAULT_MODEL="${DWHISPER_DEFAULT_MODEL:-${DAYDREAM_DEFAULT_MODEL:-whisper:base}}"
PYTHON_CMD="${PYTHON:-python3}"
LAUNCHER_DIR="${DWHISPER_LAUNCHER_DIR:-${DAYDREAM_LAUNCHER_DIR:-$HOME/.local/bin}}"

BOLD=$'\033[1m'
DIM=$'\033[2m'
RESET=$'\033[0m'
GREEN=$'\033[32m'
YELLOW=$'\033[33m'
RED=$'\033[31m'

print_banner() {
    print ""
    print "   ____                  _                                 "
    print "  |  _ \  __ _ _   _  __| |_ __ ___  __ _ _ __ ___        "
    print "  | | | |/ _\` | | | |/ _\` | '__/ _ \\/ _\` | '_ \` _ \\       "
    print "  | |_| | (_| | |_| | (_| | | |  __/ (_| | | | | | |      "
    print "  |____/ \\__,_|\\__, |\\__,_|_|  \\___|\\__,_|_| |_| |_|      "
    print "               |___/                  Whisper             "
    print ""
}

print_step() {
    print "  ${GREEN}✓${RESET} $1"
}

print_warn() {
    print "  ${YELLOW}!${RESET} $1"
}

print_fail() {
    print "  ${RED}x${RESET} $1"
}

require_command() {
    local command_name=$1
    if ! command -v "$command_name" >/dev/null 2>&1; then
        print_fail "Missing required command: $command_name"
        exit 1
    fi
}

install_brew_package() {
    local package_name=$1
    if ! command -v brew >/dev/null 2>&1; then
        print_warn "Homebrew not found. Please install ${package_name} manually."
        return 0
    fi

    if brew list "$package_name" >/dev/null 2>&1; then
        print_step "${package_name} already installed"
        return 0
    fi

    print "  Installing ${package_name} with Homebrew..."
    brew install "$package_name"
    print_step "Installed ${package_name}"
}

sync_repo() {
    if [[ -d "$INSTALL_DIR/.git" ]]; then
        print "  Updating existing Daydream Whisper checkout..."
        git -C "$INSTALL_DIR" pull --rebase
        print_step "Updated repository"
        return 0
    fi

    if [[ -e "$INSTALL_DIR" ]]; then
        print_fail "${INSTALL_DIR} exists but is not a git checkout."
        exit 1
    fi

    print "  Cloning Daydream Whisper into ${INSTALL_DIR}..."
    git clone "$REPO_URL" "$INSTALL_DIR"
    print_step "Cloned repository"
}

install_python_env() {
    print "  Creating virtual environment..."
    "$PYTHON_CMD" -m venv "$INSTALL_DIR/.venv"
    "$INSTALL_DIR/.venv/bin/python" -m pip install --upgrade pip
    "$INSTALL_DIR/.venv/bin/pip" install -e "$INSTALL_DIR"
    print_step "Installed Python dependencies"
}

install_launcher() {
    mkdir -p "$LAUNCHER_DIR"
    ln -sf "$INSTALL_DIR/.venv/bin/dwhisper" "$LAUNCHER_DIR/dwhisper"
    print_step "Installed launcher at ${LAUNCHER_DIR}/dwhisper"

    case ":$PATH:" in
        *":$LAUNCHER_DIR:"*) ;;
        *)
            print_warn "${LAUNCHER_DIR} is not in PATH for the current shell."
            print "    Add this to your shell profile:"
            print "    export PATH=\"$LAUNCHER_DIR:\$PATH\""
            ;;
    esac
}

pull_default_model() {
    print "  Pulling default speech model (${DEFAULT_MODEL})..."
    if "$INSTALL_DIR/.venv/bin/dwhisper" pull "$DEFAULT_MODEL"; then
        print_step "Pulled ${DEFAULT_MODEL}"
    else
        print_warn "Could not pull ${DEFAULT_MODEL}. You can run it manually later."
    fi
}

main() {
    print_banner
    print "  ${BOLD}Daydream Whisper Installer${RESET}"
    print "  ${DIM}Local Apple Silicon speech recognition with MLX Whisper.${RESET}"
    print ""

    require_command git
    require_command "$PYTHON_CMD"

    install_brew_package portaudio
    install_brew_package ffmpeg
    sync_repo
    install_python_env
    install_launcher
    pull_default_model

    print ""
    print "  ${BOLD}Next steps${RESET}"
    print ""
    print "    dwhisper models"
    print "    dwhisper run /path/to/audio.wav"
    print "    dwhisper devices"
    print "    dwhisper listen"
    print ""
}

main "$@"

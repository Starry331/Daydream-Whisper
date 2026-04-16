#!/bin/zsh
set -euo pipefail

INSTALL_DIR="${DWHISPER_INSTALL_DIR:-${DAYDREAM_INSTALL_DIR:-$HOME/Daydream-Whisper}}"
DWHISPER_HOME="${DWHISPER_HOME:-${DAYDREAM_HOME:-$HOME/.dwhisper}}"
LAUNCHER_PATH="${DWHISPER_LAUNCHER_PATH:-${DAYDREAM_LAUNCHER_PATH:-$HOME/.local/bin/dwhisper}}"
DEFAULT_INSTALL_DIR="$HOME/Daydream-Whisper"
DEFAULT_DWHISPER_HOME="$HOME/.dwhisper"
DEFAULT_LAUNCHER_PATH="$HOME/.local/bin/dwhisper"

BOLD=$'\033[1m'
DIM=$'\033[2m'
RESET=$'\033[0m'
GREEN=$'\033[32m'
YELLOW=$'\033[33m'
RED=$'\033[31m'

print_banner() {
    print ""
    print "   ____                  _                                 "
    print "  |  _ \\  __ _ _   _  __| |_ __ ___  __ _ _ __ ___        "
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

confirm() {
    local prompt=$1
    local default=${2:-n}
    local hint
    if [[ "$default" == "y" ]]; then
        hint="Y/n"
    else
        hint="y/N"
    fi

    print -n "  ${BOLD}${prompt}${RESET} ${DIM}[${hint}]${RESET}: "
    local input
    read -r input
    input=${input:-$default}
    [[ "$input" == [yY]* ]]
}

remove_path() {
    local target=$1
    if [[ -e "$target" || -L "$target" ]]; then
        rm -rf "$target"
        print_step "Removed $target"
    else
        print_warn "$target not found"
    fi
}

is_safe_directory_target() {
    local target=$1
    [[ -n "$target" ]] || return 1
    [[ "$target" != "/" ]] || return 1
    [[ "$target" != "$HOME" ]] || return 1
    [[ "$target" != "$HOME/" ]] || return 1
    return 0
}

is_daydream_whisper_checkout() {
    local target=$1

    [[ -d "$target" ]] || return 1
    is_safe_directory_target "$target" || return 1
    [[ -f "$target/pyproject.toml" ]] || return 1
    grep -q 'name = "daydream-whisper"' "$target/pyproject.toml" 2>/dev/null || return 1
    [[ -f "$target/dwhisper" || -f "$target/src/dwhisper/cli.py" ]] || return 1
    return 0
}

is_daydream_whisper_home() {
    local target=$1
    local basename=${target:t}

    [[ -d "$target" ]] || return 1
    is_safe_directory_target "$target" || return 1

    if [[ "$target" == "$DEFAULT_DWHISPER_HOME" || "$basename" == ".dwhisper" ]]; then
        return 0
    fi

    [[ -f "$target/registry.yaml" || -d "$target/models" ]]
}

launcher_belongs_to_daydream_whisper() {
    local launcher=$1
    local target=

    if [[ -L "$launcher" ]]; then
        target=$(readlink "$launcher")
        case "$target" in
            "$INSTALL_DIR"/.venv/bin/dwhisper|"$INSTALL_DIR"/dwhisper)
                return 0
                ;;
            *)
                ;;
        esac
        [[ "$target" == *"/Daydream-Whisper/.venv/bin/dwhisper" ]] && return 0
        [[ "$target" == *"/Daydream_whisper/dwhisper" ]] && return 0
        return 1
    fi

    [[ -f "$launcher" ]] || return 1
    grep -Eq 'python(3)? -m dwhisper|Daydream-Whisper|/\.venv/bin/dwhisper' "$launcher" 2>/dev/null
}

remove_checkout_if_managed() {
    if is_daydream_whisper_checkout "$INSTALL_DIR"; then
        remove_path "$INSTALL_DIR"
    else
        print_warn "Skipped ${INSTALL_DIR}: it does not look like a Daydream Whisper checkout."
    fi
}

remove_home_if_managed() {
    if is_daydream_whisper_home "$DWHISPER_HOME"; then
        remove_path "$DWHISPER_HOME"
    else
        print_warn "Skipped ${DWHISPER_HOME}: it does not look like a dedicated Daydream Whisper home."
    fi
}

remove_launcher_if_managed() {
    if [[ ! -L "$LAUNCHER_PATH" && ! -f "$LAUNCHER_PATH" ]]; then
        print_warn "$LAUNCHER_PATH not found"
        return 0
    fi

    if launcher_belongs_to_daydream_whisper "$LAUNCHER_PATH"; then
        rm -f "$LAUNCHER_PATH"
        print_step "Removed $LAUNCHER_PATH"
    else
        print_warn "Skipped ${LAUNCHER_PATH}: it does not point to Daydream Whisper."
    fi
}

main() {
    print_banner
    print "  ${BOLD}Daydream Whisper Uninstaller${RESET}"
    print "  ${DIM}Removes only this Daydream Whisper checkout, launcher, and config directory.${RESET}"
    print "  ${DIM}Shared Python environments, Homebrew packages, ffmpeg, portaudio, and Hugging Face caches are left untouched.${RESET}"
    print ""

    if confirm "Remove the application checkout at ${INSTALL_DIR}?" y; then
        remove_checkout_if_managed
    fi

    if confirm "Remove the Daydream Whisper config and local model registry at ${DWHISPER_HOME}?" y; then
        remove_home_if_managed
    fi

    if confirm "Remove the launcher symlink at ${LAUNCHER_PATH}?" y; then
        remove_launcher_if_managed
    fi

    print ""
    print "  ${BOLD}Done${RESET}"
    print ""
}

main "$@"

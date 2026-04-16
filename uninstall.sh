#!/bin/zsh
set -euo pipefail

INSTALL_DIR="${DWHISPER_INSTALL_DIR:-${DAYDREAM_INSTALL_DIR:-$HOME/Daydream-Whisper}}"
DWHISPER_HOME="${DWHISPER_HOME:-${DAYDREAM_HOME:-$HOME/.dwhisper}}"
LAUNCHER_PATH="${DWHISPER_LAUNCHER_PATH:-${DAYDREAM_LAUNCHER_PATH:-$HOME/.local/bin/dwhisper}}"
HF_CACHE_DIR="${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"

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

remove_whisper_cache() {
    if [[ ! -d "$HF_CACHE_DIR" ]]; then
        print_warn "Hugging Face cache not found"
        return 0
    fi

    local found=0
    local path
    for path in "$HF_CACHE_DIR"/models--mlx-community--whisper*; do
        if [[ -e "$path" ]]; then
            rm -rf "$path"
            print_step "Removed cached model $path"
            found=1
        fi
    done

    if [[ $found -eq 0 ]]; then
        print_warn "No cached mlx-community Whisper models found under $HF_CACHE_DIR"
    fi
}

main() {
    print_banner
    print "  ${BOLD}Daydream Whisper Uninstaller${RESET}"
    print "  ${DIM}Removes the local Daydream Whisper CLI, config, launcher, and cached speech models.${RESET}"
    print ""

    if confirm "Remove the application checkout at ${INSTALL_DIR}?" y; then
        remove_path "$INSTALL_DIR"
    fi

    if confirm "Remove the Daydream Whisper config and local model registry at ${DWHISPER_HOME}?" y; then
        remove_path "$DWHISPER_HOME"
    fi

    if confirm "Remove the launcher symlink at ${LAUNCHER_PATH}?" y; then
        if [[ -L "$LAUNCHER_PATH" || -f "$LAUNCHER_PATH" ]]; then
            rm -f "$LAUNCHER_PATH"
            print_step "Removed $LAUNCHER_PATH"
        else
            print_warn "$LAUNCHER_PATH not found"
        fi
    fi

    if confirm "Remove cached mlx-community Whisper models from ${HF_CACHE_DIR}?" n; then
        remove_whisper_cache
    fi

    print ""
    print "  ${BOLD}Done${RESET}"
    print ""
}

main "$@"

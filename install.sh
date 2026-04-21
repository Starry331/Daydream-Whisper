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
# Soft sky/ice blue for the banner — uses 256-color 117 with a 16-color
# fallback (bright cyan) for terminals that don't speak 256-color.
LIGHT_BLUE=$'\033[38;5;117m'
LIGHT_BLUE_FALLBACK=$'\033[96m'

print_banner() {
    # Pick the 256-color light blue if the terminal advertises 256-color
    # support, otherwise fall back to the basic bright-cyan ANSI code so
    # dumb terminals still get a soft, readable tint instead of raw escapes.
    local banner_color="$LIGHT_BLUE"
    case "${TERM:-}" in
        *256color*|xterm-kitty|alacritty|wezterm|screen-256color|tmux-256color) ;;
        *) banner_color="$LIGHT_BLUE_FALLBACK" ;;
    esac

    # Quoted heredoc keeps every backslash, apostrophe, and backtick literal
    # so the ASCII art renders identically regardless of shell interpretation.
    print -n "$banner_color"
    cat <<'__DWHISPER_BANNER__'

   ____                  _
  |  _ \  __ _ _   _  __| |_ __ ___  __ _ _ __ ___
  | | | |/ _` | | | |/ _` | '__/ _ \/ _` | '_ ` _ \
  | |_| | (_| | |_| | (_| | | |  __/ (_| | | | | | |
  |____/ \__,_|\__, |\__,_|_|  \___|\__,_|_| |_| |_|
               |___/
   __        ___     _
   \ \      / / |__ (_)___ _ __   ___ _ __
    \ \ /\ / /| '_ \| / __| '_ \ / _ \ '__|
     \ V  V / | | | | \__ \ |_) |  __/ |
      \_/\_/  |_| |_|_|___/ .__/ \___|_|
                          |_|

__DWHISPER_BANNER__
    print -n "$RESET"
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
    print "  Creating isolated virtual environment..."
    "$PYTHON_CMD" -m venv --clear "$INSTALL_DIR/.venv"
    "$INSTALL_DIR/.venv/bin/python" -m pip install --upgrade pip
    # The ``asr-extras`` extra pulls in mlx-audio so non-Whisper models
    # (Qwen3-ASR, Parakeet, SenseVoice, …) work out of the box. If the
    # extra fails to resolve for any reason (e.g. network hiccup), fall
    # back to the base install so Whisper still works.
    if ! "$INSTALL_DIR/.venv/bin/pip" install -e "${INSTALL_DIR}[asr-extras]"; then
        print_warn "Could not install mlx-audio extras; falling back to Whisper-only install."
        "$INSTALL_DIR/.venv/bin/pip" install -e "$INSTALL_DIR"
    fi
    print_step "Installed Python dependencies"
}

verify_install_layout() {
    if [[ ! -x "$INSTALL_DIR/.venv/bin/dwhisper" ]]; then
        print_fail "Missing dwhisper launcher in $INSTALL_DIR/.venv/bin"
        exit 1
    fi

    if [[ -e "$INSTALL_DIR/.venv/bin/daydream" ]]; then
        print_fail "Unexpected daydream launcher found in Daydream Whisper virtual environment."
        print_fail "Aborting to avoid conflicting with the original Daydream CLI."
        exit 1
    fi

    if [[ -e "$LAUNCHER_DIR/daydream" ]]; then
        print_warn "Existing $LAUNCHER_DIR/daydream left untouched."
    fi
}

persist_launcher_dir_on_path() {
    # Append an `export PATH` line to the user's shell rc files so new
    # terminals (and `exec $SHELL`) can find `dwhisper` without any manual
    # edit. We only touch rc files that already exist or are the canonical
    # startup file for the user's shell.
    local export_line="export PATH=\"$LAUNCHER_DIR:\$PATH\""
    local marker="# Added by Daydream Whisper installer"
    local touched=0
    local rc_file
    local candidate_rcs=()

    case "${SHELL:-}" in
        *zsh) candidate_rcs=("$HOME/.zshrc" "$HOME/.zprofile") ;;
        *bash)
            candidate_rcs=("$HOME/.bashrc" "$HOME/.bash_profile" "$HOME/.profile")
            ;;
        *) candidate_rcs=("$HOME/.profile") ;;
    esac

    for rc_file in "${candidate_rcs[@]}"; do
        # Skip rc files that don't exist unless it's the first entry (which we
        # treat as the canonical file to create for this shell).
        if [[ ! -f "$rc_file" && "$rc_file" != "${candidate_rcs[1]}" ]]; then
            continue
        fi
        if [[ -f "$rc_file" ]] && grep -Fq "$LAUNCHER_DIR" "$rc_file" 2>/dev/null; then
            continue
        fi
        {
            print ""
            print "$marker"
            print "$export_line"
        } >> "$rc_file"
        print_step "Added $LAUNCHER_DIR to PATH in $(basename "$rc_file")"
        touched=1
    done

    if (( touched == 0 )); then
        print_warn "Could not update a shell rc file automatically."
        print "    Add this line manually:"
        print "      $export_line"
    else
        print "    ${DIM}Open a new terminal, or run: exec \"\$SHELL\"${RESET}"
    fi
}

cleanup_legacy_daydream_path() {
    # The predecessor project (Daydream-cli, LLM-era) prepended its full venv
    # bin to $PATH, which shadows python/python3/pip for any tool launched
    # from a new shell. Detect the legacy block and strip only the PATH
    # export — preserve any other exports (e.g. HF_TOKEN) the user added
    # inside the block.
    local rc_file
    local cleaned=0
    for rc_file in "$HOME/.zshrc" "$HOME/.zprofile" "$HOME/.bashrc" "$HOME/.bash_profile" "$HOME/.profile"; do
        [[ -f "$rc_file" ]] || continue
        grep -Fq "# >>> daydream >>>" "$rc_file" 2>/dev/null || continue
        grep -Fq "Daydream-cli/.venv/bin" "$rc_file" 2>/dev/null || continue
        cp "$rc_file" "${rc_file}.dwhisper-backup.$(date +%Y%m%d%H%M%S)"
        # Delete just the offending export lines inside the legacy block.
        # Leaves the marker lines and any other exports intact, so if the
        # user kept (say) HF_TOKEN here it survives.
        python3 - "$rc_file" <<'PY'
import pathlib, re, sys
path = pathlib.Path(sys.argv[1])
lines = path.read_text().splitlines(keepends=True)
out, inside = [], False
for line in lines:
    if "# >>> daydream >>>" in line:
        inside = True
        out.append(line)
        continue
    if "# <<< daydream <<<" in line:
        inside = False
        out.append(line)
        continue
    if inside and re.match(r'\s*export\s+PATH=.*Daydream-cli/\.venv/bin', line):
        continue
    out.append(line)
path.write_text("".join(out))
PY
        print_step "Cleaned legacy Daydream-cli PATH entry from $(basename "$rc_file") (backup saved)"
        cleaned=1
    done
    if (( cleaned )); then
        print "    ${DIM}The old daydream CLI venv was shadowing python/python3/pip — it's now gone from new shells.${RESET}"
    fi
}

install_launcher() {
    cleanup_legacy_daydream_path
    mkdir -p "$LAUNCHER_DIR"
    ln -sf "$INSTALL_DIR/.venv/bin/dwhisper" "$LAUNCHER_DIR/dwhisper"
    print_step "Installed launcher at ${LAUNCHER_DIR}/dwhisper"

    # Best-effort: if a user-writable Homebrew bin sits on PATH already, drop
    # a second symlink there so `dwhisper` works in the CURRENT shell without
    # any rc-file reload.
    local homebrew_bin
    for homebrew_bin in /opt/homebrew/bin /usr/local/bin; do
        if [[ -d "$homebrew_bin" && -w "$homebrew_bin" ]]; then
            ln -sf "$INSTALL_DIR/.venv/bin/dwhisper" "$homebrew_bin/dwhisper" 2>/dev/null || continue
            print_step "Also linked ${homebrew_bin}/dwhisper (already on PATH)"
            break
        fi
    done

    case ":$PATH:" in
        *":$LAUNCHER_DIR:"*)
            print_step "${LAUNCHER_DIR} already on PATH"
            ;;
        *)
            persist_launcher_dir_on_path
            ;;
    esac
}

# -----------------------------------------------------------------------------
# Interactive prompts (arrow-key aware, safe for piped `curl | sh` installs).
# All helpers no-op to defaults when stdin or stdout is not a TTY.
# -----------------------------------------------------------------------------

HAS_TTY=0
if [[ -t 0 && -t 1 ]]; then
    HAS_TTY=1
fi

# Exported by prompt_menu so callers don't have to parse command substitution.
MENU_CHOICE_VALUE=""
MENU_CHOICE_INDEX=0

_read_key() {
    # Read a single keystroke, including CSI escape sequences like arrows.
    local key rest
    IFS= read -rsk1 key 2>/dev/null || return 1
    if [[ "$key" == $'\e' ]]; then
        IFS= read -rsk2 -t 0.01 rest 2>/dev/null || rest=""
        key+="$rest"
    fi
    printf '%s' "$key"
}

_hide_cursor() { print -n $'\033[?25l' >&2; }
_show_cursor() { print -n $'\033[?25h' >&2; }

prompt_yes_no() {
    # Usage: prompt_yes_no "question" <default_yes:0|1>
    # Returns 0 for Yes, 1 for No. Arrow keys / h / l / y / n toggle; Enter confirms.
    local question="$1"
    local selection=${2:-1}

    if (( HAS_TTY == 0 )) || [[ -n "${DWHISPER_YES:-}" ]]; then
        (( selection )) && return 0 || return 1
    fi

    _hide_cursor
    local drew=0
    while true; do
        if (( drew )); then
            print -n $'\r\033[K' >&2
        fi
        drew=1
        local yes_label no_label
        if (( selection )); then
            yes_label="${BOLD}${LIGHT_BLUE}[ Yes ]${RESET}"
            no_label="${DIM}  No  ${RESET}"
        else
            yes_label="${DIM} Yes  ${RESET}"
            no_label="${BOLD}${LIGHT_BLUE}[ No ]${RESET}"
        fi
        print -n "  ${question}  ${yes_label}  ${no_label}  ${DIM}(←/→ Enter)${RESET}" >&2

        local key
        key=$(_read_key) || { selection=0; break; }
        case "$key" in
            $'\e[D'|$'\e[A'|'h'|'k') selection=1 ;;
            $'\e[C'|$'\e[B'|'l'|'j') selection=0 ;;
            'y'|'Y') selection=1; break ;;
            'n'|'N') selection=0; break ;;
            $'\n'|$'\r'|'') break ;;
            $'\x03'|$'\x04') selection=0; break ;;
        esac
    done
    print "" >&2
    _show_cursor
    (( selection ))
}

prompt_text() {
    # Usage: prompt_text "label" "default"
    # Echoes the chosen string on stdout. Empty input accepts the default.
    local label="$1"
    local default="$2"
    if (( HAS_TTY == 0 )) || [[ -n "${DWHISPER_YES:-}" ]]; then
        printf '%s' "$default"
        return 0
    fi
    local response
    print "  ${BOLD}${label}${RESET}" >&2
    print "    ${DIM}Default: ${RESET}${LIGHT_BLUE}${default}${RESET}" >&2
    print "    ${DIM}Press ${RESET}${BOLD}Enter${RESET}${DIM} to accept, or type a new path then Enter${RESET}" >&2
    print -n "  ${LIGHT_BLUE}›${RESET} " >&2
    IFS= read -r response || response=""
    if [[ -z "$response" ]]; then
        response="$default"
    fi
    # Expand leading ~ to $HOME for convenience.
    response="${response/#\~/$HOME}"
    print "  ${GREEN}✓${RESET} ${label} → ${LIGHT_BLUE}${response}${RESET}" >&2
    print "" >&2
    printf '%s' "$response"
}

prompt_menu() {
    # Usage: prompt_menu "label" <default_index> option1 option2 ...
    # Writes the chosen option text to $MENU_CHOICE_VALUE and the index to
    # $MENU_CHOICE_INDEX. Arrow keys / j / k navigate; Enter confirms.
    local label="$1"
    local default_index=$2
    shift 2
    local options=("$@")
    local count=${#options[@]}

    MENU_CHOICE_INDEX=$default_index
    MENU_CHOICE_VALUE="${options[$((default_index + 1))]}"

    if (( HAS_TTY == 0 )) || [[ -n "${DWHISPER_YES:-}" ]]; then
        return 0
    fi

    print "  ${BOLD}${label}${RESET}" >&2
    print "    ${DIM}Use ${RESET}${BOLD}↑/↓${RESET}${DIM} (or j/k, or number key) to navigate, ${RESET}${BOLD}Enter${RESET}${DIM} to confirm${RESET}" >&2
    _hide_cursor

    local drew=0 i selection=$default_index
    while true; do
        if (( drew )); then
            # Move cursor up by count lines to overwrite previous render.
            print -n "\033[${count}A" >&2
        fi
        drew=1
        for ((i = 1; i <= count; i++)); do
            print -n $'\r\033[K' >&2
            if (( i - 1 == selection )); then
                print "    ${BOLD}${LIGHT_BLUE}▶${RESET} ${LIGHT_BLUE}${options[$i]}${RESET}" >&2
            else
                print "      ${DIM}${options[$i]}${RESET}" >&2
            fi
        done

        local key
        key=$(_read_key) || break
        case "$key" in
            $'\e[A'|'k') (( selection = (selection - 1 + count) % count )) ;;
            $'\e[B'|'j') (( selection = (selection + 1) % count )) ;;
            $'\n'|$'\r'|'') break ;;
            $'\x03'|$'\x04') break ;;
            [1-9])
                # Number-key shortcut.
                local n=$((key))
                if (( n >= 1 && n <= count )); then
                    selection=$((n - 1))
                    break
                fi
                ;;
        esac
    done
    _show_cursor
    MENU_CHOICE_INDEX=$selection
    MENU_CHOICE_VALUE="${options[$((selection + 1))]}"

    # Collapse the rendered menu into a single confirmation line so later
    # prompts don't crowd it.
    print -n "\033[${count}A" >&2
    for ((i = 1; i <= count; i++)); do
        print "\033[K" >&2
    done
    print -n "\033[${count}A" >&2
    print "  ${GREEN}✓${RESET} ${label%% *} → ${LIGHT_BLUE}${MENU_CHOICE_VALUE}${RESET}" >&2
    print "" >&2
}

detect_existing_install() {
    # Print a friendly report of anything we find on disk so the user knows
    # whether this run will be a fresh install or an update of existing state.
    local found_any=0
    local findings=()

    if [[ -d "$INSTALL_DIR/.git" ]]; then
        findings+=("repo: ${INSTALL_DIR}")
        found_any=1
    fi
    if [[ -x "$INSTALL_DIR/.venv/bin/dwhisper" ]]; then
        findings+=("venv launcher: ${INSTALL_DIR}/.venv/bin/dwhisper")
        found_any=1
    fi
    if [[ -L "$LAUNCHER_DIR/dwhisper" || -x "$LAUNCHER_DIR/dwhisper" ]]; then
        findings+=("PATH launcher: ${LAUNCHER_DIR}/dwhisper")
        found_any=1
    fi
    local extra_bin
    for extra_bin in /opt/homebrew/bin/dwhisper /usr/local/bin/dwhisper; do
        if [[ -L "$extra_bin" || -x "$extra_bin" ]]; then
            findings+=("extra launcher: ${extra_bin}")
            found_any=1
        fi
    done

    if (( found_any == 0 )); then
        print "  ${LIGHT_BLUE}●${RESET} No prior Daydream Whisper install detected — this will be a fresh install."
        print ""
        return 0
    fi

    print "  ${LIGHT_BLUE}●${RESET} ${BOLD}Existing Daydream Whisper install detected${RESET}"
    local item
    for item in "${findings[@]}"; do
        print "    ${DIM}•${RESET} ${item}"
    done
    print "    ${DIM}This run will ${RESET}${BOLD}refresh${RESET}${DIM} the install in-place (git pull + pip reinstall).${RESET}"
    print ""
}

configure_interactively() {
    # Only prompt for values the user didn't already pin via env vars.
    local install_dir_locked=0 launcher_locked=0 model_locked=0
    [[ -n "${DWHISPER_INSTALL_DIR:-${DAYDREAM_INSTALL_DIR:-}}" ]] && install_dir_locked=1
    [[ -n "${DWHISPER_LAUNCHER_DIR:-${DAYDREAM_LAUNCHER_DIR:-}}" ]] && launcher_locked=1
    [[ -n "${DWHISPER_DEFAULT_MODEL:-${DAYDREAM_DEFAULT_MODEL:-}}" ]] && model_locked=1

    if (( HAS_TTY == 0 )) || [[ -n "${DWHISPER_YES:-}" ]]; then
        print_step "Non-interactive mode — using defaults."
        print "    ${DIM}install dir : ${INSTALL_DIR}${RESET}"
        print "    ${DIM}launcher dir: ${LAUNCHER_DIR}${RESET}"
        print "    ${DIM}default model: ${DEFAULT_MODEL}${RESET}"
        return 0
    fi

    print "  ${BOLD}Review installation settings${RESET}"
    print ""

    if (( ! install_dir_locked )); then
        INSTALL_DIR=$(prompt_text "Install directory" "$INSTALL_DIR")
    else
        print "  Install directory ${DIM}(pinned via env)${RESET}: ${LIGHT_BLUE}${INSTALL_DIR}${RESET}"
    fi

    if (( ! launcher_locked )); then
        LAUNCHER_DIR=$(prompt_text "Launcher directory" "$LAUNCHER_DIR")
    else
        print "  Launcher directory ${DIM}(pinned via env)${RESET}: ${LIGHT_BLUE}${LAUNCHER_DIR}${RESET}"
    fi

    if (( ! model_locked )); then
        local model_options=(
            "whisper:tiny"
            "whisper:base"
            "whisper:small"
            "whisper:medium"
            "whisper:large-v3-turbo"
        )
        # Default to current DEFAULT_MODEL if it's in the list, otherwise "base".
        local default_index=1
        local idx=0 opt
        for opt in "${model_options[@]}"; do
            if [[ "$opt" == "$DEFAULT_MODEL" ]]; then
                default_index=$idx
                break
            fi
            (( idx++ ))
        done
        prompt_menu "Default speech model ${DIM}(Enter to accept)${RESET}" "$default_index" "${model_options[@]}"
        DEFAULT_MODEL="$MENU_CHOICE_VALUE"
    else
        print "  Default model ${DIM}(pinned via env)${RESET}: ${LIGHT_BLUE}${DEFAULT_MODEL}${RESET}"
    fi

    print ""
    print "  ${BOLD}Summary${RESET}"
    print "    install dir  : ${LIGHT_BLUE}${INSTALL_DIR}${RESET}"
    print "    launcher dir : ${LIGHT_BLUE}${LAUNCHER_DIR}${RESET}"
    print "    default model: ${LIGHT_BLUE}${DEFAULT_MODEL}${RESET}"
    print ""

    if ! prompt_yes_no "Proceed with installation?" 1; then
        print_warn "Installation cancelled by user."
        exit 0
    fi
    print ""
    print "  ${BOLD}${LIGHT_BLUE}▶ Starting installation...${RESET}"
    print ""
}

verify_ready() {
    # Final sanity check: the launcher must actually run. If not, say so
    # loudly — the user's complaint was that the installer "does nothing
    # and pops up" at the end.
    local launcher="$INSTALL_DIR/.venv/bin/dwhisper"
    if [[ ! -x "$launcher" ]]; then
        print_fail "Launcher missing at ${launcher}. Installation did NOT complete."
        exit 1
    fi
    local help_line
    if ! help_line=$("$launcher" --help 2>&1 | head -n 1); then
        print_fail "Launcher exists but failed to run. Try: ${launcher} --help"
        exit 1
    fi
    print_step "dwhisper launcher verified (${help_line})"
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

    detect_existing_install
    configure_interactively

    install_brew_package portaudio
    install_brew_package ffmpeg
    sync_repo
    install_python_env
    verify_install_layout
    install_launcher
    pull_default_model
    verify_ready

    print ""
    print "  ${BOLD}${GREEN}━━━ Installation complete ━━━${RESET}"
    print ""
    print "  Launcher: ${LIGHT_BLUE}${LAUNCHER_DIR}/dwhisper${RESET}"
    print "  Repo:     ${LIGHT_BLUE}${INSTALL_DIR}${RESET}"
    print "  Model:    ${LIGHT_BLUE}${DEFAULT_MODEL}${RESET}"
    print ""
    print "  ${BOLD}Try it now${RESET}"
    print ""
    print "    dwhisper doctor"
    print "    dwhisper models"
    print "    dwhisper transcribe /path/to/audio.wav"
    print "    dwhisper devices"
    print "    dwhisper listen"
    print ""
    print "  ${DIM}If ${RESET}${BOLD}dwhisper${RESET}${DIM} is not found, open a new terminal or run: ${RESET}${BOLD}exec \"\$SHELL\"${RESET}"
    print ""
}

main "$@"

# Daydream Whisper

![Daydream Whisper logo](docs/images/daydream-whisper-logo.svg)

English | [中文](#中文)

Daydream Whisper is an Apple Silicon speech recognition CLI built around MLX Whisper. The repository is now speech-first: local audio transcription, realtime microphone listening, subtitle export, profile-driven correction, and a local HTTP API for dictation or note-taking apps.

## Status

Current release: `v0.2.0`

Implemented commands:

- `dwhisper pull`
- `dwhisper list`
- `dwhisper rm`
- `dwhisper show`
- `dwhisper models`
- `dwhisper profiles`
- `dwhisper run`
- `dwhisper transcribe`
- `dwhisper devices`
- `dwhisper listen`
- `dwhisper serve`

## Why `dwhisper`

The project name stays `Daydream Whisper`, but the executable is `dwhisper`.

- This avoids colliding with the original Daydream CLI and commands like `daydream run`, `daydream serve`, or `daydream cp`.
- The installer only creates `dwhisper`.
- The uninstaller only removes Daydream Whisper files and leaves the original Daydream CLI alone.

## Core Features

- Runs local Whisper models through MLX on Apple Silicon
- Transcribes audio files to `text`, `json`, `srt`, `vtt`, or `verbose_json`
- Supports realtime microphone transcription
- Applies transcript auto-correction from reusable hotword, vocabulary, and cleanup rules
- Can optionally send finished transcripts to a separate local Qwen or other text model for cleanup, summary, or meeting-note formatting
- Supports named profiles for dictation, meetings, subtitles, domain terms, or repeated deployment presets
- Serves an OpenAI-style speech-to-text API for local dictation or note-taking apps
- Supports built-in aliases like `whisper:base` and `whisper:large-v3-turbo`
- Accepts local bundle roots that contain an embedded Whisper checkpoint, not just bare model directories
- Lets you override model, language, task, VAD, chunking, and device settings via CLI, config, or environment variables

## Requirements

- Apple Silicon Mac
- Python `3.14+`
- `ffmpeg` for compressed audio formats such as MP3 and M4A
- `portaudio` for microphone capture through `sounddevice`

## Installation

### Interactive Install

```bash
curl -fsSL https://raw.githubusercontent.com/Starry331/Daydream-Whisper/main/install.sh -o /tmp/dwhisper-install.sh
zsh /tmp/dwhisper-install.sh
```

The installer will:

- install `portaudio` and `ffmpeg` when Homebrew is available
- clone or update the repo into `~/Daydream-Whisper`
- recreate an isolated virtual environment inside `~/Daydream-Whisper/.venv`
- install the package
- install a `dwhisper` launcher into `~/.local/bin`
- pull `whisper:base` as the default starter model

The installer does not install, overwrite, or remove the original `daydream` command from Daydream CLI.

### Manual Install

```bash
git clone https://github.com/Starry331/Daydream-Whisper.git
cd Daydream-Whisper
python3.14 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -e .
```

Use a dedicated environment for Daydream Whisper. Do not install it into an existing Daydream CLI virtual environment.

### Run From A Repo Checkout

```bash
./dwhisper --help
python3 -m dwhisper --help
```

If your shell `python3` points at some other virtual environment, prefer `./dwhisper` or `./.venv/bin/python -m dwhisper`.

### Uninstall

```bash
curl -fsSL https://raw.githubusercontent.com/Starry331/Daydream-Whisper/main/uninstall.sh -o /tmp/dwhisper-uninstall.sh
zsh /tmp/dwhisper-uninstall.sh
```

The uninstaller is intentionally conservative:

- it removes only the Daydream Whisper checkout
- it removes only the `dwhisper` launcher if that launcher points to Daydream Whisper
- it removes only `~/.dwhisper` or another clearly dedicated Daydream Whisper home
- it does not remove Homebrew packages, system Python, unrelated virtual environments, or shared Hugging Face caches

## Quick Start

Show built-in and registered models:

```bash
dwhisper models
```

List reusable speech profiles:

```bash
dwhisper profiles
```

Pull a speech model:

```bash
dwhisper pull whisper:base
```

List downloaded and discovered models:

```bash
dwhisper list
```

Inspect a local model:

```bash
dwhisper show whisper:base
```

Transcribe an audio file:

```bash
dwhisper run ./samples/interview.wav
```

`run` is a convenience alias for `transcribe`.

Transcribe with a reusable profile, correction rules, and glossary terms:

```bash
dwhisper run ./samples/interview.wav \
  --profile meeting-zh \
  --hotword Daydream \
  --vocabulary-entry helo=hello
```

Export subtitles:

```bash
dwhisper run ./samples/interview.wav --format srt -o interview.srt
dwhisper run ./samples/interview.wav --format vtt -o interview.vtt
```

Bias decoding and cleanup for domain audio:

```bash
dwhisper run ./samples/demo.wav \
  --beam-size 5 \
  --best-of 3 \
  --hotword MLX \
  --hotword Whisper \
  --corrections-file ~/.dwhisper/corrections.yaml \
  --vocabulary-file ~/.dwhisper/vocabulary.yaml
```

Optional local Qwen post-processing after Whisper finishes:

```bash
dwhisper run ./samples/meeting.wav \
  --postprocess \
  --postprocess-model qwen3.5:0.8b \
  --postprocess-base-url http://127.0.0.1:11435/v1 \
  --postprocess-mode clean
```

List microphone devices:

```bash
dwhisper devices
```

Start realtime listening:

```bash
dwhisper listen --model whisper:base
```

Start realtime listening with profile defaults:

```bash
dwhisper listen --profile meeting-zh
```

Keyboard-gated listening:

```bash
dwhisper listen --push-to-talk
```

In the current implementation, `--push-to-talk` uses the space bar to toggle capture on and off, and `q` exits the session.

Serve a local speech API for other apps:

```bash
dwhisper serve --model whisper:base --host 127.0.0.1 --port 11434
```

For lower first-request latency in dictation apps, preload the default model worker:

```bash
dwhisper serve --model whisper:base --port 11434 --preload
```

## Command Reference

| Command | Purpose | Notes |
| --- | --- | --- |
| `dwhisper pull <model>` | Download a model or register a local reference | Accepts aliases, HF repo IDs, and local paths |
| `dwhisper list` | Show downloaded and discovered models | Includes registered local entries |
| `dwhisper rm <model>` | Remove a downloaded cached model | Use `--force` to skip confirmation |
| `dwhisper show <model>` | Print local model metadata | Useful for validation and debugging |
| `dwhisper models` | List built-in and registered aliases | Shows local vs Hugging Face source |
| `dwhisper profiles` | List reusable profiles | Reads `~/.dwhisper/profiles.yaml` |
| `dwhisper run <audio>` | Transcribe a file | Alias of `dwhisper transcribe` |
| `dwhisper transcribe <audio>` | Transcribe a file | Supports text, JSON, SRT, and VTT |
| `dwhisper devices` | List input devices | Helps select `--device` for `listen` |
| `dwhisper listen` | Realtime microphone transcription | Supports VAD and push-to-talk |
| `dwhisper serve` | Start local speech-to-text API server | Designed for external app integration |

## Model Resolution

Built-in model aliases:

- `whisper`
- `whisper:tiny`
- `whisper:base`
- `whisper:small`
- `whisper:medium`
- `whisper:large-v3`
- `whisper:large-v3-turbo`
- `whisper-quantized`
- `whisper-quantized:large-v3-4bit`
- `whisper-quantized:large-v3-8bit`

You can also pass:

- a full Hugging Face repo ID such as `mlx-community/whisper-base-mlx`
- an `hf.co/...` reference
- a local Whisper MLX model directory
- a local app or bundle root that contains an embedded Whisper subdirectory such as `speech_encoder/`, `audio_encoder/`, or `whisper/`

That bundle-root support matters when you want to ship or integrate a larger local app package but still expose only the embedded speech runtime to `dwhisper`.

## Output Formats

`dwhisper run` and `dwhisper transcribe` support:

- `text`
- `json`
- `srt`
- `vtt`

The API server additionally supports:

- `verbose_json`

## Profiles, Corrections, And Vocabulary

Daydream Whisper can load reusable presets and post-processing rules from files under `~/.dwhisper/`.

`profiles.yaml` example:

```yaml
default: meeting-zh

profiles:
  meeting-zh:
    description: Chinese meeting notes
    model: whisper:large-v3-turbo
    output_format: text
    transcribe:
      language: zh
      task: transcribe
      word_timestamps: true
      beam_size: 5
      best_of: 3
      hotwords:
        - Daydream
        - MLX
      vocabulary:
        helo: hello
      correction:
        capitalize_sentences: true
        ensure_terminal_punctuation: true
    listen:
      chunk_duration: 2.5
      overlap_duration: 0.5
      silence_threshold: 1.0
      vad_sensitivity: 0.6
      push_to_talk: false
```

`corrections.yaml` example:

```yaml
capitalize_sentences: true
ensure_terminal_punctuation: true
drop_hallucinations: true
regex_substitutions:
  - pattern: "\\bteh\\b"
    replacement: "the"
```

`vocabulary.yaml` example:

```yaml
vocabulary:
  mlxwhspr: mlx-whisper
  helo: hello
```

Practical precedence:

- explicit CLI flags win over profile defaults
- explicit API request fields win over profile defaults
- profile defaults fill in repeated settings
- config and default correction files provide fallback values

Useful options for quality tuning:

- `--language`
- `--task`
- `--word-timestamps`
- `--beam-size`
- `--best-of`
- `--no-speech-threshold`
- `--hotword`
- `--vocabulary-entry SOURCE=TARGET`
- `--corrections-file`
- `--vocabulary-file`

Useful options for optional local text-model post-processing:

- `--postprocess`
- `--postprocess-model`
- `--postprocess-base-url`
- `--postprocess-api-key`
- `--postprocess-mode`
- `--postprocess-prompt`
- `--postprocess-timeout`

## Realtime Listening

`dwhisper listen` is designed for local dictation and short-latency transcription.

Important controls:

- `--device` selects a specific input device
- `--sample-rate` controls capture rate
- `--chunk-duration` controls how much audio is accumulated before a decode window
- `--overlap` keeps a shared audio margin between windows and helps prevent word-boundary loss
- `--silence-threshold` controls how long silence must last before a segment is flushed
- `--vad-sensitivity` controls how aggressively speech is detected
- `--push-to-talk` disables VAD gating and lets you manually gate capture

Recommended starting points:

- meetings or lectures: `--chunk-duration 3.0 --overlap 0.5`
- short dictation: `--chunk-duration 1.5 --overlap 0.25`
- noisy room: raise `--silence-threshold` slightly and lower `--vad-sensitivity`

## API Server

`dwhisper serve` exposes a local HTTP service for other transcription apps.

Available endpoints:

- `GET /health`
- `GET /ready`
- `GET /metrics`
- `GET /v1/models`
- `POST /v1/audio/transcriptions`
- `POST /v1/audio/translations`

Useful serve options:

- `--model`
- `--host`
- `--port`
- `--auto-pull`
- `--max-concurrency`
- `--request-timeout`
- `--max-request-bytes`
- `--preload`
- `--allow-origin`

Multipart request example:

```bash
curl http://127.0.0.1:11434/v1/audio/transcriptions \
  -H "X-Request-ID: demo-123" \
  -F "file=@./samples/interview.wav" \
  -F "model=whisper:base" \
  -F "response_format=verbose_json"
```

JSON request example using a local file path and a profile:

```bash
curl http://127.0.0.1:11434/v1/audio/transcriptions \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: demo-123" \
  -d '{
    "audio_path": "/absolute/path/to/interview.wav",
    "profile": "meeting-zh",
    "response_format": "verbose_json",
    "hotwords": ["Daydream", "MLX"],
    "vocabulary": {"helo": "hello"},
    "correction": {"capitalize_sentences": true}
  }'
```

JSON request example with optional local Qwen cleanup after Whisper:

```bash
curl http://127.0.0.1:11434/v1/audio/transcriptions \
  -H "Content-Type: application/json" \
  -d '{
    "audio_path": "/absolute/path/to/interview.wav",
    "postprocess": true,
    "postprocess_model": "qwen3.5:0.8b",
    "postprocess_base_url": "http://127.0.0.1:11435/v1",
    "postprocess_mode": "clean"
  }'
```

The server is designed for smoother app integration:

- it keeps a persistent transcription worker per loaded model
- it exposes readiness and lightweight Prometheus-style metrics
- it returns `X-Request-ID`, active request, and processing-time headers
- it supports CORS through `--allow-origin`
- it caps concurrency so repeated requests do not stampede the runtime

## Configuration

Persistent config lives in `~/.dwhisper/config.yaml`.

Example:

```yaml
model: whisper:large-v3-turbo

transcribe:
  profile: null
  language: null
  task: transcribe
  output_format: text
  word_timestamps: false

correction:
  corrections_file: ~/.dwhisper/corrections.yaml
  vocabulary_file: ~/.dwhisper/vocabulary.yaml

postprocess:
  enabled: false
  model: qwen3.5:0.8b
  base_url: http://127.0.0.1:11435/v1
  api_key: dwhisper-local
  mode: clean
  timeout: 30

profiles:
  file: ~/.dwhisper/profiles.yaml

audio:
  sample_rate: 16000
  device: null

listen:
  chunk_duration: 3.0
  overlap_duration: 0.5
  silence_threshold: 1.0
  vad_sensitivity: 0.6
  push_to_talk: false

serve:
  host: 127.0.0.1
  port: 11434
  max_concurrency: 2
  request_timeout: 120
  max_request_bytes: 52428800
  preload: false
  allow_origin: "*"
```

Environment variable overrides:

- `DWHISPER_MODEL`
- `DWHISPER_LANGUAGE`
- `DWHISPER_PROFILE`
- `DWHISPER_TASK`
- `DWHISPER_OUTPUT_FORMAT`
- `DWHISPER_WORD_TIMESTAMPS`
- `DWHISPER_POSTPROCESS`
- `DWHISPER_POSTPROCESS_MODEL`
- `DWHISPER_POSTPROCESS_BASE_URL`
- `DWHISPER_POSTPROCESS_API_KEY`
- `DWHISPER_POSTPROCESS_MODE`
- `DWHISPER_POSTPROCESS_TIMEOUT`
- `DWHISPER_CORRECTIONS_FILE`
- `DWHISPER_VOCABULARY_FILE`
- `DWHISPER_PROFILES_FILE`
- `DWHISPER_AUDIO_DEVICE`
- `DWHISPER_SAMPLE_RATE`
- `DWHISPER_CHUNK_DURATION`
- `DWHISPER_OVERLAP_DURATION`
- `DWHISPER_SILENCE_THRESHOLD`
- `DWHISPER_VAD_SENSITIVITY`
- `DWHISPER_PUSH_TO_TALK`
- `DWHISPER_TRANSCRIBE_TIMEOUT`
- `DWHISPER_HOST`
- `DWHISPER_PORT`
- `DWHISPER_SERVE_MAX_CONCURRENCY`
- `DWHISPER_SERVE_REQUEST_TIMEOUT`
- `DWHISPER_SERVE_MAX_REQUEST_BYTES`
- `DWHISPER_SERVE_PRELOAD`
- `DWHISPER_SERVE_ALLOW_ORIGIN`

## Stability Notes

- Runtime inference is isolated in a subprocess so native MLX crashes do not take down the main CLI process.
- The API server reuses model workers so repeated requests are smoother than cold-starting the runtime each time.
- Default correction and vocabulary files can be applied automatically, which reduces repeated typo cleanup in production dictation flows.
- Optional post-processing stays off by default and runs through a separate local text-model endpoint, so Whisper recognition and Qwen cleanup remain decoupled.
- Local bundle-root resolution avoids a class of “model not found” errors when the runtime is embedded inside another package.

## Troubleshooting

- `ModuleNotFoundError` for `rich`, `yaml`, or `sounddevice` usually means you are using the wrong Python interpreter. Use `./dwhisper` or activate the Daydream Whisper virtual environment.
- If `python3 -m dwhisper` fails but `./dwhisper` works, your shell `python3` is pointing somewhere else.
- If microphone capture fails, verify `portaudio` is installed and macOS microphone permissions are granted.
- If MP3 or M4A decoding fails, verify `ffmpeg` is installed and reachable in `PATH`.
- If a local app keeps timing out on first request, use `dwhisper serve --preload`.
- If you need browser-based local integration, set `--allow-origin http://localhost:3000` or another exact origin instead of `*`.

## 中文

Daydream Whisper 现在已经不是文本大模型 CLI，而是一个专注于本地语音识别、实时听写和语音转文字服务化接入的 Apple Silicon 工具。项目名仍然叫 `Daydream Whisper`，但可执行命令统一改成 `dwhisper`，专门用来和原本的 `Daydream CLI` 做隔离，避免 `daydream run`、`daydream serve`、`daydream cp` 这类命令冲突。

### 现在支持什么

- 本地 MLX Whisper 模型转录
- 音频文件转 `text`、`json`、`srt`、`vtt`
- 实时麦克风听写
- 命名 profile 复用转录参数
- 热词、词表、正则清洗、标点补全等自动纠错
- 本地 HTTP API，便于接入听写、语音记录、会议纪要类应用
- 本地 bundle 根目录模型解析，不要求你一定传裸 Whisper 目录
- Whisper 转录完成后，可选再交给本地 Qwen 或其他文本模型做清洗、摘要或会议纪要整理

### 为什么命令一定是 `dwhisper`

- 避免和 Daydream CLI 的 `daydream` 命令冲突
- 安装脚本只创建 `dwhisper`
- 卸载脚本只删除 Daydream Whisper 自己的目录、自己的启动器和自己的 `~/.dwhisper`
- 不会主动删除你原有的 `daydream`、系统 Python、Homebrew 包、其他虚拟环境和共享 Hugging Face 缓存

### 安装

交互式安装：

```bash
curl -fsSL https://raw.githubusercontent.com/Starry331/Daydream-Whisper/main/install.sh -o /tmp/dwhisper-install.sh
zsh /tmp/dwhisper-install.sh
```

安装脚本会做这些事：

- 在有 Homebrew 时安装 `portaudio` 和 `ffmpeg`
- 克隆或更新仓库到 `~/Daydream-Whisper`
- 在 `~/Daydream-Whisper/.venv` 里重建独立虚拟环境
- 安装 `dwhisper`
- 在 `~/.local/bin` 放一个 `dwhisper` 启动器
- 默认拉取 `whisper:base`

手动安装：

```bash
git clone https://github.com/Starry331/Daydream-Whisper.git
cd Daydream-Whisper
python3.14 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -e .
```

如果你只是临时从仓库目录运行，直接用：

```bash
./dwhisper --help
```

如果你的 `python3` 指向别的虚拟环境，优先用 `./dwhisper`，不要直接假设 `python3 -m dwhisper` 一定能走到这套环境。

### 卸载

```bash
curl -fsSL https://raw.githubusercontent.com/Starry331/Daydream-Whisper/main/uninstall.sh -o /tmp/dwhisper-uninstall.sh
zsh /tmp/dwhisper-uninstall.sh
```

卸载脚本是保守策略：

- 只删除 Daydream Whisper 的代码目录
- 只删除明确指向 Daydream Whisper 的 `dwhisper` 启动器
- 只删除默认 `~/.dwhisper` 或明确属于本项目的自定义 home
- 不会删除 `ffmpeg`、`portaudio`、系统 Python、用户原本已有的虚拟环境、Hugging Face 共享缓存

### 快速开始

常用命令：

```bash
dwhisper models
dwhisper profiles
dwhisper pull whisper:base
dwhisper list
dwhisper show whisper:base
dwhisper run ./audio.wav
dwhisper run ./audio.wav --profile meeting-zh
dwhisper run ./audio.wav --format srt -o audio.srt
dwhisper devices
dwhisper listen
dwhisper listen --push-to-talk
dwhisper serve --model whisper:base --port 11434
```

`dwhisper run` 就是 `dwhisper transcribe` 的便捷别名。

如果你想给某类场景固定一整套参数，可以先写 profile，再直接套用：

```bash
dwhisper run ./meeting.wav --profile meeting-zh
dwhisper listen --profile meeting-zh
```

如果你想在转录阶段增强术语命中和后处理：

```bash
dwhisper run ./demo.wav \
  --beam-size 5 \
  --best-of 3 \
  --hotword Daydream \
  --hotword MLX \
  --vocabulary-entry helo=hello \
  --corrections-file ~/.dwhisper/corrections.yaml \
  --vocabulary-file ~/.dwhisper/vocabulary.yaml
```

如果你想在 Whisper 完成后，再让本地 Qwen 做文本后处理：

```bash
dwhisper run ./meeting.wav \
  --postprocess \
  --postprocess-model qwen3.5:0.8b \
  --postprocess-base-url http://127.0.0.1:11435/v1 \
  --postprocess-mode clean
```

### 命令说明

| 命令 | 用途 | 说明 |
| --- | --- | --- |
| `dwhisper pull <model>` | 下载模型或接入本地模型 | 支持别名、HF repo、以及本地路径 |
| `dwhisper list` | 查看已下载和已发现的模型 | 包含本地注册项 |
| `dwhisper rm <model>` | 删除本地缓存模型 | 可加 `--force` |
| `dwhisper show <model>` | 查看模型元数据 | 适合排查模型目录问题 |
| `dwhisper models` | 列出内置别名和注册表 | 会显示来源是本地还是 Hugging Face |
| `dwhisper profiles` | 列出命名 profile | 读取 `~/.dwhisper/profiles.yaml` |
| `dwhisper run <audio>` | 转录文件 | `transcribe` 的别名 |
| `dwhisper transcribe <audio>` | 转录文件 | 支持文字、JSON、SRT、VTT |
| `dwhisper devices` | 列出输入设备 | 给 `listen --device` 用 |
| `dwhisper listen` | 实时麦克风听写 | 支持 VAD 和按键说话 |
| `dwhisper serve` | 启动本地语音 API 服务 | 给外部应用接入 |

### 本地模型与别名

内置别名：

- `whisper`
- `whisper:tiny`
- `whisper:base`
- `whisper:small`
- `whisper:medium`
- `whisper:large-v3`
- `whisper:large-v3-turbo`
- `whisper-quantized`
- `whisper-quantized:large-v3-4bit`
- `whisper-quantized:large-v3-8bit`

除了这些别名，还支持：

- `mlx-community/whisper-base-mlx` 这种完整 Hugging Face repo
- `hf.co/...` 形式引用
- 本地 Whisper MLX 模型目录
- 更大的本地应用或 bundle 根目录，只要里面嵌有 `speech_encoder/`、`audio_encoder/`、`whisper/` 这类语音子目录

这点对本地打包应用接入很重要。你不一定要把语音模型单独拆出来，只要 bundle 内有可识别的语音 runtime 目录，`dwhisper` 就能解析进去。

### Profile、纠错和词表

Daydream Whisper 约定这些文件：

- `~/.dwhisper/profiles.yaml`
- `~/.dwhisper/corrections.yaml`
- `~/.dwhisper/vocabulary.yaml`

`profiles.yaml` 示例：

```yaml
default: meeting-zh

profiles:
  meeting-zh:
    description: 中文会议记录
    model: whisper:large-v3-turbo
    output_format: text
    transcribe:
      language: zh
      task: transcribe
      word_timestamps: true
      beam_size: 5
      best_of: 3
      hotwords:
        - Daydream
        - MLX
      vocabulary:
        helo: hello
      correction:
        capitalize_sentences: true
        ensure_terminal_punctuation: true
    listen:
      chunk_duration: 2.5
      overlap_duration: 0.5
      silence_threshold: 1.0
      vad_sensitivity: 0.6
      push_to_talk: false
```

`corrections.yaml` 示例：

```yaml
capitalize_sentences: true
ensure_terminal_punctuation: true
drop_hallucinations: true
regex_substitutions:
  - pattern: "\\bteh\\b"
    replacement: "the"
```

`vocabulary.yaml` 示例：

```yaml
vocabulary:
  mlxwhspr: mlx-whisper
  helo: hello
```

优先级可以这样理解：

- 显式 CLI 参数优先级最高
- 显式 API 请求字段优先级最高
- profile 用于补齐重复场景的默认参数
- `config.yaml` 和默认纠错文件提供兜底值

如果你在做会议纪要、医学术语、品牌词、产品名等场景，这套 `profile + corrections + vocabulary` 会比每次手工改字稳定很多。

如果你已经有本地 Qwen 服务，这一层还可以继续往后接：

- `clean`：清洗转录文本但不改原意
- `summary`：输出摘要
- `meeting-notes`：输出结构化会议纪要
- `speaker-format`：只整理已有说话人标签，不凭空编造说话人

### 实时听写调优

`dwhisper listen` 的核心参数：

- `--device` 指定输入设备
- `--sample-rate` 控制采样率
- `--chunk-duration` 控制每次送去识别的窗口大小
- `--overlap` 控制窗口重叠，减少词边界丢失
- `--silence-threshold` 控制静音多久后刷出一段结果
- `--vad-sensitivity` 控制语音检测激进程度
- `--push-to-talk` 关闭 VAD 门控，改为手动按键开关录音

建议起点：

- 会议或讲座：`--chunk-duration 3.0 --overlap 0.5`
- 短句听写：`--chunk-duration 1.5 --overlap 0.25`
- 嘈杂环境：适当提高 `--silence-threshold`，适当降低 `--vad-sensitivity`

### 本地 API 服务

`dwhisper serve` 是为了给其他语音转文字、听写、速记、会议记录类应用接本地模型。

可用路由：

- `GET /health`
- `GET /ready`
- `GET /metrics`
- `GET /v1/models`
- `POST /v1/audio/transcriptions`
- `POST /v1/audio/translations`

常用服务参数：

- `--model`
- `--host`
- `--port`
- `--auto-pull`
- `--max-concurrency`
- `--request-timeout`
- `--max-request-bytes`
- `--preload`
- `--allow-origin`

上传文件方式：

```bash
curl http://127.0.0.1:11434/v1/audio/transcriptions \
  -H "X-Request-ID: demo-123" \
  -F "file=@./samples/interview.wav" \
  -F "model=whisper:base" \
  -F "response_format=verbose_json"
```

本地文件路径 JSON 请求方式：

```bash
curl http://127.0.0.1:11434/v1/audio/transcriptions \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: demo-123" \
  -d '{
    "audio_path": "/absolute/path/to/interview.wav",
    "profile": "meeting-zh",
    "response_format": "verbose_json",
    "hotwords": ["Daydream", "MLX"],
    "vocabulary": {"helo": "hello"},
    "correction": {"capitalize_sentences": true}
  }'
```

如果要让服务端在 Whisper 后面再串一个本地 Qwen，可在请求里加：

```json
{
  "postprocess": true,
  "postprocess_model": "qwen3.5:0.8b",
  "postprocess_base_url": "http://127.0.0.1:11435/v1",
  "postprocess_mode": "clean"
}
```

这个服务端为了接外部应用更平滑，做了这些事：

- 每个模型维持一个常驻转录 worker，减少重复请求抖动
- 支持 `/ready` 和 `/metrics`，方便健康检查和轻量监控
- 返回 `X-Request-ID`、并发状态和处理时长相关 header
- 支持 `--allow-origin` 做本地浏览器应用跨域接入
- 有并发上限，避免外部请求把底层 runtime 冲垮

如果你要降低首个请求延迟，建议：

```bash
dwhisper serve --model whisper:base --preload
```

### 配置文件

默认配置在 `~/.dwhisper/config.yaml`。

示例：

```yaml
model: whisper:large-v3-turbo

transcribe:
  profile: null
  language: null
  task: transcribe
  output_format: text
  word_timestamps: false

correction:
  corrections_file: ~/.dwhisper/corrections.yaml
  vocabulary_file: ~/.dwhisper/vocabulary.yaml

postprocess:
  enabled: false
  model: qwen3.5:0.8b
  base_url: http://127.0.0.1:11435/v1
  api_key: dwhisper-local
  mode: clean
  timeout: 30

profiles:
  file: ~/.dwhisper/profiles.yaml

audio:
  sample_rate: 16000
  device: null

listen:
  chunk_duration: 3.0
  overlap_duration: 0.5
  silence_threshold: 1.0
  vad_sensitivity: 0.6
  push_to_talk: false

serve:
  host: 127.0.0.1
  port: 11434
  max_concurrency: 2
  request_timeout: 120
  max_request_bytes: 52428800
  preload: false
  allow_origin: "*"
```

常用环境变量：

- `DWHISPER_MODEL`
- `DWHISPER_LANGUAGE`
- `DWHISPER_PROFILE`
- `DWHISPER_OUTPUT_FORMAT`
- `DWHISPER_POSTPROCESS`
- `DWHISPER_POSTPROCESS_MODEL`
- `DWHISPER_POSTPROCESS_BASE_URL`
- `DWHISPER_POSTPROCESS_API_KEY`
- `DWHISPER_POSTPROCESS_MODE`
- `DWHISPER_POSTPROCESS_TIMEOUT`
- `DWHISPER_CORRECTIONS_FILE`
- `DWHISPER_VOCABULARY_FILE`
- `DWHISPER_PROFILES_FILE`
- `DWHISPER_AUDIO_DEVICE`
- `DWHISPER_SAMPLE_RATE`
- `DWHISPER_CHUNK_DURATION`
- `DWHISPER_OVERLAP_DURATION`
- `DWHISPER_SILENCE_THRESHOLD`
- `DWHISPER_VAD_SENSITIVITY`
- `DWHISPER_PUSH_TO_TALK`
- `DWHISPER_TRANSCRIBE_TIMEOUT`
- `DWHISPER_HOST`
- `DWHISPER_PORT`
- `DWHISPER_SERVE_MAX_CONCURRENCY`
- `DWHISPER_SERVE_REQUEST_TIMEOUT`
- `DWHISPER_SERVE_MAX_REQUEST_BYTES`
- `DWHISPER_SERVE_PRELOAD`
- `DWHISPER_SERVE_ALLOW_ORIGIN`

### 稳定性与排障

- 推理放在子进程里执行，原生 MLX 崩溃不会直接把主 CLI 进程一起带死
- 服务端会复用模型 worker，比每次冷启动更平滑
- 默认纠错和词表文件可以自动套用，减少重复错字清洗
- 可选的 Qwen 文本后处理默认关闭，而且是通过独立本地接口接入的，不会把 Whisper 主识别链路和文本模型重新耦死
- 本地 bundle root 解析减少了“模型路径存在但识别不到”的问题

常见问题：

- 如果报 `ModuleNotFoundError: rich`、`yaml`、`sounddevice`，通常是你用了错误的 Python 解释器。优先用 `./dwhisper` 或激活项目自己的 `.venv`。
- 如果 `python3 -m dwhisper` 不通，但 `./dwhisper` 可用，说明你 shell 里的 `python3` 指到了别的虚拟环境。
- 如果麦克风不可用，先检查 `portaudio` 和 macOS 麦克风权限。
- 如果 MP3/M4A 无法解码，先检查 `ffmpeg` 是否已安装并在 `PATH` 里。
- 如果本地应用首个请求超时，优先用 `dwhisper serve --preload`。
- 如果你要给浏览器本地页面调用，别默认用 `*`，直接设成明确来源，例如 `--allow-origin http://localhost:3000`。

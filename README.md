# Daydream Whisper

![Daydream Whisper logo](docs/images/daydream-whisper-logo.svg)

English | [中文](#中文)

Daydream Whisper is an Apple Silicon speech recognition CLI built around MLX Whisper. The project is now speech-first: local audio transcription, realtime microphone listening, subtitle export, and configurable voice workflows.

## Status

Current release: `v0.2.0`

Implemented commands:

- `dwhisper pull`
- `dwhisper list`
- `dwhisper rm`
- `dwhisper show`
- `dwhisper models`
- `dwhisper run`
- `dwhisper transcribe`
- `dwhisper devices`
- `dwhisper listen`
- `dwhisper serve`

## What It Does

- Runs local Whisper models through MLX on Apple Silicon
- Transcribes audio files to `text`, `json`, `srt`, or `vtt`
- Supports realtime microphone transcription
- Serves an OpenAI-style speech-to-text API for external dictation or note-taking apps
- Supports built-in aliases like `whisper:base` and `whisper:large-v3-turbo`
- Lets you override model, language, task, VAD, chunking, and device settings via CLI, config, or environment variables

## Requirements

- Apple Silicon Mac
- Python `3.14+`
- `ffmpeg` for formats like MP3 and M4A
- `portaudio` for microphone capture through `sounddevice`

## Installation

Interactive install:

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

The installer only creates `dwhisper`. It does not install, overwrite, or remove the original `daydream` command from Daydream CLI.

Manual install:

```bash
git clone https://github.com/Starry331/Daydream-Whisper.git
cd Daydream-Whisper
python3.14 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -e .
```

Use a dedicated environment for Daydream Whisper. Do not install it into an existing Daydream CLI virtual environment.

Directly from a repo checkout:

```bash
./dwhisper --help
```

After editable install, module entry also works:

```bash
python3 -m dwhisper --help
```

## Quick Start

Show built-in and registered models:

```bash
dwhisper models
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

Export subtitles:

```bash
dwhisper run ./samples/interview.wav --format srt -o interview.srt
dwhisper run ./samples/interview.wav --format vtt -o interview.vtt
```

List microphone devices:

```bash
dwhisper devices
```

Start realtime listening:

```bash
dwhisper listen --model whisper:base
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

Available endpoints:

- `GET /health`
- `GET /v1/models`
- `POST /v1/audio/transcriptions`
- `POST /v1/audio/translations`

Example request:

```bash
curl http://127.0.0.1:11434/v1/audio/transcriptions \
  -F "file=@./samples/interview.wav" \
  -F "model=whisper:base" \
  -F "response_format=verbose_json"
```

The server keeps a persistent transcription worker per loaded model, so repeated API calls from other speech apps do not pay full worker startup cost on every request.

## Built-In Model Aliases

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

## Output Formats

`dwhisper run` and `dwhisper transcribe` support:

- `text`
- `json`
- `srt`
- `vtt`

## Configuration

Persistent config lives in `~/.dwhisper/config.yaml`.

Example:

```yaml
model: whisper:large-v3-turbo

transcribe:
  language: null
  task: transcribe
  output_format: text
  word_timestamps: false

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
```

Environment variable overrides:

- `DWHISPER_MODEL`
- `DWHISPER_LANGUAGE`
- `DWHISPER_TASK`
- `DWHISPER_OUTPUT_FORMAT`
- `DWHISPER_WORD_TIMESTAMPS`
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

## Notes

- This project is no longer a text-generation CLI.
- GGUF models are not supported.
- Realtime transcription depends on the installed `mlx-whisper` and `sounddevice` packages.
- Decoding compressed audio formats depends on a working `ffmpeg` installation.
- Runtime inference is isolated in a subprocess so native MLX crashes do not take down the main CLI process.

## 中文

Daydream Whisper 现在已经不是文本大模型 CLI，而是一个专注于本地语音识别的 Apple Silicon 工具。核心目标是：

- 只保留 Whisper / MLX 语音模型能力
- 提供稳定的本地音频文件转录
- 提供实时麦克风听写
- 支持字幕导出和高自定义配置

常用命令：

```bash
dwhisper pull whisper:base
dwhisper list
dwhisper show whisper:base
dwhisper run ./audio.wav
dwhisper run ./audio.wav --format srt -o audio.srt
dwhisper devices
dwhisper listen
dwhisper listen --push-to-talk
dwhisper serve --model whisper:base --port 11434
```

`dwhisper serve` 现在会暴露一个面向语音转文字应用的本地 HTTP API，包含 `/v1/audio/transcriptions`、`/v1/audio/translations` 和 `/v1/models`。服务端会为每个已加载模型保留常驻转录 worker，减少外部听写应用重复请求时的启动抖动；如果你希望首个请求也尽量快，可以加 `--preload`。配置文件在 `~/.dwhisper/config.yaml`，可以设置默认模型、语言、任务模式、音频设备、采样率、分段时长、重叠时长、静音阈值，以及 serve 的 host、port、并发数、请求大小限制和预加载开关。所有核心选项也都支持 `DWHISPER_*` 环境变量覆盖。

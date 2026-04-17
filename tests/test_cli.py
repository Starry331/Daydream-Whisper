from __future__ import annotations

import os
import subprocess
import sys
import unittest
from unittest import mock

from click.testing import CliRunner

from dwhisper.cli import cli
from dwhisper.doctor import DoctorCheck


class CliTests(unittest.TestCase):
    def test_module_entrypoints_use_dwhisper_and_block_daydream(self) -> None:
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [
                os.path.abspath("src"),
                env.get("PYTHONPATH", ""),
            ]
        ).rstrip(os.pathsep)

        dwhisper_result = subprocess.run(
            [sys.executable, "-m", "dwhisper", "--version"],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        legacy_result = subprocess.run(
            [sys.executable, "-m", "daydream"],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )

        self.assertEqual(dwhisper_result.returncode, 0, dwhisper_result.stderr)
        self.assertIn("dwhisper", dwhisper_result.stdout.lower())
        self.assertNotEqual(legacy_result.returncode, 0)
        self.assertIn("Use 'dwhisper' or 'python -m dwhisper' instead.", legacy_result.stderr + legacy_result.stdout)

    def test_version_uses_dwhisper_prog_name(self) -> None:
        runner = CliRunner()
        invoke = runner.invoke(cli, ["--version"])

        self.assertEqual(invoke.exit_code, 0, invoke.output)
        self.assertIn("dwhisper", invoke.output.lower())

    def test_transcribe_command_uses_whisper_transcriber(self) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem():
            with open("sample.wav", "wb") as handle:
                handle.write(b"RIFF0000WAVE")

            with mock.patch("dwhisper.config.ensure_home"), \
                mock.patch("dwhisper.cli._load_selected_profile", return_value=None), \
                mock.patch("dwhisper.transcriber.WhisperTranscriber") as transcriber_cls:
                transcriber = transcriber_cls.return_value
                result = mock.Mock()
                result.render.return_value = "hello world"
                result.duration = 1.2
                result.processing_time = 0.4
                transcriber.transcribe_file.return_value = result

                invoke = runner.invoke(cli, ["transcribe", "sample.wav", "--model", "whisper:base"])

        self.assertEqual(invoke.exit_code, 0, invoke.output)
        transcriber_cls.assert_called_once_with("whisper:base")
        transcriber.transcribe_file.assert_called_once()
        self.assertIn("hello world", invoke.output)

    def test_run_alias_uses_transcribe_path(self) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem():
            with open("sample.wav", "wb") as handle:
                handle.write(b"RIFF0000WAVE")

            with mock.patch("dwhisper.config.ensure_home"), \
                mock.patch("dwhisper.cli._load_selected_profile", return_value=None), \
                mock.patch("dwhisper.transcriber.WhisperTranscriber") as transcriber_cls:
                transcriber = transcriber_cls.return_value
                result = mock.Mock()
                result.render.return_value = "alias output"
                result.duration = 1.0
                result.processing_time = 0.2
                transcriber.transcribe_file.return_value = result

                invoke = runner.invoke(cli, ["run", "sample.wav", "--model", "whisper:base"])

        self.assertEqual(invoke.exit_code, 0, invoke.output)
        transcriber_cls.assert_called_once_with("whisper:base")
        transcriber.transcribe_file.assert_called_once()
        self.assertIn("alias output", invoke.output)

    def test_transcribe_profile_applies_profile_defaults(self) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem():
            with open("sample.wav", "wb") as handle:
                handle.write(b"RIFF0000WAVE")

            with mock.patch("dwhisper.config.ensure_home"), \
                mock.patch("dwhisper.cli._load_selected_profile") as load_profile, \
                mock.patch("dwhisper.transcriber.WhisperTranscriber") as transcriber_cls:
                profile = mock.Mock()
                profile.name = "meeting-zh"
                profile.model = "whisper:large-v3-turbo"
                profile.output_format = "text"
                profile.transcribe = {
                    "language": "zh",
                    "hotwords": ["Daydream"],
                    "vocabulary": {"helo": "hello"},
                    "postprocess": True,
                    "postprocess_model": "local-mm-model",
                    "postprocess_base_url": "http://127.0.0.1:11435/v1",
                }
                load_profile.return_value = profile
                transcriber = transcriber_cls.return_value
                result = mock.Mock()
                result.render.return_value = "hello"
                result.duration = 1.0
                result.processing_time = 0.2
                transcriber.transcribe_file.return_value = result

                invoke = runner.invoke(cli, ["transcribe", "sample.wav", "--profile", "meeting-zh"])

        self.assertEqual(invoke.exit_code, 0, invoke.output)
        transcriber_cls.assert_called_once_with("whisper:large-v3-turbo")
        options = transcriber.transcribe_file.call_args.kwargs["options"]
        self.assertEqual(options.profile, "meeting-zh")
        self.assertEqual(options.language, "zh")
        self.assertEqual(options.hotwords, ["Daydream"])
        self.assertEqual(options.vocabulary, {"helo": "hello"})
        self.assertTrue(options.postprocess)
        self.assertEqual(options.postprocess_model, "local-mm-model")
        self.assertEqual(options.postprocess_base_url, "http://127.0.0.1:11435/v1")

    def test_transcribe_accepts_explicit_postprocess_flags(self) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem():
            with open("sample.wav", "wb") as handle:
                handle.write(b"RIFF0000WAVE")

            with mock.patch("dwhisper.config.ensure_home"), \
                mock.patch("dwhisper.cli._load_selected_profile", return_value=None), \
                mock.patch("dwhisper.transcriber.WhisperTranscriber") as transcriber_cls:
                transcriber = transcriber_cls.return_value
                result = mock.Mock()
                result.render.return_value = "cleaned text"
                result.duration = 1.0
                result.processing_time = 0.2
                transcriber.transcribe_file.return_value = result

                invoke = runner.invoke(
                    cli,
                    [
                        "transcribe",
                        "sample.wav",
                        "--postprocess",
                        "--postprocess-model",
                        "local-mm-model",
                        "--post-url",
                        "http://127.0.0.1:11435/v1",
                        "--post-mode",
                        "summary",
                    ],
                )

        self.assertEqual(invoke.exit_code, 0, invoke.output)
        options = transcriber.transcribe_file.call_args.kwargs["options"]
        self.assertTrue(options.postprocess)
        self.assertEqual(options.postprocess_model, "local-mm-model")
        self.assertEqual(options.postprocess_base_url, "http://127.0.0.1:11435/v1")
        self.assertEqual(options.postprocess_mode, "summary")

    def test_transcribe_shortcut_enables_generic_postprocess_model(self) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem():
            with open("sample.wav", "wb") as handle:
                handle.write(b"RIFF0000WAVE")

            with mock.patch("dwhisper.config.ensure_home"), \
                mock.patch("dwhisper.cli._load_selected_profile", return_value=None), \
                mock.patch("dwhisper.transcriber.WhisperTranscriber") as transcriber_cls:
                transcriber = transcriber_cls.return_value
                result = mock.Mock()
                result.render.return_value = "cleaned text"
                result.duration = 1.0
                result.processing_time = 0.2
                transcriber.transcribe_file.return_value = result

                invoke = runner.invoke(
                    cli,
                    [
                        "transcribe",
                        "sample.wav",
                        "--post-model",
                        "glm-4.1v",
                    ],
                )

        self.assertEqual(invoke.exit_code, 0, invoke.output)
        options = transcriber.transcribe_file.call_args.kwargs["options"]
        self.assertTrue(options.postprocess)
        self.assertEqual(options.postprocess_model, "glm-4.1v")
        self.assertEqual(options.postprocess_base_url, "http://127.0.0.1:11435/v1")

    def test_transcribe_legacy_shortcut_flag_still_works(self) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem():
            with open("sample.wav", "wb") as handle:
                handle.write(b"RIFF0000WAVE")

            with mock.patch("dwhisper.config.ensure_home"), \
                mock.patch("dwhisper.cli._load_selected_profile", return_value=None), \
                mock.patch("dwhisper.transcriber.WhisperTranscriber") as transcriber_cls:
                transcriber = transcriber_cls.return_value
                result = mock.Mock()
                result.render.return_value = "cleaned text"
                result.duration = 1.0
                result.processing_time = 0.2
                transcriber.transcribe_file.return_value = result

                invoke = runner.invoke(
                    cli,
                    [
                        "transcribe",
                        "sample.wav",
                        "--with-postprocess-model",
                        "llava-next",
                    ],
                )

        self.assertEqual(invoke.exit_code, 0, invoke.output)
        options = transcriber.transcribe_file.call_args.kwargs["options"]
        self.assertTrue(options.postprocess)
        self.assertEqual(options.postprocess_model, "llava-next")
        self.assertEqual(options.postprocess_base_url, "http://127.0.0.1:11435/v1")

    def test_listen_command_builds_realtime_config(self) -> None:
        runner = CliRunner()
        with mock.patch("dwhisper.config.ensure_home"), \
            mock.patch("dwhisper.cli._load_selected_profile", return_value=None), \
            mock.patch("dwhisper.realtime.run_listen_session") as run_listen_session:
            invoke = runner.invoke(
                cli,
                [
                    "listen",
                    "--model",
                    "whisper:base",
                    "--language",
                    "zh",
                    "--chunk-duration",
                    "2.5",
                    "--overlap",
                    "0.2",
                    "--push-to-talk",
                ],
            )

        self.assertEqual(invoke.exit_code, 0, invoke.output)
        kwargs = run_listen_session.call_args.kwargs
        self.assertEqual(kwargs["model"], "whisper:base")
        self.assertEqual(kwargs["transcribe_options"].language, "zh")
        self.assertEqual(kwargs["realtime_config"].chunk_duration, 2.5)
        self.assertEqual(kwargs["realtime_config"].overlap_duration, 0.2)
        self.assertTrue(kwargs["realtime_config"].push_to_talk)

    def test_devices_command_renders_audio_devices(self) -> None:
        runner = CliRunner()
        with mock.patch("dwhisper.config.ensure_home"), mock.patch(
            "dwhisper.audio.list_audio_devices",
            return_value=[
                {
                    "index": 0,
                    "name": "Built-in Microphone",
                    "max_input_channels": 1,
                    "default_samplerate": 16000,
                    "is_default": True,
                }
            ],
        ):
            invoke = runner.invoke(cli, ["devices"])

        self.assertEqual(invoke.exit_code, 0, invoke.output)
        self.assertIn("Built-in Microphone", invoke.output)
        self.assertIn("16000", invoke.output)

    def test_serve_command_invokes_speech_server(self) -> None:
        runner = CliRunner()
        with mock.patch("dwhisper.config.ensure_home"), mock.patch("dwhisper.server.start_server") as start_server:
            invoke = runner.invoke(
                cli,
                [
                    "serve",
                    "--model",
                    "whisper:base",
                    "--host",
                    "0.0.0.0",
                    "--port",
                    "9001",
                    "--auto-pull",
                    "--max-concurrency",
                    "3",
                    "--request-timeout",
                    "45",
                    "--max-request-bytes",
                    "777777",
                    "--preload",
                ],
            )

        self.assertEqual(invoke.exit_code, 0, invoke.output)
        start_server.assert_called_once_with(
            model="whisper:base",
            host="0.0.0.0",
            port=9001,
            auto_pull=True,
            max_concurrency=3,
            request_timeout=45.0,
            max_request_bytes=777777,
            preload=True,
            allow_origin="*",
            postprocess_defaults={
                "postprocess": False,
                "postprocess_model": None,
                "postprocess_base_url": None,
                "postprocess_api_key": "dwhisper-local",
                "postprocess_mode": "clean",
                "postprocess_prompt": None,
                "postprocess_timeout": 30.0,
                "postprocess_backend": "auto",
            },
        )

    def test_serve_shortcut_enables_generic_postprocess_model(self) -> None:
        runner = CliRunner()
        with mock.patch("dwhisper.config.ensure_home"), mock.patch("dwhisper.server.start_server") as start_server:
            invoke = runner.invoke(
                cli,
                [
                    "serve",
                    "--post-model",
                    "qwen2.5-vl",
                ],
            )

        self.assertEqual(invoke.exit_code, 0, invoke.output)
        self.assertEqual(
            start_server.call_args.kwargs["postprocess_defaults"]["postprocess_model"],
            "qwen2.5-vl",
        )
        self.assertTrue(start_server.call_args.kwargs["postprocess_defaults"]["postprocess"])
        self.assertEqual(
            start_server.call_args.kwargs["postprocess_defaults"]["postprocess_base_url"],
            "http://127.0.0.1:11435/v1",
        )

    def test_doctor_command_renders_checks_and_summary(self) -> None:
        runner = CliRunner()
        fake_results = [
            DoctorCheck(name="Python version", status="ok", message="Python 3.14.3"),
            DoctorCheck(
                name="sounddevice / PortAudio",
                status="warn",
                message="not installed",
                hint="Install PortAudio.",
            ),
            DoctorCheck(
                name="Post-process defaults",
                status="info",
                message="disabled",
            ),
        ]

        with mock.patch("dwhisper.config.ensure_home"), mock.patch(
            "dwhisper.doctor.run_doctor",
            return_value=fake_results,
        ):
            invoke = runner.invoke(cli, ["doctor"])

        self.assertEqual(invoke.exit_code, 0, invoke.output)
        self.assertIn("Python version", invoke.output)
        self.assertIn("sounddevice / PortAudio", invoke.output)
        self.assertIn("Install PortAudio.", invoke.output)
        self.assertIn("Summary: 1 ok", invoke.output)
        self.assertIn("1 warn", invoke.output)
        self.assertIn("1 info", invoke.output)

    def test_doctor_strict_exits_non_zero_on_warning(self) -> None:
        runner = CliRunner()
        fake_results = [
            DoctorCheck(name="Cached Whisper models", status="warn", message="none found"),
        ]

        with mock.patch("dwhisper.config.ensure_home"), mock.patch(
            "dwhisper.doctor.run_doctor",
            return_value=fake_results,
        ):
            invoke = runner.invoke(cli, ["doctor", "--strict"])

        self.assertEqual(invoke.exit_code, 1, invoke.output)
        self.assertIn("Cached Whisper models", invoke.output)
        self.assertIn("none found", invoke.output)


if __name__ == "__main__":
    unittest.main()

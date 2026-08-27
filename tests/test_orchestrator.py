import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import requests

from orchestrator import (
    CapriolePromptProvider,
    MultiLLMClient,
    OllamaPromptProvider,
    PromptOrchestrator,
    PromptProviderError,
    PromptRoute,
    create_prompt_client,
)
from osc_output import RecordingOutputPublisher
from runtime_config import RuntimeConfig


class PromptRouteTests(unittest.TestCase):
    def test_parses_provider_and_preserves_colons_in_model_name(self):
        route = PromptRoute.parse(" OLLAMA:kimi-k2.6:cloud ")

        self.assertEqual(route.provider, "ollama")
        self.assertEqual(route.model, "kimi-k2.6:cloud")

    def test_rejects_malformed_and_unknown_routes(self):
        for route in ("missing-model", "ollama:", ":model", "other:model"):
            with self.subTest(route=route):
                with self.assertRaises(ValueError):
                    PromptRoute.parse(route)


class ProviderTests(unittest.TestCase):
    @staticmethod
    def response(payload):
        response = Mock()
        response.json.return_value = payload
        return response

    def test_ollama_sends_bounded_non_streaming_request(self):
        session = Mock()
        session.post.return_value = self.response({"response": "a neon city"})
        provider = OllamaPromptProvider(
            session,
            "http://localhost:11434/api/generate",
            4.5,
        )

        result = provider.generate("model-a", "private input")

        self.assertEqual(result, "a neon city")
        session.post.assert_called_once_with(
            "http://localhost:11434/api/generate",
            json={
                "model": "model-a",
                "prompt": "private input",
                "stream": False,
            },
            timeout=4.5,
        )
        session.post.return_value.raise_for_status.assert_called_once_with()

    def test_ollama_wraps_transport_and_response_errors(self):
        session = Mock()
        session.post.side_effect = requests.Timeout("slow")
        provider = OllamaPromptProvider(session, "http://ollama.test", 1.0)

        with self.assertRaises(PromptProviderError):
            provider.generate("model-a", "input")

        session.post.side_effect = None
        session.post.return_value = self.response({"response": ""})
        with self.assertRaises(PromptProviderError):
            provider.generate("model-a", "input")

    def test_capriole_sends_authorization_and_accepts_supported_shapes(self):
        shapes = (
            ({"output": "from output"}, "from output"),
            ({"response": "from response"}, "from response"),
            (
                {"choices": [{"message": {"content": "from choices"}}]},
                "from choices",
            ),
        )
        for payload, expected in shapes:
            with self.subTest(payload=payload):
                session = Mock()
                session.post.return_value = self.response(payload)
                provider = CapriolePromptProvider(
                    session,
                    "https://capriole.test/v1/chat",
                    "secret-key",
                    3.0,
                )

                result = provider.generate("model-b", "private input")

                self.assertEqual(result, expected)
                session.post.assert_called_once_with(
                    "https://capriole.test/v1/chat",
                    headers={
                        "Authorization": "Bearer secret-key",
                        "Content-Type": "application/json",
                    },
                    json={"model": "model-b", "input": "private input"},
                    timeout=3.0,
                )

    def test_capriole_rejects_missing_key_and_unrecognized_response(self):
        with self.assertRaisesRegex(ValueError, "configured API key"):
            CapriolePromptProvider(Mock(), "https://capriole.test", "", 2.0)

        session = Mock()
        session.post.return_value = self.response({"choices": []})
        provider = CapriolePromptProvider(
            session,
            "https://capriole.test",
            "secret-key",
            2.0,
        )
        with self.assertRaises(PromptProviderError):
            provider.generate("model-b", "input")


class MultiLLMClientTests(unittest.TestCase):
    def test_falls_through_to_next_model_without_logging_prompt_text(self):
        first = Mock()
        first.generate.side_effect = RuntimeError(
            "failure containing highly private spoken words"
        )
        second = Mock()
        second.generate.return_value = "  vivid   refined prompt  "
        logger = Mock()
        client = MultiLLMClient(
            (
                PromptRoute("ollama", "first-model"),
                PromptRoute("capriole", "second-model"),
            ),
            {"ollama": first, "capriole": second},
            logger=logger,
        )

        result = client.generate_prompt(
            "highly private spoken words",
            "private scene context",
        )

        self.assertEqual(result, "vivid refined prompt")
        self.assertEqual(first.generate.call_args.args[0], "first-model")
        self.assertEqual(second.generate.call_args.args[0], "second-model")
        log_calls = str(logger.method_calls)
        self.assertNotIn("highly private spoken words", log_calls)
        self.assertNotIn("private scene context", log_calls)
        events = [
            call.kwargs["extra"]["event"] for call in logger.method_calls
        ]
        self.assertEqual(
            events,
            [
                "prompt_refinement_attempt",
                "prompt_refinement_failed",
                "prompt_refinement_attempt",
                "prompt_refinement_completed",
            ],
        )

    def test_rejects_overlong_output_and_uses_the_next_route(self):
        provider = Mock()
        provider.generate.side_effect = ["x" * 11, "short"]
        client = MultiLLMClient(
            (
                PromptRoute("ollama", "large"),
                PromptRoute("ollama", "small"),
            ),
            {"ollama": provider},
            max_output_characters=10,
        )

        self.assertEqual(client.generate_prompt("input"), "short")
        self.assertEqual(provider.generate.call_count, 2)

    def test_returns_none_after_every_route_fails(self):
        provider = Mock()
        provider.generate.return_value = ""
        logger = Mock()
        client = MultiLLMClient(
            (PromptRoute("ollama", "model"),),
            {"ollama": provider},
            logger=logger,
        )

        self.assertIsNone(client.generate_prompt("input"))
        self.assertEqual(
            logger.warning.call_args.kwargs["extra"]["event"],
            "prompt_refinement_exhausted",
        )

    def test_closes_only_an_owned_http_session_and_only_once(self):
        owned_session = Mock()
        client = MultiLLMClient(
            (PromptRoute("ollama", "model"),),
            {"ollama": Mock()},
            owned_session=owned_session,
        )

        client.close()
        client.close()

        owned_session.close.assert_called_once_with()

    def test_factory_does_not_take_ownership_of_injected_session(self):
        session = Mock()
        client = create_prompt_client(RuntimeConfig(), session=session)

        client.close()

        session.close.assert_not_called()

    def test_factory_closes_an_owned_session_if_initialization_fails(self):
        session = Mock()
        config = replace(
            RuntimeConfig(),
            prompt_refinement_chain=("capriole:model",),
            capriole_api_key="",
        )

        with patch("orchestrator.requests.Session", return_value=session):
            with self.assertRaisesRegex(ValueError, "configured API key"):
                create_prompt_client(config)

        session.close.assert_called_once_with()


class PromptOrchestratorTests(unittest.TestCase):
    def test_publishes_once_and_suppresses_a_duplicate(self):
        llm_client = Mock()
        llm_client.generate_prompt.return_value = "refined prompt"
        publisher = RecordingOutputPublisher()
        orchestrator = PromptOrchestrator(llm_client, publisher)

        first = orchestrator.refine_and_send("raw text")
        second = orchestrator.refine_and_send("raw text")

        self.assertEqual(first, "refined prompt")
        self.assertEqual(second, "refined prompt")
        self.assertEqual(len(publisher.messages), 1)
        self.assertEqual(publisher.messages[0].address, "/prompt")
        self.assertEqual(publisher.messages[0].value, "refined prompt")

    def test_uses_deterministic_fallback_and_does_not_log_input(self):
        llm_client = Mock()
        llm_client.generate_prompt.return_value = None
        publisher = RecordingOutputPublisher()
        logger = Mock()
        orchestrator = PromptOrchestrator(
            llm_client,
            publisher,
            logger=logger,
        )

        result = orchestrator.refine_and_send("private raw words")

        self.assertEqual(result, "private raw words, cinematic, 8k")
        self.assertNotIn("private raw words", str(logger.method_calls))
        self.assertEqual(
            logger.warning.call_args.kwargs["extra"]["event"],
            "prompt_refinement_fallback",
        )

    def test_failed_delivery_can_be_retried_and_cleanup_is_idempotent(self):
        llm_client = Mock()
        llm_client.generate_prompt.return_value = "refined prompt"
        publisher = Mock()
        publisher.send.side_effect = [False, True]
        orchestrator = PromptOrchestrator(llm_client, publisher)

        orchestrator.refine_and_send("raw text")
        orchestrator.refine_and_send("raw text")
        orchestrator.close()
        orchestrator.close()

        self.assertEqual(publisher.send.call_count, 2)
        publisher.close.assert_called_once_with()
        llm_client.close.assert_called_once_with()

    def test_closes_the_llm_client_when_publisher_cleanup_fails(self):
        llm_client = Mock()
        publisher = Mock()
        publisher.close.side_effect = OSError("socket cleanup failed")
        orchestrator = PromptOrchestrator(llm_client, publisher)

        with self.assertRaises(OSError):
            orchestrator.close()

        llm_client.close.assert_called_once_with()

    def test_ignores_empty_input(self):
        llm_client = Mock()
        publisher = Mock()
        orchestrator = PromptOrchestrator(llm_client, publisher)

        self.assertIsNone(orchestrator.refine_and_send("  \n "))
        llm_client.generate_prompt.assert_not_called()
        publisher.send.assert_not_called()


class ImportSafetyTests(unittest.TestCase):
    def test_import_has_no_environment_network_or_socket_side_effects(self):
        project_root = Path(__file__).resolve().parents[1]
        script = """
import os
from unittest.mock import patch
with patch('requests.Session') as session, patch(
    'pythonosc.udp_client.SimpleUDPClient'
) as osc_client:
    import orchestrator
    assert os.environ.get('IMPORT_SIDE_EFFECT_SENTINEL') is None
    session.assert_not_called()
    osc_client.assert_not_called()
"""
        environment = os.environ.copy()
        environment.pop("IMPORT_SIDE_EFFECT_SENTINEL", None)
        environment["PYTHONPATH"] = str(project_root)
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, ".env").write_text(
                "IMPORT_SIDE_EFFECT_SENTINEL=loaded\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                [sys.executable, "-c", script],
                cwd=directory,
                env=environment,
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()

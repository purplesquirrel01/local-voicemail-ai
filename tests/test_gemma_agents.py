import os
import unittest
from unittest.mock import patch

from gemma_agents import CachedAgent, HttpAgentTextGenerator, _stdlib_post
from json_utils import AgentOutputError


class GemmaAgentTests(unittest.TestCase):
    def test_stdlib_worker_request_uses_litert_bearer_token(self):
        captured = {}

        class FakeResponse:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def read(self):
                return b'{"response":"ok"}'

        def fake_urlopen(request, *, timeout):
            captured["authorization"] = request.get_header("Authorization")
            captured["timeout"] = timeout
            return FakeResponse()

        with (
            patch.dict(os.environ, {"LITERT_API_KEY": "worker-secret"}),
            patch("urllib.request.urlopen", side_effect=fake_urlopen),
        ):
            response = _stdlib_post(
                "http://127.0.0.1:8788/api/chat",
                json={"message": "test"},
                timeout=12,
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(captured["authorization"], "Bearer worker-secret")
        self.assertEqual(captured["timeout"], 12)

    def test_cached_agent_builds_cache_friendly_prompt_and_parses_json(self):
        calls = []

        def fake_generator(prompt, *, max_output_tokens, agent_name):
            calls.append({"prompt": prompt, "max_output_tokens": max_output_tokens, "agent_name": agent_name})
            return '{"callback_numbers": [], "fax_numbers": [], "uncertain_numbers": [], "possible_errors": []}'

        agent = CachedAgent(
            name="numbers",
            static_prompt="Return JSON only.",
            max_output_tokens=192,
            expected_fields=["callback_numbers", "fax_numbers", "uncertain_numbers", "possible_errors"],
            text_generator=fake_generator,
        )

        result = agent.run({"number_candidates": []})

        self.assertEqual(result["callback_numbers"], [])
        self.assertEqual(calls[0]["max_output_tokens"], 192)
        self.assertEqual(calls[0]["agent_name"], "numbers")
        self.assertTrue(calls[0]["prompt"].startswith("Return JSON only.\n\nInput JSON:\n"))
        self.assertIn('"number_candidates":[]', calls[0]["prompt"])
        self.assertGreaterEqual(agent.metrics["last_duration_ms"], 0)

    def test_cached_agent_passes_agent_name_to_text_generator(self):
        calls = []

        def fake_generator(prompt, *, max_output_tokens, agent_name):
            calls.append(
                {
                    "prompt": prompt,
                    "max_output_tokens": max_output_tokens,
                    "agent_name": agent_name,
                }
            )
            return '{"callback_numbers":[],"fax_numbers":[],"uncertain_numbers":[],"possible_errors":[]}'

        agent = CachedAgent(
            "numbers",
            "Numbers prompt",
            192,
            ["callback_numbers", "fax_numbers", "uncertain_numbers", "possible_errors"],
            text_generator=fake_generator,
        )

        agent.run({"number_candidates": []})

        self.assertEqual(calls[0]["agent_name"], "numbers")
        self.assertEqual(calls[0]["max_output_tokens"], 192)
        self.assertTrue(calls[0]["prompt"].startswith("Numbers prompt\n\nInput JSON:\n"))

    def test_cached_agent_passes_constraint_metadata_when_builder_supplies_schema(self):
        calls = []
        schema = {
            "type": "object",
            "properties": {"callback_ids": {"type": "array"}},
            "required": ["callback_ids"],
            "additionalProperties": False,
        }

        def fake_generator(prompt, *, max_output_tokens, agent_name, constraint_schema=None, constraint_name=None):
            calls.append(
                {
                    "prompt": prompt,
                    "max_output_tokens": max_output_tokens,
                    "agent_name": agent_name,
                    "constraint_schema": constraint_schema,
                    "constraint_name": constraint_name,
                }
            )
            return '{"callback_ids":[],"fax_ids":[],"uncertain_ids":[],"errors":[]}'

        agent = CachedAgent(
            "numbers",
            "Numbers prompt",
            192,
            ["callback_ids", "fax_ids", "uncertain_ids", "errors"],
            text_generator=fake_generator,
            constraint_builder=lambda agent_name, payload: schema,
        )

        result = agent.run({"number_candidates": [{"id": "number:0"}]})

        self.assertEqual(result["callback_ids"], [])
        self.assertEqual(calls[0]["constraint_schema"], schema)
        self.assertEqual(calls[0]["constraint_name"], "numbers_compact_json")
        self.assertEqual(agent.metrics["last_constraint_mode"], "json_schema")

    def test_cached_agent_raises_parse_error_for_bad_json(self):
        agent = CachedAgent(
            name="numbers",
            static_prompt="Return JSON only.",
            max_output_tokens=192,
            expected_fields=["callback_numbers"],
            text_generator=lambda *_args, **_kwargs: "not json",
        )

        with self.assertRaises(AgentOutputError):
            agent.run({})

    def test_http_agent_text_generator_posts_chat_request_and_extracts_response(self):
        calls = []

        class FakeResponse:
            def raise_for_status(self):
                calls.append({"raised": True})

            def json(self):
                return {"response": '{"name_ids":["name:0"],"errors":[]}'}

        def fake_post(url, *, json, timeout):
            calls.append({"url": url, "json": json, "timeout": timeout})
            return FakeResponse()

        generator = HttpAgentTextGenerator(
            {"name": "http://127.0.0.1:8789/api/chat"},
            timeout_seconds=12,
            post=fake_post,
        )

        text = generator("Name prompt", max_output_tokens=96, agent_name="name")

        self.assertEqual(text, '{"name_ids":["name:0"],"errors":[]}')
        self.assertEqual(calls[0]["url"], "http://127.0.0.1:8789/api/chat")
        self.assertEqual(calls[0]["timeout"], 12)
        self.assertEqual(
            calls[0]["json"],
            {"message": "Name prompt", "history": [], "show_thinking": False, "max_output_tokens": 96},
        )
        self.assertEqual(calls[1], {"raised": True})

    def test_http_agent_text_generator_posts_constraint_schema_when_supplied(self):
        calls = []
        schema = {
            "type": "object",
            "properties": {"name_ids": {"type": "array"}},
            "required": ["name_ids"],
            "additionalProperties": False,
        }

        class FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "response": '{"name_ids":["name:0"],"name_correction_ids":[],"errors":[]}',
                    "constraint_mode": "json_schema",
                }

        def fake_post(url, *, json, timeout):
            calls.append({"url": url, "json": json, "timeout": timeout})
            return FakeResponse()

        generator = HttpAgentTextGenerator(
            {"name": "http://127.0.0.1:8789/api/chat"},
            timeout_seconds=12,
            post=fake_post,
        )

        text = generator(
            "Name prompt",
            max_output_tokens=96,
            agent_name="name",
            constraint_schema=schema,
            constraint_name="name_compact_json",
        )

        self.assertEqual(text, '{"name_ids":["name:0"],"name_correction_ids":[],"errors":[]}')
        self.assertEqual(calls[0]["json"]["constraint_schema"], schema)
        self.assertEqual(calls[0]["json"]["constraint_name"], "name_compact_json")
        self.assertEqual(generator.last_constraint_modes["name"], "json_schema")


if __name__ == "__main__":
    unittest.main()

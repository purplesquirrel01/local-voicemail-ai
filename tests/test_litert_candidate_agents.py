import json
import os
import unittest

import litert_chat_web


class SelectingNumbersAgent:
    def run(self, payload):
        first = payload["number_candidates"][0]
        return {
            "callback_numbers": [{"candidate_id": first["id"], "evidence_text": first["evidence_text"]}],
            "fax_numbers": [],
            "uncertain_numbers": [],
            "possible_errors": [],
        }


class EmptyIdentityAgent:
    def run(self, payload):
        return {"patient_names": [], "dob_candidates": [], "possible_errors": []}


class FailingAgent:
    def run(self, _payload):
        raise RuntimeError("synthetic focused-agent failure")


class LitertCandidateAgentsTests(unittest.TestCase):
    def setUp(self):
        self.old_env = os.environ.copy()
        self.old_engine = litert_chat_web.engine
        self.old_auth_required = litert_chat_web.LITERT_REQUIRE_AUTH
        self.old_get_agents = getattr(litert_chat_web, "get_candidate_agents", None)
        self.old_generate_text = litert_chat_web.generate_text
        self.old_run_generate_text_locked = getattr(litert_chat_web, "run_generate_text_locked", None)
        self.old_create_conversation_with_output_limit = litert_chat_web.create_conversation_with_output_limit
        self.old_generate_candidate_or_shadow_text = getattr(litert_chat_web, "generate_candidate_or_shadow_text", None)
        self.old_candidate_agents_cache = getattr(litert_chat_web, "candidate_agents_cache", None)
        self.old_orchestrator_only = getattr(litert_chat_web, "LITERT_ORCHESTRATOR_ONLY", False)
        self.old_litert_lm = litert_chat_web.litert_lm
        self.old_last_generation_constraint_mode = litert_chat_web.last_generation_constraint_mode
        self.old_last_generation_constraint_name = litert_chat_web.last_generation_constraint_name
        self.old_last_generation_constraint_supported = litert_chat_web.last_generation_constraint_supported
        litert_chat_web.engine = object()
        litert_chat_web.LITERT_REQUIRE_AUTH = False
        litert_chat_web.last_generation_constraint_mode = "disabled"
        litert_chat_web.last_generation_constraint_name = ""
        litert_chat_web.last_generation_constraint_supported = False

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.old_env)
        litert_chat_web.engine = self.old_engine
        litert_chat_web.LITERT_REQUIRE_AUTH = self.old_auth_required
        litert_chat_web.generate_text = self.old_generate_text
        litert_chat_web.create_conversation_with_output_limit = self.old_create_conversation_with_output_limit
        if self.old_run_generate_text_locked is not None:
            litert_chat_web.run_generate_text_locked = self.old_run_generate_text_locked
        if self.old_generate_candidate_or_shadow_text is not None:
            litert_chat_web.generate_candidate_or_shadow_text = self.old_generate_candidate_or_shadow_text
        if hasattr(litert_chat_web, "candidate_agents_cache"):
            litert_chat_web.candidate_agents_cache = self.old_candidate_agents_cache
        if self.old_get_agents is not None:
            litert_chat_web.get_candidate_agents = self.old_get_agents
        litert_chat_web.LITERT_ORCHESTRATOR_ONLY = self.old_orchestrator_only
        litert_chat_web.litert_lm = self.old_litert_lm
        litert_chat_web.last_generation_constraint_mode = self.old_last_generation_constraint_mode
        litert_chat_web.last_generation_constraint_name = self.old_last_generation_constraint_name
        litert_chat_web.last_generation_constraint_supported = self.old_last_generation_constraint_supported

    def test_candidate_mode_returns_chat_wrapper_with_final_json_response(self):
        os.environ["GEMMA_EXTRACT_MODE"] = "candidate_agents"
        litert_chat_web.get_candidate_agents = lambda: {
            "numbers": SelectingNumbersAgent(),
            "identity": EmptyIdentityAgent(),
        }
        prompt = 'Rules\n\nInput JSON:\n{"transcript":"Please call me back at 202-555-0108.","caller_id":"SYNTHETIC"}'

        response = litert_chat_web.chat(litert_chat_web.ChatRequest(message=prompt))

        final = json.loads(response["response"])
        self.assertEqual(final["callback_numbers"][0]["normalized"], "2025550108")
        self.assertEqual(response["message"]["content"], response["response"])
        self.assertIn("raw", response)

    def test_shadow_mode_returns_legacy_response_text(self):
        os.environ["GEMMA_EXTRACT_MODE"] = "shadow_candidate_agents"
        legacy_text = json.dumps(
            {
                "patient_names": [],
                "dob_candidates": [],
                "callback_numbers": [],
                "fax_numbers": [],
                "uncertain_numbers": [],
                "possible_errors": ["legacy baseline"],
            }
        )
        litert_chat_web.generate_text = lambda _req, **_kwargs: legacy_text
        litert_chat_web.get_candidate_agents = lambda: {
            "numbers": SelectingNumbersAgent(),
            "identity": EmptyIdentityAgent(),
        }
        prompt = 'Rules\n\nInput JSON:\n{"transcript":"Please call me back at 202-555-0108."}'

        response = litert_chat_web.chat(litert_chat_web.ChatRequest(message=prompt))

        self.assertEqual(response["response"], legacy_text)

    def test_health_includes_candidate_agent_fields(self):
        health = litert_chat_web.health()

        self.assertIn("candidate_extractor_enabled", health)
        self.assertIn("candidate_agent_mode", health)
        self.assertIn("agents_loaded", health)
        self.assertIn("agent_execution_mode", health)
        self.assertEqual(health["agent_execution_mode"], "sequential_conversation")
        self.assertIn("candidate_agent_failures", health)
        self.assertIn("legacy_fallbacks", health)
        self.assertIn("candidate_agent_constrained_decoding", health)
        self.assertIn("litert_constrained_decoding_enabled", health)
        self.assertIn("litert_constrained_decoding_supported", health)
        self.assertIn("litert_constrained_decoding_required", health)
        self.assertIn("last_agent_constraint_modes", health)
        self.assertNotIn("prompt_cache", health)
        self.assertNotIn("agent_prompt_cache_mode", health)
        self.assertNotIn("candidate_agent_prompt_cache_enabled", health)
        self.assertNotIn("candidate_agent_prompt_cache_require_all", health)
        self.assertNotIn("candidate_agent_prompt_cache_namespaces", health)

    def test_six_agent_topology_health_reports_exact_agents_and_sequential_mode(self):
        os.environ["GEMMA_EXTRACT_MODE"] = "candidate_agents"
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "scout_subject_general_fallback"
        os.environ["CANDIDATE_AGENT_E2B_SCOUT_ENABLED"] = "true"
        os.environ["CANDIDATE_AGENT_EXECUTION"] = "sequential_conversation"
        litert_chat_web.candidate_agents_cache = None

        health = litert_chat_web.health()

        self.assertEqual(health["candidate_agent_mode"], "candidate_agents")
        self.assertEqual(
            health["candidate_agent_topology"],
            "scout_subject_general_fallback",
        )
        self.assertEqual(health["agent_execution_mode"], "sequential_conversation")
        self.assertEqual(
            health["agents_loaded"],
            ["caller_name_fallback", "candidate_scout", "dob", "name", "numbers", "subject_name"],
        )

    def test_numbers_only_topology_health_reports_only_numbers_agent(self):
        os.environ["GEMMA_EXTRACT_MODE"] = "candidate_agents"
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "numbers_only"
        os.environ["CANDIDATE_AGENT_EXECUTION"] = "sequential_conversation"
        litert_chat_web.candidate_agents_cache = None

        health = litert_chat_web.health()

        self.assertEqual(health["candidate_agent_mode"], "candidate_agents")
        self.assertEqual(health["candidate_agent_topology"], "numbers_only")
        self.assertEqual(health["agents_loaded"], ["numbers"])

    def test_candidate_mode_uses_sequential_agent_token_caps_without_prompt_cache(self):
        os.environ["GEMMA_EXTRACT_MODE"] = "candidate_agents"
        os.environ["CANDIDATE_AGENT_MAX_OUTPUT_TOKENS_NUMBERS"] = "64"
        os.environ["CANDIDATE_AGENT_MAX_OUTPUT_TOKENS_IDENTITY"] = "96"
        calls = []

        def fake_run_generate_text_locked(req, *, max_output_tokens=None):
            calls.append((req.message, max_output_tokens))
            if max_output_tokens == 64:
                return '{"callback_ids":["number:0"],"fax_ids":[],"uncertain_ids":[],"errors":[]}'
            if max_output_tokens == 96:
                return '{"name_ids":[],"dob_ids":[],"errors":[]}'
            raise AssertionError(f"unexpected max_output_tokens={max_output_tokens}")

        litert_chat_web.candidate_agents_cache = None
        litert_chat_web.run_generate_text_locked = fake_run_generate_text_locked
        prompt = 'Rules\n\nInput JSON:\n{"transcript":"Please call me back at 202-555-0108.","caller_id":"SYNTHETIC"}'

        response = litert_chat_web.chat(litert_chat_web.ChatRequest(message=prompt))

        final = json.loads(response["response"])
        self.assertEqual(final["callback_numbers"][0]["normalized"], "2025550108")
        self.assertEqual([max_tokens for _prompt, max_tokens in calls], [64, 96])
        self.assertTrue(all("prompt_cache" not in prompt for prompt, _max_tokens in calls))

    def test_worker_chat_honors_request_max_output_tokens(self):
        calls = []

        def fake_run_generate_text_locked(req, *, max_output_tokens=None):
            calls.append((req.message, max_output_tokens))
            return "{}"

        litert_chat_web.run_generate_text_locked = fake_run_generate_text_locked

        response = litert_chat_web.chat(
            litert_chat_web.ChatRequest(
                message="Worker prompt",
                history=[],
                show_thinking=False,
                max_output_tokens=64,
            )
        )

        self.assertEqual(response["response"], "{}")
        self.assertEqual(calls, [("Worker prompt", 64)])

    def test_worker_uses_constrained_adapter_when_request_has_schema(self):
        os.environ["LITERT_CONSTRAINED_DECODING"] = "true"
        calls = []

        class FakeConversation:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def send_message_async(self, message, *, decoding_constraint=None):
                calls.append({"message": message, "decoding_constraint": decoding_constraint})
                return ["{}"]

        litert_chat_web.create_conversation_with_output_limit = lambda *_args, **_kwargs: FakeConversation()

        text = litert_chat_web.generate_text(
            litert_chat_web.ChatRequest(
                message="Worker prompt",
                constraint_schema={"type": "object", "properties": {}, "required": []},
                constraint_name="numbers_compact_json",
            ),
            max_output_tokens=64,
        )

        self.assertEqual(text, "{}")
        self.assertEqual(calls[0]["decoding_constraint"]["type"], "json_schema")
        self.assertEqual(litert_chat_web.last_generation_constraint_mode, "json_schema")
        self.assertTrue(litert_chat_web.last_generation_constraint_supported)

    def test_worker_falls_back_when_constraints_unsupported_and_not_required(self):
        os.environ["LITERT_CONSTRAINED_DECODING"] = "true"

        class FakeConversation:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def send_message_async(self, message):
                return ["{}"]

        litert_chat_web.create_conversation_with_output_limit = lambda *_args, **_kwargs: FakeConversation()

        text = litert_chat_web.generate_text(
            litert_chat_web.ChatRequest(
                message="Worker prompt",
                constraint_schema={"type": "object", "properties": {}, "required": []},
                constraint_name="numbers_compact_json",
            ),
            max_output_tokens=64,
        )

        self.assertEqual(text, "{}")
        self.assertEqual(litert_chat_web.last_generation_constraint_mode, "unsupported_fallback")
        self.assertFalse(litert_chat_web.last_generation_constraint_supported)

    def test_worker_fails_when_constraints_unsupported_and_required(self):
        os.environ["LITERT_CONSTRAINED_DECODING"] = "true"
        os.environ["LITERT_CONSTRAINED_DECODING_REQUIRE"] = "true"

        class FakeConversation:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def send_message_async(self, message):
                return ["{}"]

        litert_chat_web.create_conversation_with_output_limit = lambda *_args, **_kwargs: FakeConversation()

        with self.assertRaises(RuntimeError):
            litert_chat_web.generate_text(
                litert_chat_web.ChatRequest(
                    message="Worker prompt",
                    constraint_schema={"type": "object", "properties": {}, "required": []},
                    constraint_name="numbers_compact_json",
                ),
                max_output_tokens=64,
            )

    def test_candidate_mode_falls_back_to_legacy_when_focused_agent_unavailable(self):
        os.environ["GEMMA_EXTRACT_MODE"] = "candidate_agents"
        os.environ["CANDIDATE_AGENT_FALLBACK_TO_LEGACY"] = "true"
        legacy_text = json.dumps(
            {
                "patient_names": [],
                "dob_candidates": [],
                "callback_numbers": ["202-555-0108"],
                "fax_numbers": [],
                "uncertain_numbers": [],
                "possible_errors": [],
            }
        )
        litert_chat_web.generate_text = lambda _req, **_kwargs: legacy_text
        litert_chat_web.get_candidate_agents = lambda: {
            "numbers": FailingAgent(),
            "identity": FailingAgent(),
        }
        prompt = 'Rules\n\nInput JSON:\n{"transcript":"Please call me back at 202-555-0108."}'

        response = litert_chat_web.chat(litert_chat_web.ChatRequest(message=prompt))

        self.assertEqual(response["response"], legacy_text)

    def test_orchestrator_only_startup_skips_engine_load(self):
        litert_chat_web.LITERT_ORCHESTRATOR_ONLY = True
        litert_chat_web.litert_lm = None
        litert_chat_web.engine = object()

        litert_chat_web.startup()

        self.assertIsNone(litert_chat_web.engine)

    def test_orchestrator_only_chat_accepts_extraction_prompt_without_engine(self):
        os.environ["GEMMA_EXTRACT_MODE"] = "candidate_agents"
        litert_chat_web.LITERT_ORCHESTRATOR_ONLY = True
        litert_chat_web.engine = None
        final_text = json.dumps(
            {
                "patient_names": [],
                "dob_candidates": [],
                "callback_numbers": ["202-555-0108"],
                "fax_numbers": [],
                "uncertain_numbers": [],
                "possible_errors": [],
            }
        )
        litert_chat_web.generate_candidate_or_shadow_text = lambda _req, _payload: final_text
        prompt = 'Rules\n\nInput JSON:\n{"transcript":"Please call me back at 202-555-0108."}'

        response = litert_chat_web.chat(litert_chat_web.ChatRequest(message=prompt))

        self.assertEqual(response["response"], final_text)

    def test_orchestrator_only_chat_rejects_normal_chat_without_engine(self):
        os.environ["GEMMA_EXTRACT_MODE"] = "candidate_agents"
        litert_chat_web.LITERT_ORCHESTRATOR_ONLY = True
        litert_chat_web.engine = None

        with self.assertRaises(litert_chat_web.HTTPException) as caught:
            litert_chat_web.chat(litert_chat_web.ChatRequest(message="Hello"))

        self.assertEqual(caught.exception.status_code, 503)

    def test_orchestrator_only_health_reports_worker_urls_and_timings(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "split_identity_correction"
        os.environ["CANDIDATE_AGENT_EXECUTION"] = "parallel_http"
        os.environ["CANDIDATE_AGENT_NUMBERS_URL"] = "http://127.0.0.1:8788/api/chat"
        os.environ["CANDIDATE_AGENT_NAME_URL"] = "http://127.0.0.1:8789/api/chat"
        os.environ["CANDIDATE_AGENT_DOB_URL"] = "http://127.0.0.1:8790/api/chat"
        os.environ["CANDIDATE_AGENT_NAME_CORRECTION_URL"] = "http://127.0.0.1:8791/api/chat"
        litert_chat_web.LITERT_ORCHESTRATOR_ONLY = True
        litert_chat_web.engine = None

        health = litert_chat_web.health()

        self.assertTrue(health["orchestrator_only"])
        self.assertFalse(health["model_loaded"])
        self.assertEqual(health["backend"], "orchestrator")
        self.assertEqual(health["candidate_agent_topology"], "split_identity_correction")
        self.assertEqual(health["agent_execution_mode"], "parallel_http")
        self.assertEqual(
            health["candidate_agent_worker_urls"],
            {
                "numbers": "http://127.0.0.1:8788/api/chat",
                "name": "http://127.0.0.1:8789/api/chat",
                "dob": "http://127.0.0.1:8790/api/chat",
                "name_correction": "http://127.0.0.1:8791/api/chat",
            },
        )
        self.assertIn("last_agent_timings_ms", health)
        self.assertNotIn("prompt_cache", health)

    def test_four_worker_six_agent_orchestrator_health(self):
        os.environ.update(
            {
                "GEMMA_EXTRACT_MODE": "candidate_agents",
                "CANDIDATE_AGENT_TOPOLOGY": "scout_subject_general_fallback",
                "CANDIDATE_AGENT_EXECUTION": "parallel_http",
                "CANDIDATE_AGENT_E2B_SCOUT_ENABLED": "true",
                "CANDIDATE_AGENT_E2B_SCOUT_URL": "http://127.0.0.1:8788/api/chat",
                "CANDIDATE_AGENT_NUMBERS_URL": "http://127.0.0.1:8788/api/chat",
                "CANDIDATE_AGENT_SUBJECT_NAME_URL": "http://127.0.0.1:8789/api/chat",
                "CANDIDATE_AGENT_DOB_URL": "http://127.0.0.1:8790/api/chat",
                "CANDIDATE_AGENT_NAME_URL": "http://127.0.0.1:8791/api/chat",
                "CANDIDATE_AGENT_CALLER_NAME_FALLBACK_URL": "http://127.0.0.1:8788/api/chat",
            }
        )
        litert_chat_web.LITERT_ORCHESTRATOR_ONLY = True
        litert_chat_web.engine = None
        litert_chat_web.candidate_agents_cache = None

        health = litert_chat_web.health()

        self.assertTrue(health["orchestrator_only"])
        self.assertFalse(health["model_loaded"])
        self.assertEqual(health["backend"], "orchestrator")
        self.assertEqual(
            health["candidate_agent_topology"],
            "scout_subject_general_fallback",
        )
        self.assertEqual(health["agent_execution_mode"], "parallel_http")
        self.assertEqual(
            health["agents_loaded"],
            [
                "caller_name_fallback",
                "candidate_scout",
                "dob",
                "name",
                "numbers",
                "subject_name",
            ],
        )
        self.assertEqual(
            health["candidate_agent_worker_urls"],
            {
                "candidate_scout": "http://127.0.0.1:8788/api/chat",
                "numbers": "http://127.0.0.1:8788/api/chat",
                "name": "http://127.0.0.1:8791/api/chat",
                "subject_name": "http://127.0.0.1:8789/api/chat",
                "caller_name_fallback": "http://127.0.0.1:8788/api/chat",
                "dob": "http://127.0.0.1:8790/api/chat",
            },
        )
        self.assertNotIn("prompt_cache", health)

    def test_orchestrator_only_health_reports_dual_correction_worker_urls(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "split_identity_dual_correction"
        os.environ["CANDIDATE_AGENT_EXECUTION"] = "parallel_http"
        os.environ["CANDIDATE_AGENT_NUMBERS_URL"] = "http://127.0.0.1:8788/api/chat"
        os.environ["CANDIDATE_AGENT_NAME_URL"] = "http://127.0.0.1:8789/api/chat"
        os.environ["CANDIDATE_AGENT_DOB_URL"] = "http://127.0.0.1:8790/api/chat"
        os.environ["CANDIDATE_AGENT_SPELLING_CORRECTION_URL"] = "http://127.0.0.1:8791/api/chat"
        os.environ["CANDIDATE_AGENT_CALLER_ID_CORRECTION_URL"] = "http://127.0.0.1:8792/api/chat"
        litert_chat_web.LITERT_ORCHESTRATOR_ONLY = True
        litert_chat_web.engine = None

        health = litert_chat_web.health()

        self.assertEqual(health["candidate_agent_topology"], "split_identity_dual_correction")
        self.assertEqual(
            health["candidate_agent_worker_urls"],
            {
                "numbers": "http://127.0.0.1:8788/api/chat",
                "name": "http://127.0.0.1:8789/api/chat",
                "dob": "http://127.0.0.1:8790/api/chat",
                "spelling_correction": "http://127.0.0.1:8791/api/chat",
                "caller_id_correction": "http://127.0.0.1:8792/api/chat",
            },
        )
        self.assertEqual(
            health["agents_loaded"],
            ["numbers", "name", "spelling_correction", "caller_id_correction", "dob"],
        )
        self.assertNotIn("prompt_cache", health)

    def test_orchestrator_only_health_hides_disabled_dual_correction_workers(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "split_identity_dual_correction"
        os.environ["CANDIDATE_AGENT_EXECUTION"] = "parallel_http"
        os.environ["CANDIDATE_AGENT_NUMBERS_URL"] = "http://127.0.0.1:8788/api/chat"
        os.environ["CANDIDATE_AGENT_NAME_URL"] = "http://127.0.0.1:8789/api/chat"
        os.environ["CANDIDATE_AGENT_DOB_URL"] = "http://127.0.0.1:8790/api/chat"
        os.environ["CANDIDATE_AGENT_SPELLING_CORRECTION_URL"] = "http://127.0.0.1:8791/api/chat"
        os.environ["CANDIDATE_AGENT_CALLER_ID_CORRECTION_URL"] = "http://127.0.0.1:8792/api/chat"
        os.environ["CANDIDATE_AGENT_SPELLING_CORRECTION_ENABLED"] = "false"
        os.environ["CANDIDATE_AGENT_CALLER_ID_CORRECTION_ENABLED"] = "false"
        litert_chat_web.LITERT_ORCHESTRATOR_ONLY = True
        litert_chat_web.engine = None
        litert_chat_web.candidate_agents_cache = None

        health = litert_chat_web.health()

        self.assertNotIn("spelling_correction", health["candidate_agent_worker_urls"])
        self.assertNotIn("caller_id_correction", health["candidate_agent_worker_urls"])
        self.assertEqual(health["agents_loaded"], ["numbers", "name", "dob"])

    def test_orchestrator_only_health_reports_subject_fallback_worker_urls(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "split_identity_subject_fallback_dual_correction"
        os.environ["CANDIDATE_AGENT_EXECUTION"] = "parallel_http"
        os.environ["CANDIDATE_AGENT_NUMBERS_URL"] = "http://127.0.0.1:8788/api/chat"
        os.environ["CANDIDATE_AGENT_SUBJECT_NAME_URL"] = "http://127.0.0.1:8789/api/chat"
        os.environ["CANDIDATE_AGENT_CALLER_NAME_FALLBACK_URL"] = "http://127.0.0.1:8793/api/chat"
        os.environ["CANDIDATE_AGENT_DOB_URL"] = "http://127.0.0.1:8790/api/chat"
        os.environ["CANDIDATE_AGENT_SPELLING_CORRECTION_URL"] = "http://127.0.0.1:8791/api/chat"
        os.environ["CANDIDATE_AGENT_CALLER_ID_CORRECTION_URL"] = "http://127.0.0.1:8792/api/chat"
        litert_chat_web.LITERT_ORCHESTRATOR_ONLY = True
        litert_chat_web.engine = None

        health = litert_chat_web.health()

        self.assertEqual(health["candidate_agent_topology"], "split_identity_subject_fallback_dual_correction")
        self.assertEqual(
            health["candidate_agent_worker_urls"],
            {
                "numbers": "http://127.0.0.1:8788/api/chat",
                "subject_name": "http://127.0.0.1:8789/api/chat",
                "caller_name_fallback": "http://127.0.0.1:8793/api/chat",
                "dob": "http://127.0.0.1:8790/api/chat",
                "spelling_correction": "http://127.0.0.1:8791/api/chat",
                "caller_id_correction": "http://127.0.0.1:8792/api/chat",
            },
        )
        self.assertEqual(
            health["agents_loaded"],
            ["numbers", "subject_name", "caller_name_fallback", "spelling_correction", "caller_id_correction", "dob"],
        )
        self.assertNotIn("prompt_cache", health)

    def test_orchestrator_only_health_reports_e2b_scout_default_worker_url_when_enabled(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "split_identity_subject_fallback_dual_correction"
        os.environ["CANDIDATE_AGENT_EXECUTION"] = "parallel_http"
        os.environ["CANDIDATE_AGENT_E2B_SCOUT_ENABLED"] = "true"
        os.environ["CANDIDATE_AGENT_NUMBERS_URL"] = "http://127.0.0.1:8788/api/chat"
        os.environ["CANDIDATE_AGENT_SUBJECT_NAME_URL"] = "http://127.0.0.1:8789/api/chat"
        os.environ["CANDIDATE_AGENT_CALLER_NAME_FALLBACK_URL"] = "http://127.0.0.1:8793/api/chat"
        os.environ["CANDIDATE_AGENT_DOB_URL"] = "http://127.0.0.1:8790/api/chat"
        os.environ["CANDIDATE_AGENT_SPELLING_CORRECTION_URL"] = "http://127.0.0.1:8791/api/chat"
        os.environ["CANDIDATE_AGENT_CALLER_ID_CORRECTION_URL"] = "http://127.0.0.1:8792/api/chat"
        litert_chat_web.LITERT_ORCHESTRATOR_ONLY = True
        litert_chat_web.engine = None

        health = litert_chat_web.health()

        self.assertEqual(
            health["candidate_agent_worker_urls"]["candidate_scout"],
            "http://127.0.0.1:8794/api/chat",
        )
        self.assertTrue(health["candidate_agent_e2b_scout_enabled"])


if __name__ == "__main__":
    unittest.main()

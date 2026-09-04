import json
import os
import threading
import unittest

from extraction_orchestrator import (
    build_default_agents,
    caller_id_name_shaped,
    candidate_agent_health,
    candidate_agent_selection,
    extract_with_candidate_agents,
)
from final_resolver import FINAL_SCHEMA_KEYS


FULL_SYNTHETIC_TRANSCRIPT = (
    "This is Avery Exampel. I have an appointment on June the 1st. "
    "I would like to ask about moving the appointment to July. "
    "I am going to be available then. Please check the available dates. "
    "For the return call this is Avery A-V-E-R-Y Exampel E-X-A-M-P-L-E "
    "and my number is 202-555-0123 thank you"
)


class SelectingNumbersAgent:
    def run(self, payload):
        candidates = payload.get("number_candidates", [])
        first = candidates[0]
        return {
            "callback_numbers": [{"candidate_id": first["id"], "evidence_text": first["evidence_text"]}],
            "fax_numbers": [],
            "uncertain_numbers": [],
            "possible_errors": [],
        }


class CompactSelectingNumbersAgent:
    def run(self, payload):
        candidates = payload.get("number_candidates", [])
        self.last_payload = payload
        return {"callback_ids": [candidates[0]["id"]], "fax_ids": [], "uncertain_ids": [], "errors": []}


class CompactSelectingNameAgent:
    def run(self, payload):
        self.last_payload = payload
        for candidate in payload.get("name_candidates", []):
            if candidate.get("raw") == "Jordan Example" and candidate.get("source") != "transcript_spelling_corrected":
                return {"name_ids": [candidate["id"]], "name_correction_ids": [], "errors": []}
        return {"name_ids": [], "name_correction_ids": [], "errors": ["missing_name_candidate"]}


class SubjectSelectingNameAgent:
    def run(self, payload):
        self.last_payload = payload
        for candidate in payload.get("name_candidates", []):
            if candidate.get("source") == "relationship_subject":
                return {"name_ids": [candidate["id"]], "name_correction_ids": [], "errors": []}
        return {"name_ids": [], "name_correction_ids": [], "errors": ["missing_subject_candidate"]}


class BroadRecallSelectingNameAgent:
    def run(self, payload):
        self.last_payload = payload
        for candidate in payload.get("name_candidates", []):
            if candidate.get("source") == "broad_name_recall":
                return {"name_ids": [candidate["id"]], "name_correction_ids": [], "errors": []}
        return {"name_ids": [], "name_correction_ids": [], "errors": ["missing_broad_recall_candidate"]}


class CompactSelectingNameCorrectionAgent:
    def run(self, payload):
        self.last_payload = payload
        for candidate in payload.get("name_candidates", []):
            if candidate.get("source") in {"transcript_spelling_corrected", "caller_id_corrected"}:
                return {"name_ids": [candidate["id"]], "name_correction_ids": [], "errors": []}
        corrections = payload.get("name_correction_candidates", [])
        return {
            "name_ids": [],
            "name_correction_ids": [corrections[0]["id"]] if corrections else [],
            "errors": [],
        }


class CompactSelectingDobAgent:
    def run(self, payload):
        self.last_payload = payload
        candidates = payload.get("dob_candidates", [])
        return {"dob_ids": [candidates[0]["id"]] if candidates else [], "errors": []}


class EmptyIdentityAgent:
    def run(self, payload):
        return {"patient_names": [], "name_correction_candidates": [], "dob_candidates": [], "possible_errors": []}


class FailingAgent:
    def run(self, payload):
        raise RuntimeError("synthetic failure")


class FailIfCalledAgent:
    def run(self, payload):
        raise AssertionError(f"agent should have been skipped, payload={payload!r}")


class CompactSelectingIdentityAgent:
    def run(self, payload):
        candidates = payload.get("name_candidates", [])
        corrections = payload.get("name_correction_candidates", [])
        return {
            "name_ids": [candidates[0]["id"]] if candidates else [],
            "name_correction_ids": [corrections[0]["id"]] if corrections else [],
            "dob_ids": [],
            "errors": [],
        }


class RecordingAgent:
    def __init__(self, name, output):
        self.name = name
        self.output = output
        self.calls = 0

    def run(self, payload):
        self.calls += 1
        self.last_payload = payload
        return self.output


class OrderedRecordingAgent:
    def __init__(self, name, order, output):
        self.name = name
        self.order = order
        self.output = output
        self.calls = 0

    def run(self, payload):
        self.calls += 1
        self.last_payload = payload
        self.order.append(self.name)
        return self.output


class PromptCapturingGenerator:
    def __init__(self):
        self.calls = []

    def __call__(
        self,
        prompt,
        *,
        max_output_tokens,
        agent_name,
        constraint_schema=None,
        constraint_name=None,
    ):
        self.calls.append(
            {
                "agent_name": agent_name,
                "prompt": prompt,
                "max_output_tokens": max_output_tokens,
                "constraint_schema": constraint_schema,
                "constraint_name": constraint_name,
            }
        )
        payload = json.loads(prompt.rsplit("\n\nInput JSON:\n", 1)[1])
        if agent_name == "name":
            candidates = payload.get("name_candidates", [])
            return json.dumps(
                {
                    "name_ids": [candidates[0]["id"]] if candidates else [],
                    "name_correction_ids": [],
                    "errors": [],
                }
            )
        if agent_name == "subject_name":
            candidates = payload.get("name_candidates", [])
            return json.dumps(
                {
                    "name_ids": [candidates[0]["id"]] if candidates else [],
                    "name_correction_ids": [],
                    "errors": [],
                }
            )
        if agent_name == "caller_name_fallback":
            candidates = payload.get("name_candidates", [])
            return json.dumps(
                {
                    "name_ids": [candidates[0]["id"]] if candidates else [],
                    "name_correction_ids": [],
                    "errors": [],
                }
            )
        if agent_name == "name_correction":
            candidates = payload.get("name_candidates", [])
            corrections = payload.get("name_correction_candidates", [])
            corrected = [
                candidate for candidate in candidates
                if candidate.get("source") in {"transcript_spelling_corrected", "caller_id_corrected"}
            ]
            return json.dumps(
                {
                    "name_ids": [corrected[0]["id"]] if corrected else [],
                    "name_correction_ids": [corrections[0]["id"]] if corrections else [],
                    "errors": [],
                }
            )
        if agent_name == "spelling_correction":
            candidates = payload.get("name_candidates", [])
            corrected = [
                candidate for candidate in candidates
                if candidate.get("source") == "transcript_spelling_corrected"
            ]
            return json.dumps(
                {
                    "name_ids": [corrected[0]["id"]] if corrected else [],
                    "name_correction_ids": [],
                    "errors": [],
                }
            )
        if agent_name == "caller_id_correction":
            candidates = payload.get("name_candidates", [])
            corrections = payload.get("name_correction_candidates", [])
            corrected = [
                candidate for candidate in candidates
                if candidate.get("source") == "caller_id_corrected"
            ]
            return json.dumps(
                {
                    "name_ids": [corrected[0]["id"]] if corrected else [],
                    "name_correction_ids": [corrections[0]["id"]] if corrections else [],
                    "errors": [],
                }
            )
        if agent_name == "numbers":
            return json.dumps({"callback_ids": [], "fax_ids": [], "uncertain_ids": [], "errors": []})
        if agent_name == "dob":
            return json.dumps({"dob_ids": [], "errors": []})
        return "{}"


class BarrierAgent:
    def __init__(self, name, output, started, release, expected=3):
        self.name = name
        self.output = output
        self.started = started
        self.release = release
        self.expected = expected

    def run(self, payload):
        del payload
        self.started.append(self.name)
        if len(self.started) == self.expected:
            self.release.set()
        if not self.release.wait(1.0):
            raise RuntimeError(f"{self.name} did not run concurrently")
        return self.output


class ScoutCompletionAgent:
    def __init__(self, finished):
        self.finished = finished
        self.calls = 0

    def run(self, payload):
        self.calls += 1
        self.last_payload = payload
        self.finished.set()
        return {"name_candidates": [], "errors": []}


class WaveBarrierDelegatingAgent:
    def __init__(self, name, delegate, started, released, finished, scout_finished, expected=4):
        self.name = name
        self.delegate = delegate
        self.started = started
        self.released = released
        self.finished = finished
        self.scout_finished = scout_finished
        self.expected = expected
        self.calls = 0

    def run(self, payload):
        self.calls += 1
        if not self.scout_finished.is_set():
            raise AssertionError(f"{self.name} started before scout completed")
        self.started.append(self.name)
        if len(self.started) == self.expected:
            self.released.set()
        if not self.released.wait(1.0):
            raise RuntimeError(f"{self.name} did not enter the parallel wave")
        result = self.delegate.run(payload)
        self.finished.add(self.name)
        return result


class PostWaveFallbackAgent:
    def __init__(self, delegate, released, finished, expected):
        self.delegate = delegate
        self.released = released
        self.finished = finished
        self.expected = set(expected)
        self.calls = 0

    def run(self, payload):
        self.calls += 1
        if not self.released.is_set() or self.finished != self.expected:
            raise AssertionError(
                f"fallback started before Wave 2 joined: finished={sorted(self.finished)}"
            )
        return self.delegate.run(payload)


class ExtractionOrchestratorTests(unittest.TestCase):
    def setUp(self):
        self.old_env = os.environ.copy()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.old_env)

    def test_candidate_agents_active_mode_returns_legacy_schema(self):
        result = extract_with_candidate_agents(
            {"transcript": "Please call me back at 202-555-0108.", "caller_id": "SYNTHETIC CALLER"},
            agents={"numbers": SelectingNumbersAgent(), "identity": EmptyIdentityAgent()},
            mode="candidate_agents",
        )

        self.assertEqual(list(result.keys()), FINAL_SCHEMA_KEYS)
        self.assertEqual(result["callback_numbers"][0]["normalized"], "2025550108")

    def test_build_default_agents_attaches_constraints_when_enabled(self):
        os.environ["CANDIDATE_AGENT_CONSTRAINED_DECODING"] = "true"
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "split_identity_subject_fallback_dual_correction"
        generator = PromptCapturingGenerator()

        result = extract_with_candidate_agents(
            {
                "transcript": (
                    "Hello this is Casey Example. I am calling from my husband Morgan Example. "
                    "Please call 217-555-0100."
                ),
                "caller_id": "Unavailable",
            },
            agents=build_default_agents(text_generator=generator),
            mode="candidate_agents",
        )

        self.assertEqual(list(result.keys()), FINAL_SCHEMA_KEYS)
        calls_by_agent = {call["agent_name"]: call for call in generator.calls}
        for agent_name in ("numbers", "subject_name", "caller_name_fallback"):
            self.assertIsNotNone(calls_by_agent[agent_name]["constraint_schema"])
            self.assertEqual(calls_by_agent[agent_name]["constraint_name"], f"{agent_name}_compact_json")
        self.assertEqual(
            calls_by_agent["numbers"]["constraint_schema"]["properties"]["callback_ids"]["items"]["enum"],
            ["number:0"],
        )
        subject_payload = json.loads(calls_by_agent["subject_name"]["prompt"].rsplit("\n\nInput JSON:\n", 1)[1])
        subject_ids = [candidate["id"] for candidate in subject_payload["name_candidates"]]
        self.assertEqual(
            calls_by_agent["subject_name"]["constraint_schema"]["properties"]["name_ids"]["items"]["enum"],
            subject_ids,
        )
        self.assertTrue(subject_ids)

        health = candidate_agent_health(
            agents_loaded=["numbers", "subject_name", "caller_name_fallback", "spelling_correction", "dob"]
        )
        self.assertTrue(health["candidate_agent_constrained_decoding"])
        self.assertEqual(health["last_agent_constraint_modes"]["numbers"], "json_schema")
        self.assertEqual(health["last_agent_constraint_modes"]["subject_name"], "json_schema")

    def test_compact_agent_output_and_payload_return_legacy_schema(self):
        numbers_agent = CompactSelectingNumbersAgent()

        result = extract_with_candidate_agents(
            {"transcript": "Please call me back at 202-555-0108.", "caller_id": "SYNTHETIC CALLER"},
            agents={"numbers": numbers_agent, "identity": EmptyIdentityAgent()},
            mode="candidate_agents",
        )

        self.assertEqual(result["callback_numbers"][0]["normalized"], "2025550108")
        candidate = numbers_agent.last_payload["number_candidates"][0]
        self.assertEqual(set(candidate), {"id", "raw", "normalized", "evidence_text", "cues", "context"})
        self.assertNotIn("window", candidate)
        self.assertNotIn("span", candidate)

    def test_build_default_agents_use_compact_schema_and_lower_token_defaults(self):
        os.environ.pop("CANDIDATE_AGENT_MAX_OUTPUT_TOKENS_NUMBERS", None)
        os.environ.pop("CANDIDATE_AGENT_MAX_OUTPUT_TOKENS_IDENTITY", None)

        agents = build_default_agents(text_generator=lambda *_args, **_kwargs: "{}")

        self.assertEqual(agents["numbers"].max_output_tokens, 192)
        self.assertEqual(agents["identity"].max_output_tokens, 220)
        self.assertIn("callback_ids", agents["numbers"].expected_fields)
        self.assertIn("callback_numbers", agents["numbers"].expected_fields)
        self.assertIn("name_ids", agents["identity"].expected_fields)
        self.assertIn("patient_names", agents["identity"].expected_fields)
        self.assertIn("name_correction_ids", agents["identity"].expected_fields)
        self.assertIn("name_correction_candidates", agents["identity"].expected_fields)

    def test_numbers_only_topology_runs_no_identity_agents(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "numbers_only"
        numbers_agent = CompactSelectingNumbersAgent()
        agents = build_default_agents(text_generator=lambda *_args, **_kwargs: "{}")

        self.assertEqual(sorted(agents), ["numbers"])

        result = extract_with_candidate_agents(
            {
                "transcript": (
                    "This is Jordan Example, date of birth 01/02/1980. "
                    "Please call me back at 202-555-0108."
                ),
                "caller_id": "SYNTHETIC CALLER",
            },
            agents={"numbers": numbers_agent},
            mode="candidate_agents",
            fallback_to_legacy=False,
        )

        self.assertEqual(result["callback_numbers"][0]["normalized"], "2025550108")
        self.assertEqual(result["fax_numbers"], [])
        self.assertEqual(result["patient_names"], [])
        self.assertEqual(result["name_correction_candidates"], [])
        self.assertEqual(result["dob_candidates"], [])
        self.assertEqual(
            set(candidate_agent_health()["last_agent_timings_ms"]),
            {"numbers"},
        )

    def test_custom_topology_builds_only_selected_agents(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "custom"
        os.environ["CANDIDATE_AGENT_SELECTION"] = "dob,numbers,dob"

        agents = build_default_agents(text_generator=lambda *_args, **_kwargs: "{}")

        self.assertEqual(candidate_agent_selection(), ("numbers", "dob"))
        self.assertEqual(sorted(agents), ["dob", "numbers"])

    def test_custom_topology_invokes_only_selected_specialists(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "custom"
        os.environ["CANDIDATE_AGENT_SELECTION"] = "numbers,dob"
        numbers = CompactSelectingNumbersAgent()
        dob = CompactSelectingDobAgent()

        result = extract_with_candidate_agents(
            {
                "transcript": (
                    "This is Jordan Example, date of birth 01/02/1980. "
                    "Please call me back at 202-555-0108."
                ),
                "caller_id": "SYNTHETIC CALLER",
            },
            agents={"numbers": numbers, "dob": dob, "name": FailIfCalledAgent()},
            mode="candidate_agents",
            fallback_to_legacy=False,
        )

        self.assertEqual(result["callback_numbers"][0]["normalized"], "2025550108")
        self.assertEqual(result["dob_candidates"][0]["normalized"], "01/02/1980")
        self.assertEqual(result["patient_names"], [])
        self.assertEqual(
            set(candidate_agent_health()["last_agent_timings_ms"]),
            {"numbers", "dob"},
        )

    def test_custom_topology_can_execute_selected_agents_in_parallel(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "custom"
        os.environ["CANDIDATE_AGENT_SELECTION"] = "numbers,dob"
        os.environ["CANDIDATE_AGENT_EXECUTION"] = "parallel_http"
        rendezvous = threading.Barrier(2, timeout=2)

        class ParallelNumbers(CompactSelectingNumbersAgent):
            def run(self, payload):
                rendezvous.wait()
                return super().run(payload)

        class ParallelDob(CompactSelectingDobAgent):
            def run(self, payload):
                rendezvous.wait()
                return super().run(payload)

        result = extract_with_candidate_agents(
            {
                "transcript": (
                    "Date of birth 01/02/1980. Call 202-555-0108."
                ),
                "caller_id": "SYNTHETIC CALLER",
            },
            agents={"numbers": ParallelNumbers(), "dob": ParallelDob()},
            mode="candidate_agents",
            fallback_to_legacy=False,
        )

        self.assertEqual(result["callback_numbers"][0]["normalized"], "2025550108")
        self.assertEqual(result["dob_candidates"][0]["normalized"], "01/02/1980")

    def test_scout_subject_general_fallback_builds_exactly_six_agents(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "scout_subject_general_fallback"
        os.environ.pop("CANDIDATE_AGENT_E2B_SCOUT_ENABLED", None)

        agents = build_default_agents(text_generator=lambda *_args, **_kwargs: "{}")

        self.assertEqual(
            sorted(agents),
            ["caller_name_fallback", "candidate_scout", "dob", "name", "numbers", "subject_name"],
        )
        self.assertNotIn("identity", agents)
        self.assertNotIn("name_correction", agents)
        self.assertNotIn("spelling_correction", agents)
        self.assertNotIn("caller_id_correction", agents)

    def test_six_agent_topology_runs_sequentially_and_subject_wins_over_general(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "scout_subject_general_fallback"
        os.environ["CANDIDATE_AGENT_E2B_SCOUT_ENABLED"] = "true"
        os.environ["CANDIDATE_AGENT_EXECUTION"] = "sequential_conversation"
        order = []
        agents = {
            "candidate_scout": OrderedRecordingAgent(
                "candidate_scout", order, {"name_candidates": [], "errors": []}
            ),
            "numbers": OrderedRecordingAgent(
                "numbers",
                order,
                {"callback_ids": [], "fax_ids": [], "uncertain_ids": [], "errors": []},
            ),
            "dob": OrderedRecordingAgent("dob", order, {"dob_ids": [], "errors": []}),
            "subject_name": SubjectSelectingNameAgent(),
            "name": CompactSelectingNameAgent(),
            "caller_name_fallback": FailIfCalledAgent(),
        }
        subject = agents["subject_name"]
        original_subject_run = subject.run
        subject.run = lambda payload: (order.append("subject_name") or original_subject_run(payload))
        general = agents["name"]
        original_general_run = general.run
        general.run = lambda payload: (order.append("name") or original_general_run(payload))

        trace = []
        result = extract_with_candidate_agents(
            {
                "transcript": (
                    "This is Jordan Example calling for my husband Morgan Example. "
                    "His date of birth is 01/02/1980. Call 217-555-0100."
                ),
                "caller_id": "SYNTHETIC",
            },
            agents=agents,
            mode="candidate_agents",
            fallback_to_legacy=False,
            trace_sink=trace.append,
        )

        self.assertEqual(result["patient_names"][0]["value"], "Morgan Example")
        self.assertEqual(order, ["candidate_scout", "numbers", "dob", "subject_name", "name"])
        self.assertEqual(
            [entry["agent"] for entry in trace],
            ["candidate_scout", "numbers", "dob", "subject_name", "name"],
        )
        self.assertTrue(all(isinstance(entry["output"], dict) for entry in trace))
        self.assertTrue(all(isinstance(entry["duration_ms"], int) for entry in trace))
        self.assertTrue(all("prompt" not in entry and "input" not in entry for entry in trace))

    def test_six_agent_parallel_http_runs_four_specialists_after_scout(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "scout_subject_general_fallback"
        os.environ["CANDIDATE_AGENT_E2B_SCOUT_ENABLED"] = "true"
        os.environ["CANDIDATE_AGENT_EXECUTION"] = "parallel_http"
        scout_finished = threading.Event()
        wave_released = threading.Event()
        started = []
        finished = set()

        def wave(name, delegate):
            return WaveBarrierDelegatingAgent(
                name,
                delegate,
                started,
                wave_released,
                finished,
                scout_finished,
            )

        agents = {
            "candidate_scout": ScoutCompletionAgent(scout_finished),
            "numbers": wave(
                "numbers",
                RecordingAgent(
                    "numbers",
                    {"callback_ids": [], "fax_ids": [], "uncertain_ids": [], "errors": []},
                ),
            ),
            "dob": wave("dob", RecordingAgent("dob", {"dob_ids": [], "errors": []})),
            "subject_name": wave("subject_name", SubjectSelectingNameAgent()),
            "name": wave("name", CompactSelectingNameAgent()),
            "caller_name_fallback": FailIfCalledAgent(),
        }

        result = extract_with_candidate_agents(
            {
                "transcript": (
                    "This is Jordan Example calling for my husband Morgan Example. "
                    "His date of birth is 01/02/1980. "
                    "Call 217-555-0100 and fax 217-555-0101."
                ),
                "caller_id": "SYNTHETIC",
            },
            agents=agents,
            mode="candidate_agents",
            fallback_to_legacy=False,
        )

        self.assertEqual(set(started), {"numbers", "dob", "subject_name", "name"})
        self.assertEqual(finished, {"numbers", "dob", "subject_name", "name"})
        self.assertEqual(result["patient_names"][0]["value"], "Morgan Example")
        self.assertEqual(candidate_agent_health()["agent_execution_mode"], "parallel_http")

    def test_six_agent_parallel_http_runs_fallback_only_after_wave_join(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "scout_subject_general_fallback"
        os.environ["CANDIDATE_AGENT_E2B_SCOUT_ENABLED"] = "true"
        os.environ["CANDIDATE_AGENT_EXECUTION"] = "parallel_http"
        scout_finished = threading.Event()
        wave_released = threading.Event()
        started = []
        finished = set()
        wave_names = {"numbers", "dob", "subject_name", "name"}

        def wave(name, output):
            return WaveBarrierDelegatingAgent(
                name,
                RecordingAgent(name, output),
                started,
                wave_released,
                finished,
                scout_finished,
            )

        fallback = PostWaveFallbackAgent(
            CompactSelectingNameAgent(),
            wave_released,
            finished,
            wave_names,
        )
        agents = {
            "candidate_scout": ScoutCompletionAgent(scout_finished),
            "numbers": wave(
                "numbers",
                {"callback_ids": [], "fax_ids": [], "uncertain_ids": [], "errors": []},
            ),
            "dob": wave("dob", {"dob_ids": [], "errors": []}),
            "subject_name": wave(
                "subject_name",
                {"name_ids": [], "name_correction_ids": [], "errors": []},
            ),
            "name": wave(
                "name",
                {"name_ids": [], "name_correction_ids": [], "errors": []},
            ),
            "caller_name_fallback": fallback,
        }

        result = extract_with_candidate_agents(
            {
                "transcript": (
                    "This is Jordan Example calling for my husband Morgan Example. "
                    "His date of birth is 01/02/1980. "
                    "Call 217-555-0100 and fax 217-555-0101."
                ),
                "caller_id": "SYNTHETIC",
            },
            agents=agents,
            mode="candidate_agents",
            fallback_to_legacy=False,
        )

        self.assertEqual(set(started), wave_names)
        self.assertEqual(finished, wave_names)
        self.assertEqual(fallback.calls, 1)
        self.assertEqual(result["patient_names"][0]["value"], "Jordan Example")

    def test_six_agent_parallel_http_isolates_one_specialist_failure(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "scout_subject_general_fallback"
        os.environ["CANDIDATE_AGENT_E2B_SCOUT_ENABLED"] = "true"
        os.environ["CANDIDATE_AGENT_EXECUTION"] = "parallel_http"
        before = candidate_agent_health()["legacy_fallbacks"]
        agents = {
            "candidate_scout": RecordingAgent(
                "candidate_scout", {"name_candidates": [], "errors": []}
            ),
            "numbers": CompactSelectingNumbersAgent(),
            "dob": FailingAgent(),
            "subject_name": SubjectSelectingNameAgent(),
            "name": CompactSelectingNameAgent(),
            "caller_name_fallback": FailIfCalledAgent(),
        }

        result = extract_with_candidate_agents(
            {
                "transcript": (
                    "This is Jordan Example calling for my husband Morgan Example. "
                    "His date of birth is 01/02/1980. Call 217-555-0100."
                ),
                "caller_id": "SYNTHETIC",
            },
            agents=agents,
            mode="candidate_agents",
            fallback_to_legacy=False,
        )

        self.assertEqual(result["patient_names"][0]["value"], "Morgan Example")
        self.assertEqual(result["callback_numbers"][0]["normalized"], "2175550100")
        self.assertEqual(result["dob_candidates"], [])
        self.assertTrue(result["possible_errors"])
        self.assertEqual(candidate_agent_health()["legacy_fallbacks"], before)

    def test_six_agent_topology_uses_general_name_before_caller_fallback(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "scout_subject_general_fallback"
        os.environ["CANDIDATE_AGENT_E2B_SCOUT_ENABLED"] = "true"
        agents = {
            "candidate_scout": RecordingAgent(
                "candidate_scout", {"name_candidates": [], "errors": []}
            ),
            "numbers": FailIfCalledAgent(),
            "dob": FailIfCalledAgent(),
            "subject_name": RecordingAgent(
                "subject_name", {"name_ids": [], "name_correction_ids": [], "errors": []}
            ),
            "name": BroadRecallSelectingNameAgent(),
            "caller_name_fallback": FailIfCalledAgent(),
        }

        result = extract_with_candidate_agents(
            {
                "transcript": (
                    "Hello, this is a synthetic office message. "
                    "The message concerns Casey Sample and the appointment."
                ),
                "caller_id": "SYNTHETIC",
            },
            agents=agents,
            mode="candidate_agents",
            fallback_to_legacy=False,
        )

        self.assertEqual(result["patient_names"][0]["value"], "Casey Sample")

    def test_six_agent_topology_calls_fallback_only_when_primary_names_are_empty(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "scout_subject_general_fallback"
        os.environ["CANDIDATE_AGENT_E2B_SCOUT_ENABLED"] = "true"
        fallback = CompactSelectingNameAgent()
        agents = {
            "candidate_scout": RecordingAgent(
                "candidate_scout", {"name_candidates": [], "errors": []}
            ),
            "numbers": FailIfCalledAgent(),
            "dob": FailIfCalledAgent(),
            "subject_name": RecordingAgent(
                "subject_name", {"name_ids": [], "name_correction_ids": [], "errors": []}
            ),
            "name": RecordingAgent(
                "name", {"name_ids": [], "name_correction_ids": [], "errors": []}
            ),
            "caller_name_fallback": fallback,
        }

        result = extract_with_candidate_agents(
            {"transcript": "Hello, this is Jordan Example calling back.", "caller_id": "SYNTHETIC"},
            agents=agents,
            mode="candidate_agents",
            fallback_to_legacy=False,
        )

        self.assertEqual(result["patient_names"][0]["value"], "Jordan Example")
        self.assertEqual(fallback.last_payload["name_candidates"][0]["source"], "self_identification")

    def test_six_agent_topology_rejects_unknown_subject_id_and_runs_fallback(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "scout_subject_general_fallback"
        os.environ["CANDIDATE_AGENT_E2B_SCOUT_ENABLED"] = "true"
        fallback = CompactSelectingNameAgent()
        agents = {
            "candidate_scout": RecordingAgent(
                "candidate_scout", {"name_candidates": [], "errors": []}
            ),
            "numbers": FailIfCalledAgent(),
            "dob": FailIfCalledAgent(),
            "subject_name": RecordingAgent(
                "subject_name",
                {"name_ids": ["name:999"], "name_correction_ids": [], "errors": []},
            ),
            "name": RecordingAgent(
                "name", {"name_ids": [], "name_correction_ids": [], "errors": []}
            ),
            "caller_name_fallback": fallback,
        }

        result = extract_with_candidate_agents(
            {"transcript": "Hello, this is Jordan Example calling back.", "caller_id": "SYNTHETIC"},
            agents=agents,
            mode="candidate_agents",
            fallback_to_legacy=False,
        )

        self.assertEqual(result["patient_names"][0]["value"], "Jordan Example")
        self.assertTrue(hasattr(fallback, "last_payload"))

    def test_six_agent_topology_scout_failure_keeps_deterministic_name_candidates(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "scout_subject_general_fallback"
        os.environ["CANDIDATE_AGENT_E2B_SCOUT_ENABLED"] = "true"
        fallback = CompactSelectingNameAgent()
        agents = {
            "candidate_scout": FailingAgent(),
            "numbers": FailIfCalledAgent(),
            "dob": FailIfCalledAgent(),
            "subject_name": RecordingAgent(
                "subject_name", {"name_ids": [], "name_correction_ids": [], "errors": []}
            ),
            "name": RecordingAgent(
                "name", {"name_ids": [], "name_correction_ids": [], "errors": []}
            ),
            "caller_name_fallback": fallback,
        }

        result = extract_with_candidate_agents(
            {"transcript": "Hello, this is Jordan Example calling back.", "caller_id": "SYNTHETIC"},
            agents=agents,
            mode="candidate_agents",
            fallback_to_legacy=False,
        )

        self.assertEqual(result["patient_names"][0]["value"], "Jordan Example")
        self.assertTrue(hasattr(fallback, "last_payload"))

    def test_six_agent_topology_isolates_specialist_failure_without_legacy_fallback(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "scout_subject_general_fallback"
        os.environ["CANDIDATE_AGENT_E2B_SCOUT_ENABLED"] = "true"
        legacy_calls = []
        agents = {
            "candidate_scout": FailingAgent(),
            "numbers": FailingAgent(),
            "dob": FailingAgent(),
            "subject_name": FailingAgent(),
            "name": FailingAgent(),
            "caller_name_fallback": FailingAgent(),
        }

        result = extract_with_candidate_agents(
            {
                "transcript": (
                    "This is Casey Example calling for Morgan Example born January 2 1980. "
                    "Call 217-555-0100."
                ),
                "caller_id": "SYNTHETIC",
            },
            legacy_extractor=lambda request: legacy_calls.append(request) or {},
            agents=agents,
            mode="candidate_agents",
            fallback_to_legacy=False,
        )

        self.assertEqual(legacy_calls, [])
        self.assertEqual(list(result), FINAL_SCHEMA_KEYS)

    def test_split_identity_topology_builds_numbers_name_and_dob_agents(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "split_identity"
        os.environ.pop("CANDIDATE_AGENT_MAX_OUTPUT_TOKENS_NUMBERS", None)
        os.environ.pop("CANDIDATE_AGENT_MAX_OUTPUT_TOKENS_NAME", None)
        os.environ.pop("CANDIDATE_AGENT_MAX_OUTPUT_TOKENS_DOB", None)

        agents = build_default_agents(text_generator=lambda *_args, **_kwargs: "{}")

        self.assertEqual(sorted(agents), ["dob", "name", "numbers"])
        self.assertEqual(agents["numbers"].max_output_tokens, 192)
        self.assertEqual(agents["name"].max_output_tokens, 220)
        self.assertEqual(agents["dob"].max_output_tokens, 160)
        self.assertIn("name_ids", agents["name"].expected_fields)
        self.assertIn("name_correction_ids", agents["name"].expected_fields)
        self.assertIn("dob_ids", agents["dob"].expected_fields)

    def test_split_identity_correction_topology_builds_four_agents(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "split_identity_correction"
        os.environ.pop("CANDIDATE_AGENT_MAX_OUTPUT_TOKENS_NAME_CORRECTION", None)

        agents = build_default_agents(text_generator=lambda *_args, **_kwargs: "{}")

        self.assertEqual(sorted(agents), ["dob", "name", "name_correction", "numbers"])
        self.assertEqual(agents["name_correction"].max_output_tokens, 128)
        self.assertIn("name_ids", agents["name_correction"].expected_fields)
        self.assertIn("name_correction_ids", agents["name_correction"].expected_fields)

    def test_split_identity_dual_correction_topology_builds_five_agents(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "split_identity_dual_correction"
        os.environ.pop("CANDIDATE_AGENT_MAX_OUTPUT_TOKENS_SPELLING_CORRECTION", None)
        os.environ.pop("CANDIDATE_AGENT_MAX_OUTPUT_TOKENS_CALLER_ID_CORRECTION", None)

        agents = build_default_agents(text_generator=lambda *_args, **_kwargs: "{}")

        self.assertEqual(sorted(agents), ["caller_id_correction", "dob", "name", "numbers", "spelling_correction"])
        self.assertEqual(agents["spelling_correction"].max_output_tokens, 64)
        self.assertEqual(agents["caller_id_correction"].max_output_tokens, 96)
        self.assertIn("name_ids", agents["spelling_correction"].expected_fields)
        self.assertIn("name_correction_ids", agents["caller_id_correction"].expected_fields)

    def test_split_subject_fallback_dual_correction_topology_builds_six_agents(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "split_identity_subject_fallback_dual_correction"
        os.environ.pop("CANDIDATE_AGENT_MAX_OUTPUT_TOKENS_SUBJECT_NAME", None)
        os.environ.pop("CANDIDATE_AGENT_MAX_OUTPUT_TOKENS_CALLER_NAME_FALLBACK", None)

        agents = build_default_agents(text_generator=lambda *_args, **_kwargs: "{}")

        self.assertEqual(
            sorted(agents),
            ["caller_id_correction", "caller_name_fallback", "dob", "numbers", "spelling_correction", "subject_name"],
        )
        self.assertEqual(agents["subject_name"].max_output_tokens, 160)
        self.assertEqual(agents["caller_name_fallback"].max_output_tokens, 160)
        self.assertIn("name_ids", agents["subject_name"].expected_fields)
        self.assertIn("name_correction_ids", agents["caller_name_fallback"].expected_fields)

    def test_dual_correction_flags_remove_disabled_agents_from_default_build(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "split_identity_subject_fallback_dual_correction"

        cases = (
            (
                "CANDIDATE_AGENT_SPELLING_CORRECTION_ENABLED",
                "spelling_correction",
                "caller_id_correction",
            ),
            (
                "CANDIDATE_AGENT_CALLER_ID_CORRECTION_ENABLED",
                "caller_id_correction",
                "spelling_correction",
            ),
        )
        for env_name, disabled_agent, still_enabled_agent in cases:
            with self.subTest(env_name=env_name):
                os.environ["CANDIDATE_AGENT_SPELLING_CORRECTION_ENABLED"] = "true"
                os.environ["CANDIDATE_AGENT_CALLER_ID_CORRECTION_ENABLED"] = "true"
                os.environ[env_name] = "false"

                agents = build_default_agents(text_generator=lambda *_args, **_kwargs: "{}")

                self.assertNotIn(disabled_agent, agents)
                self.assertIn(still_enabled_agent, agents)

    def test_dual_correction_flags_skip_disabled_agents_even_when_candidates_exist(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "split_identity_dual_correction"
        os.environ["CANDIDATE_AGENT_SPELLING_CORRECTION_ENABLED"] = "false"
        os.environ["CANDIDATE_AGENT_CALLER_ID_CORRECTION_ENABLED"] = "false"
        generator = PromptCapturingGenerator()
        agents = build_default_agents(text_generator=generator)

        result = extract_with_candidate_agents(
            {
                "transcript": "Yes, this is Jordan Exampel. J-O-R-D-A-N E-X-A-M-P-E-L.",
                "caller_id": '"JORDAN EXAMPLE" (217)-555-0100',
            },
            agents=agents,
            mode="candidate_agents",
            fallback_to_legacy=False,
        )

        self.assertNotIn("spelling_correction", agents)
        self.assertNotIn("caller_id_correction", agents)
        self.assertTrue(result["patient_names"])
        agent_names = [call["agent_name"] for call in generator.calls]
        self.assertNotIn("spelling_correction", agent_names)
        self.assertNotIn("caller_id_correction", agent_names)
        health = candidate_agent_health(agents_loaded=sorted(agents))
        self.assertIn("spelling_correction", health["last_agent_skipped"])
        self.assertIn("caller_id_correction", health["last_agent_skipped"])
        self.assertEqual(health["last_agent_timings_ms"]["spelling_correction"], 0)
        self.assertEqual(health["last_agent_timings_ms"]["caller_id_correction"], 0)

    def test_e2b_scout_enabled_topology_builds_candidate_scout_agent(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "split_identity_subject_fallback_dual_correction"
        os.environ["CANDIDATE_AGENT_E2B_SCOUT_ENABLED"] = "true"
        os.environ.pop("CANDIDATE_AGENT_E2B_SCOUT_MAX_OUTPUT_TOKENS", None)

        agents = build_default_agents(text_generator=lambda *_args, **_kwargs: "{}")

        self.assertIn("candidate_scout", agents)
        self.assertEqual(agents["candidate_scout"].max_output_tokens, 256)
        self.assertIn("name_candidates", agents["candidate_scout"].expected_fields)
        self.assertIn("errors", agents["candidate_scout"].expected_fields)
        self.assertNotIn("number_candidates", agents["candidate_scout"].expected_fields)

    def test_caller_id_name_shaped_accepts_people_and_rejects_generic_or_org_values(self):
        self.assertTrue(caller_id_name_shaped("LORI VANMETER"))
        self.assertTrue(caller_id_name_shaped("QUINN L EXAMPLE"))
        self.assertTrue(caller_id_name_shaped('"QUINN L EXAMPLE" <2025550124>'))
        self.assertTrue(caller_id_name_shaped("COPELIN,JEFFREY"))
        self.assertTrue(caller_id_name_shaped("SAMPLE QUINN"))
        self.assertFalse(caller_id_name_shaped(""))
        self.assertFalse(caller_id_name_shaped("202-555-0108"))
        self.assertFalse(caller_id_name_shaped("Wireless Caller"))
        self.assertFalse(caller_id_name_shaped("Example Clinic"))
        self.assertFalse(caller_id_name_shaped("Example Clinic"))

    def test_e2b_scout_adds_missed_subject_name_before_subject_worker(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "split_identity_subject_fallback_dual_correction"
        os.environ["CANDIDATE_AGENT_E2B_SCOUT_ENABLED"] = "true"
        scout = RecordingAgent(
            "candidate_scout",
            {
                "name_candidates": [
                    {
                        "raw": "Quinn Sample",
                        "value": "Quinn Sample",
                        "source": "explicit_patient",
                        "evidence_text": "for Quinn Sample",
                    }
                ],
                "dob_candidates": [],
                "number_candidates": [],
                "spelled_sequences": [],
                "agent_hints": [],
                "errors": [],
            },
        )
        subject = RecordingAgent("subject_name", {"name_ids": ["name:0"], "name_correction_ids": [], "errors": []})

        result = extract_with_candidate_agents(
            {"transcript": "Hello from the clinic. The packet is for Quinn Sample.", "caller_id": "SYNTHETIC"},
            agents={
                "candidate_scout": scout,
                "subject_name": subject,
                "caller_name_fallback": FailIfCalledAgent(),
                "spelling_correction": FailIfCalledAgent(),
                "caller_id_correction": FailIfCalledAgent(),
                "numbers": FailIfCalledAgent(),
                "dob": FailIfCalledAgent(),
            },
            mode="candidate_agents",
            fallback_to_legacy=False,
        )

        self.assertEqual(result["patient_names"][0]["value"], "Quinn Sample")
        subject_names = {item["raw"]: item for item in subject.last_payload["name_candidates"]}
        self.assertIn("Quinn Sample", subject_names)
        self.assertEqual(subject_names["Quinn Sample"]["source"], "explicit_patient")
        health = candidate_agent_health()
        self.assertEqual(health["last_e2b_scout_added_counts"]["name_candidates"], 1)
        self.assertNotIn("Quinn", json.dumps(health))

    def test_e2b_scout_rejects_name_when_evidence_is_not_in_transcript(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "split_identity_subject_fallback_dual_correction"
        os.environ["CANDIDATE_AGENT_E2B_SCOUT_ENABLED"] = "true"
        scout = RecordingAgent(
            "candidate_scout",
            {
                "name_candidates": [
                    {
                        "raw": "Rowan Example",
                        "value": "Rowan Example",
                        "source": "explicit_patient",
                        "evidence_text": "for Rowan Example",
                    }
                ],
                "dob_candidates": [],
                "number_candidates": [],
                "spelled_sequences": [],
                "agent_hints": [],
                "errors": [],
            },
        )

        result = extract_with_candidate_agents(
            {"transcript": "Hello from the clinic. No patient name was left.", "caller_id": "SYNTHETIC"},
            agents={
                "candidate_scout": scout,
                "subject_name": FailIfCalledAgent(),
                "caller_name_fallback": FailIfCalledAgent(),
                "spelling_correction": FailIfCalledAgent(),
                "caller_id_correction": FailIfCalledAgent(),
                "numbers": FailIfCalledAgent(),
                "dob": FailIfCalledAgent(),
            },
            mode="candidate_agents",
            fallback_to_legacy=False,
        )

        self.assertEqual(result["patient_names"], [])
        health = candidate_agent_health()
        self.assertEqual(health["last_e2b_scout_rejected_counts"]["missing_evidence_text"], 1)

    def test_e2b_scout_ignores_non_name_proposals(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "split_identity_subject_fallback_dual_correction"
        os.environ["CANDIDATE_AGENT_E2B_SCOUT_ENABLED"] = "true"
        scout = RecordingAgent(
            "candidate_scout",
            {
                "name_candidates": [],
                "dob_candidates": [
                    {"raw": "12-16-62", "normalized": "12/16/1962", "evidence_text": "12-16-62"}
                ],
                "number_candidates": [
                    {
                        "raw": "two one seven five five five zero one three four",
                        "normalized": "2175550134",
                        "label_cue": "callback",
                        "evidence_text": "two one seven five five five zero one three four",
                    }
                ],
                "spelled_sequences": [],
                "agent_hints": [],
                "errors": [],
            },
        )

        result = extract_with_candidate_agents(
            {
                "transcript": "Please use two one seven five five five zero one three four. The date was 12-16-62.",
                "caller_id": "SYNTHETIC",
            },
            agents={
                "candidate_scout": scout,
                "subject_name": FailIfCalledAgent(),
                "caller_name_fallback": FailIfCalledAgent(),
                "spelling_correction": FailIfCalledAgent(),
                "caller_id_correction": FailIfCalledAgent(),
                "numbers": FailIfCalledAgent(),
                "dob": FailIfCalledAgent(),
            },
            mode="candidate_agents",
            fallback_to_legacy=False,
        )

        self.assertEqual(result["callback_numbers"], [])
        self.assertEqual(result["dob_candidates"], [])
        health = candidate_agent_health()
        self.assertEqual(health["last_e2b_scout_added_counts"], {})

    def test_name_only_prompt_excludes_spelling_and_generic_caller_id_rules(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "split_identity"
        generator = PromptCapturingGenerator()

        result = extract_with_candidate_agents(
            {"transcript": "This is Jordan Example calling.", "caller_id": "Wireless Caller"},
            agents=build_default_agents(text_generator=generator),
            mode="candidate_agents",
            fallback_to_legacy=False,
        )

        prompt = next(call["prompt"] for call in generator.calls if call["agent_name"] == "name")
        payload = json.loads(prompt.rsplit("\n\nInput JSON:\n", 1)[1])
        self.assertEqual(result["patient_names"][0]["value"], "Jordan Example")
        self.assertNotIn("Explicit spelling immediately after a spoken patient/caller name may correct the name.", prompt)
        self.assertNotIn("Caller ID may correct a clearly self-identified spoken name", prompt)
        self.assertNotIn("spelled_sequences", payload)
        self.assertEqual(payload["caller_id"], "")

    def test_correction_prompt_includes_spelling_rules_when_spelled_sequences_exist(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "split_identity_correction"
        generator = PromptCapturingGenerator()

        extract_with_candidate_agents(
            {"transcript": "My name is Example, K-O-E-S-P-E-R."},
            agents=build_default_agents(text_generator=generator),
            mode="candidate_agents",
            fallback_to_legacy=False,
        )

        prompt = next(call["prompt"] for call in generator.calls if call["agent_name"] == "name")
        correction_prompt = next(call["prompt"] for call in generator.calls if call["agent_name"] == "name_correction")
        correction_payload = json.loads(correction_prompt.rsplit("\n\nInput JSON:\n", 1)[1])
        payload = json.loads(prompt.rsplit("\n\nInput JSON:\n", 1)[1])
        self.assertNotIn("Explicit spelling immediately after a spoken patient/caller name may correct the name.", prompt)
        self.assertIn("Explicit spelling immediately after a spoken patient/caller name may correct the name.", correction_prompt)
        self.assertIn("Example spelling override:", correction_prompt)
        self.assertNotIn("Caller ID may correct a clearly self-identified spoken name", correction_prompt)
        self.assertNotIn("spelled_sequences", payload)
        self.assertIn("spelled_sequences", correction_payload)

    def test_correction_prompt_includes_caller_id_rules_only_for_name_shaped_caller_id(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "split_identity_correction"
        generator = PromptCapturingGenerator()

        extract_with_candidate_agents(
            {"transcript": "This is Avery Exampel.", "caller_id": "AVERY EXAMPLE"},
            agents=build_default_agents(text_generator=generator),
            mode="candidate_agents",
            fallback_to_legacy=False,
        )

        prompt = next(call["prompt"] for call in generator.calls if call["agent_name"] == "name")
        correction_prompt = next(call["prompt"] for call in generator.calls if call["agent_name"] == "name_correction")
        correction_payload = json.loads(correction_prompt.rsplit("\n\nInput JSON:\n", 1)[1])
        self.assertNotIn("Caller ID may correct a clearly self-identified spoken name", prompt)
        self.assertIn("Caller ID may correct a clearly self-identified spoken name", correction_prompt)
        self.assertIn("name_correction_candidates", correction_prompt)
        self.assertIn("Example Caller ID corrections:", correction_prompt)
        self.assertNotIn("Explicit spelling immediately after a spoken patient/caller name may correct the name.", correction_prompt)
        self.assertEqual(correction_payload["caller_id"], "AVERY EXAMPLE")

    def test_correction_prompt_passes_correction_candidates_for_name_shaped_caller_id(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "split_identity_correction"
        generator = PromptCapturingGenerator()

        result = extract_with_candidate_agents(
            {"transcript": "This is Bailey Sampel.", "caller_id": "EXAMPLE,AVERYS"},
            agents=build_default_agents(text_generator=generator),
            mode="candidate_agents",
            fallback_to_legacy=False,
        )

        prompt = next(call["prompt"] for call in generator.calls if call["agent_name"] == "name_correction")
        payload = json.loads(prompt.rsplit("\n\nInput JSON:\n", 1)[1])
        self.assertIn("name_correction_candidates", payload)
        self.assertEqual(result["name_correction_candidates"][0]["reason"], "last_name_phonetic_match")

    def test_correction_prompt_includes_both_fragments_when_spelling_and_caller_id_exist(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "split_identity_correction"
        generator = PromptCapturingGenerator()

        extract_with_candidate_agents(
            {"transcript": "This is Robin Sampel, S-A-M-P-L-E.", "caller_id": "SAMPLE QUINN"},
            agents=build_default_agents(text_generator=generator),
            mode="candidate_agents",
            fallback_to_legacy=False,
        )

        prompt = next(call["prompt"] for call in generator.calls if call["agent_name"] == "name_correction")
        self.assertIn("Explicit spelling immediately after a spoken patient/caller name may correct the name.", prompt)
        self.assertIn("Caller ID may correct a clearly self-identified spoken name", prompt)

    def test_dual_correction_prompt_sends_spelling_rules_only_to_spelling_worker(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "split_identity_dual_correction"
        generator = PromptCapturingGenerator()

        result = extract_with_candidate_agents(
            {"transcript": "My name is Example, K-O-E-S-P-E-R.", "caller_id": "Wireless Caller"},
            agents=build_default_agents(text_generator=generator),
            mode="candidate_agents",
            fallback_to_legacy=False,
        )

        agent_names = [call["agent_name"] for call in generator.calls]
        self.assertIn("spelling_correction", agent_names)
        self.assertNotIn("caller_id_correction", agent_names)
        spelling_prompt = next(call["prompt"] for call in generator.calls if call["agent_name"] == "spelling_correction")
        name_prompt = next(call["prompt"] for call in generator.calls if call["agent_name"] == "name")
        spelling_payload = json.loads(spelling_prompt.rsplit("\n\nInput JSON:\n", 1)[1])
        self.assertIn("Explicit spelling immediately after a spoken patient/caller name may correct the name.", spelling_prompt)
        self.assertIn("Example spelling override:", spelling_prompt)
        self.assertNotIn("Caller ID may correct a clearly self-identified spoken name", spelling_prompt)
        self.assertNotIn("Explicit spelling immediately after a spoken patient/caller name may correct the name.", name_prompt)
        self.assertIn("spelled_sequences", spelling_payload)
        self.assertEqual(spelling_payload["caller_id"], "")
        self.assertEqual(result["patient_names"][0]["source"], "transcript_spelling_corrected")

    def test_dual_correction_prompt_sends_caller_id_rules_only_to_caller_id_worker(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "split_identity_dual_correction"
        generator = PromptCapturingGenerator()

        result = extract_with_candidate_agents(
            {"transcript": "This is Avery Exampel.", "caller_id": "AVERY EXAMPLE"},
            agents=build_default_agents(text_generator=generator),
            mode="candidate_agents",
            fallback_to_legacy=False,
        )

        agent_names = [call["agent_name"] for call in generator.calls]
        self.assertIn("caller_id_correction", agent_names)
        self.assertNotIn("spelling_correction", agent_names)
        caller_prompt = next(call["prompt"] for call in generator.calls if call["agent_name"] == "caller_id_correction")
        caller_payload = json.loads(caller_prompt.rsplit("\n\nInput JSON:\n", 1)[1])
        self.assertIn("Caller ID may correct a clearly self-identified spoken name", caller_prompt)
        self.assertIn("Example Caller ID corrections:", caller_prompt)
        self.assertNotIn("Explicit spelling immediately after a spoken patient/caller name may correct the name.", caller_prompt)
        self.assertEqual(caller_payload["caller_id"], "AVERY EXAMPLE")
        self.assertEqual(result["patient_names"][0]["source"], "caller_id_corrected")

    def test_dual_correction_selects_middle_initial_same_first_last_name_caller_id_candidate(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "split_identity_dual_correction"
        generator = PromptCapturingGenerator()

        result = extract_with_candidate_agents(
            {"transcript": "This is Quinn Exampel.", "caller_id": "QUINN L EXAMPLE"},
            agents=build_default_agents(text_generator=generator),
            mode="candidate_agents",
            fallback_to_legacy=False,
        )

        caller_prompt = next(call["prompt"] for call in generator.calls if call["agent_name"] == "caller_id_correction")
        caller_payload = json.loads(caller_prompt.rsplit("\n\nInput JSON:\n", 1)[1])
        self.assertEqual(caller_payload["caller_id"], "QUINN L EXAMPLE")
        self.assertEqual(caller_payload["name_candidates"][0]["raw"], "Quinn Exampel")
        self.assertEqual(caller_payload["name_candidates"][0]["value"], "Quinn Example")
        self.assertEqual(result["patient_names"][0]["value"], "Quinn Example")
        self.assertEqual(result["patient_names"][0]["source"], "caller_id_corrected")

    def test_dual_correction_accepts_sip_display_caller_id_with_number(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "split_identity_dual_correction"
        generator = PromptCapturingGenerator()

        result = extract_with_candidate_agents(
            {
                "transcript": "This is Quinn Exampel.",
                "caller_id": '"QUINN L EXAMPLE" <2025550124>',
            },
            agents=build_default_agents(text_generator=generator),
            mode="candidate_agents",
            fallback_to_legacy=False,
        )

        caller_prompt = next(call["prompt"] for call in generator.calls if call["agent_name"] == "caller_id_correction")
        caller_payload = json.loads(caller_prompt.rsplit("\n\nInput JSON:\n", 1)[1])
        self.assertEqual(caller_payload["caller_id"], "QUINN L EXAMPLE")
        self.assertEqual(caller_payload["name_candidates"][0]["raw"], "Quinn Exampel")
        self.assertEqual(caller_payload["name_candidates"][0]["value"], "Quinn Example")
        self.assertEqual(result["patient_names"][0]["value"], "Quinn Example")
        self.assertEqual(result["patient_names"][0]["source"], "caller_id_corrected")

    def test_dual_correction_accepts_parenthesized_phone_display_caller_id(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "split_identity_dual_correction"
        generator = PromptCapturingGenerator()

        result = extract_with_candidate_agents(
            {
                "transcript": "This is Quinn Exampel.",
                "caller_id": '"QUINN L EXAMPLE" (202) 555-0104',
            },
            agents=build_default_agents(text_generator=generator),
            mode="candidate_agents",
            fallback_to_legacy=False,
        )

        caller_prompt = next(call["prompt"] for call in generator.calls if call["agent_name"] == "caller_id_correction")
        caller_payload = json.loads(caller_prompt.rsplit("\n\nInput JSON:\n", 1)[1])
        self.assertEqual(caller_payload["caller_id"], "QUINN L EXAMPLE")
        self.assertEqual(caller_payload["name_candidates"][0]["raw"], "Quinn Exampel")
        self.assertEqual(caller_payload["name_candidates"][0]["value"], "Quinn Example")
        self.assertEqual(result["patient_names"][0]["value"], "Quinn Example")
        self.assertEqual(result["patient_names"][0]["source"], "caller_id_corrected")

    def test_dual_correction_accepts_last_first_trailing_initial_caller_id(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "split_identity_dual_correction"
        generator = PromptCapturingGenerator()

        result = extract_with_candidate_agents(
            {
                "transcript": "This is Taylor Exampel.",
                "caller_id": '"EXAMPLE TAYLOR P" (202) 555-0107',
            },
            agents=build_default_agents(text_generator=generator),
            mode="candidate_agents",
            fallback_to_legacy=False,
        )

        caller_prompt = next(call["prompt"] for call in generator.calls if call["agent_name"] == "caller_id_correction")
        caller_payload = json.loads(caller_prompt.rsplit("\n\nInput JSON:\n", 1)[1])
        self.assertEqual(caller_payload["caller_id"], "EXAMPLE TAYLOR P")
        self.assertEqual(caller_payload["name_candidates"][0]["raw"], "Taylor Exampel")
        self.assertEqual(caller_payload["name_candidates"][0]["value"], "Taylor Example")
        self.assertEqual(result["patient_names"][0]["value"], "Taylor Example")
        self.assertEqual(result["patient_names"][0]["source"], "caller_id_corrected")

    def test_dual_correction_selects_close_last_first_caller_id_candidate(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "split_identity_dual_correction"
        generator = PromptCapturingGenerator()

        result = extract_with_candidate_agents(
            {
                "transcript": "Hi, this is Morgan Sampel. My number is 202-555-0102.",
                "caller_id": '"SAMPLE MORGAN" (202) 555-0100',
            },
            agents=build_default_agents(text_generator=generator),
            mode="candidate_agents",
            fallback_to_legacy=False,
        )

        caller_prompt = next(call["prompt"] for call in generator.calls if call["agent_name"] == "caller_id_correction")
        caller_payload = json.loads(caller_prompt.rsplit("\n\nInput JSON:\n", 1)[1])
        self.assertEqual(caller_payload["caller_id"], "SAMPLE MORGAN")
        self.assertEqual(caller_payload["name_candidates"][0]["raw"], "Morgan Sampel")
        self.assertEqual(caller_payload["name_candidates"][0]["value"], "Morgan Sample")
        self.assertEqual(result["patient_names"][0]["value"], "Morgan Sample")
        self.assertEqual(result["patient_names"][0]["source"], "caller_id_corrected")

    def test_dual_correction_selects_last_name_only_caller_id_candidate_by_default(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "split_identity_dual_correction"
        generator = PromptCapturingGenerator()

        result = extract_with_candidate_agents(
            {"transcript": "This is Taylor Exampel.", "caller_id": "QUINN L EXAMPLE"},
            agents=build_default_agents(text_generator=generator),
            mode="candidate_agents",
            fallback_to_legacy=False,
        )

        caller_prompt = next(call["prompt"] for call in generator.calls if call["agent_name"] == "caller_id_correction")
        caller_payload = json.loads(caller_prompt.rsplit("\n\nInput JSON:\n", 1)[1])
        self.assertEqual(caller_payload["caller_id"], "QUINN L EXAMPLE")
        self.assertEqual(caller_payload["name_candidates"][0]["raw"], "Taylor Exampel")
        self.assertEqual(caller_payload["name_candidates"][0]["value"], "Taylor Example")
        self.assertEqual(result["patient_names"][0]["value"], "Taylor Example")
        self.assertEqual(result["patient_names"][0]["source"], "caller_id_corrected")

    def test_dual_correction_selects_last_name_only_sip_display_candidate_by_default(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "split_identity_dual_correction"
        generator = PromptCapturingGenerator()

        result = extract_with_candidate_agents(
            {
                "transcript": "This is Taylor Exampel.",
                "caller_id": '"QUINN L EXAMPLE" <2025550124>',
            },
            agents=build_default_agents(text_generator=generator),
            mode="candidate_agents",
            fallback_to_legacy=False,
        )

        caller_prompt = next(call["prompt"] for call in generator.calls if call["agent_name"] == "caller_id_correction")
        caller_payload = json.loads(caller_prompt.rsplit("\n\nInput JSON:\n", 1)[1])
        self.assertEqual(caller_payload["caller_id"], "QUINN L EXAMPLE")
        self.assertEqual(caller_payload["name_candidates"][0]["raw"], "Taylor Exampel")
        self.assertEqual(caller_payload["name_candidates"][0]["value"], "Taylor Example")
        self.assertEqual(result["patient_names"][0]["value"], "Taylor Example")
        self.assertEqual(result["patient_names"][0]["source"], "caller_id_corrected")

    def test_dual_correction_last_name_only_candidate_can_be_disabled(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "split_identity_dual_correction"
        os.environ["CANDIDATE_AGENT_CALLER_ID_LAST_NAME_ONLY_CORRECTION"] = "false"
        generator = PromptCapturingGenerator()

        result = extract_with_candidate_agents(
            {"transcript": "This is Taylor Exampel.", "caller_id": "QUINN L EXAMPLE"},
            agents=build_default_agents(text_generator=generator),
            mode="candidate_agents",
            fallback_to_legacy=False,
        )

        self.assertEqual(result["patient_names"][0]["value"], "Taylor Exampel")
        self.assertEqual(result["patient_names"][0]["source"], "transcript")
        self.assertEqual(result["name_correction_candidates"][0]["suggested_value"], "Taylor Example")

    def test_dual_correction_runs_both_workers_when_spelling_and_name_shaped_caller_id_exist(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "split_identity_dual_correction"
        generator = PromptCapturingGenerator()

        result = extract_with_candidate_agents(
            {"transcript": "This is Robin Sampel, S-A-M-P-L-E.", "caller_id": "SAMPLE QUINN"},
            agents=build_default_agents(text_generator=generator),
            mode="candidate_agents",
            fallback_to_legacy=False,
        )

        spelling_prompt = next(call["prompt"] for call in generator.calls if call["agent_name"] == "spelling_correction")
        caller_prompt = next(call["prompt"] for call in generator.calls if call["agent_name"] == "caller_id_correction")
        self.assertIn("Explicit spelling immediately after a spoken patient/caller name may correct the name.", spelling_prompt)
        self.assertNotIn("Caller ID may correct a clearly self-identified spoken name", spelling_prompt)
        self.assertIn("Caller ID may correct a clearly self-identified spoken name", caller_prompt)
        self.assertNotIn("Explicit spelling immediately after a spoken patient/caller name may correct the name.", caller_prompt)
        self.assertEqual(result["patient_names"][0]["source"], "transcript_spelling_corrected")
        self.assertEqual(result["name_correction_candidates"][0]["reason"], "last_name_phonetic_match")

    def test_organization_caller_id_is_not_passed_to_name_worker(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "split_identity_correction"
        generator = PromptCapturingGenerator()

        extract_with_candidate_agents(
            {"transcript": "This is Jordan Example calling.", "caller_id": "Example Clinic"},
            agents=build_default_agents(text_generator=generator),
            mode="candidate_agents",
            fallback_to_legacy=False,
        )

        prompt = next(call["prompt"] for call in generator.calls if call["agent_name"] == "name")
        payload = json.loads(prompt.rsplit("\n\nInput JSON:\n", 1)[1])
        self.assertNotIn("Caller ID may correct a clearly self-identified spoken name", prompt)
        self.assertEqual(payload["caller_id"], "")
        self.assertNotIn("name_correction", [call["agent_name"] for call in generator.calls])

    def test_dual_correction_skips_caller_id_worker_for_organization_caller_id(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "split_identity_dual_correction"
        generator = PromptCapturingGenerator()

        extract_with_candidate_agents(
            {"transcript": "This is Jordan Example calling.", "caller_id": "Example Clinic"},
            agents=build_default_agents(text_generator=generator),
            mode="candidate_agents",
            fallback_to_legacy=False,
        )

        prompt = next(call["prompt"] for call in generator.calls if call["agent_name"] == "name")
        payload = json.loads(prompt.rsplit("\n\nInput JSON:\n", 1)[1])
        self.assertEqual(payload["caller_id"], "")
        self.assertNotIn("caller_id_correction", [call["agent_name"] for call in generator.calls])

    def test_split_identity_agents_return_legacy_schema(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "split_identity"
        name_agent = CompactSelectingNameAgent()
        dob_agent = CompactSelectingDobAgent()

        result = extract_with_candidate_agents(
            {
                "transcript": (
                    "This is Jordan Example calling. "
                    "Please call me back at 202-555-0108. Date of birth is 1/4/72."
                ),
                "caller_id": "SYNTHETIC CALLER",
            },
            agents={
                "numbers": CompactSelectingNumbersAgent(),
                "name": name_agent,
                "dob": dob_agent,
            },
            mode="candidate_agents",
            fallback_to_legacy=False,
        )

        self.assertEqual(result["patient_names"][0]["value"], "Jordan Example")
        self.assertEqual(result["dob_candidates"][0]["normalized"], "01/04/1972")
        self.assertEqual(result["callback_numbers"][0]["normalized"], "2025550108")
        self.assertIn("name_candidates", name_agent.last_payload)
        self.assertNotIn("dob_candidates", name_agent.last_payload)
        self.assertIn("dob_candidates", dob_agent.last_payload)

    def test_parallel_http_execution_starts_three_agents_concurrently(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "split_identity"
        os.environ["CANDIDATE_AGENT_EXECUTION"] = "parallel_http"
        started = []
        release = threading.Event()

        result = extract_with_candidate_agents(
            {
                "transcript": (
                    "This is Jordan Example calling. "
                    "Please call me back at 202-555-0108. Date of birth is 1/4/72."
                ),
            },
            agents={
                "numbers": BarrierAgent(
                    "numbers",
                    {"callback_ids": ["number:0"], "fax_ids": [], "uncertain_ids": [], "errors": []},
                    started,
                    release,
                ),
                "name": BarrierAgent(
                    "name",
                    {"name_ids": ["name:0"], "name_correction_ids": [], "errors": []},
                    started,
                    release,
                ),
                "dob": BarrierAgent("dob", {"dob_ids": ["dob:0"], "errors": []}, started, release),
            },
            mode="candidate_agents",
            fallback_to_legacy=False,
        )

        self.assertEqual(set(started), {"numbers", "name", "dob"})
        self.assertEqual(result["callback_numbers"][0]["normalized"], "2025550108")
        health = candidate_agent_health(agents_loaded=["numbers", "name", "dob"])
        self.assertEqual(health["candidate_agent_topology"], "split_identity")
        self.assertEqual(health["agent_execution_mode"], "parallel_http")
        self.assertIn("numbers", health["last_agent_timings_ms"])

    def test_split_identity_correction_selects_corrected_name_before_common_name(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "split_identity_correction"
        name_agent = RecordingAgent(
            "name",
            {"name_ids": ["name:0"], "name_correction_ids": [], "errors": []},
        )
        correction_agent = RecordingAgent(
            "name_correction",
            {"name_ids": ["name:1"], "name_correction_ids": [], "errors": []},
        )

        result = extract_with_candidate_agents(
            {"transcript": "This is Bailey Sampel, S-A-M-P-L-E.", "caller_id": "Wireless Caller"},
            agents={
                "numbers": FailIfCalledAgent(),
                "name": name_agent,
                "name_correction": correction_agent,
                "dob": FailIfCalledAgent(),
            },
            mode="candidate_agents",
            fallback_to_legacy=False,
        )

        self.assertEqual(len(result["patient_names"]), 1)
        self.assertEqual(result["patient_names"][0]["raw"], "Bailey Sampel")
        self.assertEqual(result["patient_names"][0]["value"], "Bailey Sample")
        self.assertEqual(result["patient_names"][0]["source"], "transcript_spelling_corrected")
        self.assertEqual(name_agent.last_payload["name_candidates"][0]["source"], "self_identification")
        self.assertEqual(correction_agent.last_payload["name_candidates"][0]["source"], "transcript_spelling_corrected")
        self.assertIn("spelled_sequences", correction_agent.last_payload)

    def test_split_identity_correction_parallel_execution_starts_four_relevant_agents(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "split_identity_correction"
        os.environ["CANDIDATE_AGENT_EXECUTION"] = "parallel_http"
        started = []
        release = threading.Event()

        result = extract_with_candidate_agents(
            {
                "transcript": (
                    "This is Bailey Sampel, S-A-M-P-L-E. "
                    "Date of birth is 1/4/72. Please call me back at 202-555-0108."
                ),
                "caller_id": "Wireless Caller",
            },
            agents={
                "numbers": BarrierAgent(
                    "numbers",
                    {"callback_ids": ["number:0"], "fax_ids": [], "uncertain_ids": [], "errors": []},
                    started,
                    release,
                    expected=4,
                ),
                "name": BarrierAgent(
                    "name",
                    {"name_ids": ["name:0"], "name_correction_ids": [], "errors": []},
                    started,
                    release,
                    expected=4,
                ),
                "name_correction": BarrierAgent(
                    "name_correction",
                    {"name_ids": ["name:1"], "name_correction_ids": [], "errors": []},
                    started,
                    release,
                    expected=4,
                ),
                "dob": BarrierAgent("dob", {"dob_ids": ["dob:0"], "errors": []}, started, release, expected=4),
            },
            mode="candidate_agents",
            fallback_to_legacy=False,
        )

        self.assertEqual(set(started), {"numbers", "name", "name_correction", "dob"})
        self.assertEqual(result["callback_numbers"][0]["normalized"], "2025550108")
        self.assertEqual(result["dob_candidates"][0]["normalized"], "01/04/1972")
        self.assertEqual(result["patient_names"][0]["value"], "Bailey Sample")

    def test_split_identity_dual_correction_parallel_execution_starts_five_relevant_agents(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "split_identity_dual_correction"
        os.environ["CANDIDATE_AGENT_EXECUTION"] = "parallel_http"
        started = []
        release = threading.Event()

        result = extract_with_candidate_agents(
            {
                "transcript": (
                    "This is Robin Sampel, S-A-M-P-L-E. "
                    "Date of birth is 1/4/72. Please call me back at 202-555-0108."
                ),
                "caller_id": "SAMPLE QUINN",
            },
            agents={
                "numbers": BarrierAgent(
                    "numbers",
                    {"callback_ids": ["number:0"], "fax_ids": [], "uncertain_ids": [], "errors": []},
                    started,
                    release,
                    expected=5,
                ),
                "name": BarrierAgent(
                    "name",
                    {"name_ids": ["name:0"], "name_correction_ids": [], "errors": []},
                    started,
                    release,
                    expected=5,
                ),
                "spelling_correction": BarrierAgent(
                    "spelling_correction",
                    {"name_ids": ["name:1"], "name_correction_ids": [], "errors": []},
                    started,
                    release,
                    expected=5,
                ),
                "caller_id_correction": BarrierAgent(
                    "caller_id_correction",
                    {"name_ids": [], "name_correction_ids": ["name_correction:0"], "errors": []},
                    started,
                    release,
                    expected=5,
                ),
                "dob": BarrierAgent("dob", {"dob_ids": ["dob:0"], "errors": []}, started, release, expected=5),
            },
            mode="candidate_agents",
            fallback_to_legacy=False,
        )

        self.assertEqual(set(started), {"numbers", "name", "spelling_correction", "caller_id_correction", "dob"})
        self.assertEqual(result["callback_numbers"][0]["normalized"], "2025550108")
        self.assertEqual(result["dob_candidates"][0]["normalized"], "01/04/1972")
        self.assertEqual(result["patient_names"][0]["source"], "transcript_spelling_corrected")
        self.assertEqual(result["name_correction_candidates"][0]["reason"], "last_name_phonetic_match")

    def test_split_identity_correction_skips_correction_when_no_spelling_or_name_shaped_caller_id(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "split_identity_correction"
        os.environ["CANDIDATE_AGENT_EXECUTION"] = "parallel_http"
        name_agent = CompactSelectingNameAgent()

        result = extract_with_candidate_agents(
            {"transcript": "This is Jordan Example calling.", "caller_id": "Wireless Caller"},
            agents={
                "numbers": FailIfCalledAgent(),
                "name": name_agent,
                "name_correction": FailIfCalledAgent(),
                "dob": FailIfCalledAgent(),
            },
            mode="candidate_agents",
            fallback_to_legacy=False,
        )

        self.assertEqual(result["patient_names"][0]["value"], "Jordan Example")
        health = candidate_agent_health(agents_loaded=["numbers", "name", "name_correction", "dob"])
        self.assertEqual(set(health["last_agent_skipped"]), {"numbers", "name_correction", "dob"})
        self.assertEqual(health["last_agent_timings_ms"]["name_correction"], 0)

    def test_split_identity_dual_correction_skips_irrelevant_correction_workers(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "split_identity_dual_correction"
        os.environ["CANDIDATE_AGENT_EXECUTION"] = "parallel_http"
        name_agent = CompactSelectingNameAgent()

        result = extract_with_candidate_agents(
            {"transcript": "This is Jordan Example calling.", "caller_id": "Wireless Caller"},
            agents={
                "numbers": FailIfCalledAgent(),
                "name": name_agent,
                "spelling_correction": FailIfCalledAgent(),
                "caller_id_correction": FailIfCalledAgent(),
                "dob": FailIfCalledAgent(),
            },
            mode="candidate_agents",
            fallback_to_legacy=False,
        )

        self.assertEqual(result["patient_names"][0]["value"], "Jordan Example")
        health = candidate_agent_health(
            agents_loaded=["numbers", "name", "spelling_correction", "caller_id_correction", "dob"]
        )
        self.assertEqual(set(health["last_agent_skipped"]), {"numbers", "spelling_correction", "caller_id_correction", "dob"})
        self.assertEqual(health["last_agent_timings_ms"]["spelling_correction"], 0)
        self.assertEqual(health["last_agent_timings_ms"]["caller_id_correction"], 0)

    def test_subject_fallback_topology_splits_name_payloads_and_prefers_subject(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "split_identity_subject_fallback_dual_correction"
        os.environ["CANDIDATE_AGENT_EXECUTION"] = "parallel_http"
        subject_agent = SubjectSelectingNameAgent()
        fallback_agent = RecordingAgent(
            "caller_name_fallback",
            {"name_ids": ["name:0"], "name_correction_ids": [], "errors": []},
        )

        result = extract_with_candidate_agents(
            {
                "transcript": (
                    "Hello, this is Casey Example from Sample Example Clinic. "
                    "I am calling from my husband Morgan Example about his knee."
                ),
                "caller_id": "Unavailable",
            },
            agents={
                "numbers": FailIfCalledAgent(),
                "subject_name": subject_agent,
                "caller_name_fallback": fallback_agent,
                "spelling_correction": FailIfCalledAgent(),
                "caller_id_correction": FailIfCalledAgent(),
                "dob": FailIfCalledAgent(),
            },
            mode="candidate_agents",
            fallback_to_legacy=False,
        )

        self.assertEqual(result["patient_names"], [
            {
                "raw": "Morgan Example",
                "value": "Morgan Example",
                "evidence_text": "my husband Morgan Example",
                "source": "relationship_subject",
                "caller_id_used": "",
            }
        ])
        subject_names = {item["raw"]: item for item in subject_agent.last_payload["name_candidates"]}
        fallback_names = {item["raw"]: item for item in fallback_agent.last_payload["name_candidates"]}
        self.assertIn("Morgan Example", subject_names)
        self.assertNotIn("Casey Example", subject_names)
        self.assertIn("my husband Morgan Example", subject_names["Morgan Example"]["sentence_context"])
        self.assertIn("Casey Example", fallback_names)
        self.assertNotIn("Morgan Example", fallback_names)
        health = candidate_agent_health(
            agents_loaded=[
                "numbers",
                "subject_name",
                "caller_name_fallback",
                "spelling_correction",
                "caller_id_correction",
                "dob",
            ]
        )
        self.assertNotIn("subject_name", health["last_agent_auto_accepted"])
        self.assertNotIn("caller_name_fallback", health["last_agent_auto_accepted"])
        self.assertNotIn("spelling_correction", health["last_agent_auto_accepted"])

    def test_subject_fallback_topology_routes_calling_about_patient_to_subject_worker(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "split_identity_subject_fallback_dual_correction"
        os.environ["CANDIDATE_AGENT_EXECUTION"] = "parallel_http"
        subject_agent = SubjectSelectingNameAgent()
        fallback_agent = RecordingAgent(
            "caller_name_fallback",
            {"name_ids": ["name:0"], "name_correction_ids": [], "errors": []},
        )

        result = extract_with_candidate_agents(
            {
                "transcript": (
                    "Hello, my name is Taylor Example with Sample Home Health. "
                    "I am calling about patient Casey Example."
                ),
                "caller_id": "Unavailable",
            },
            agents={
                "numbers": FailIfCalledAgent(),
                "subject_name": subject_agent,
                "caller_name_fallback": fallback_agent,
                "spelling_correction": FailIfCalledAgent(),
                "caller_id_correction": FailIfCalledAgent(),
                "dob": FailIfCalledAgent(),
            },
            mode="candidate_agents",
            fallback_to_legacy=False,
        )

        subject_names = {item["raw"] for item in subject_agent.last_payload["name_candidates"]}
        fallback_names = {item["raw"] for item in fallback_agent.last_payload["name_candidates"]}
        self.assertIn("Casey Example", subject_names)
        self.assertNotIn("Taylor Example", subject_names)
        self.assertIn("Taylor Example", fallback_names)
        self.assertNotIn("Casey Example", fallback_names)
        self.assertEqual(result["patient_names"][0]["value"], "Casey Example")
        self.assertEqual(result["patient_names"][0]["source"], "relationship_subject")

    def test_subject_fallback_topology_subject_selection_beats_caller_id_correction(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "split_identity_subject_fallback_dual_correction"
        os.environ["CANDIDATE_AGENT_EXECUTION"] = "parallel_http"
        subject_agent = SubjectSelectingNameAgent()
        fallback_agent = RecordingAgent(
            "caller_name_fallback",
            {"name_ids": ["name:0"], "name_correction_ids": [], "errors": []},
        )
        caller_id_agent = CompactSelectingNameCorrectionAgent()

        result = extract_with_candidate_agents(
            {
                "transcript": "This is Taylor Exampel. I'm calling from my husband Morgan Example.",
                "caller_id": "QUINN L EXAMPLE",
            },
            agents={
                "numbers": FailIfCalledAgent(),
                "subject_name": subject_agent,
                "caller_name_fallback": fallback_agent,
                "spelling_correction": FailIfCalledAgent(),
                "caller_id_correction": caller_id_agent,
                "dob": FailIfCalledAgent(),
            },
            mode="candidate_agents",
            fallback_to_legacy=False,
        )

        caller_id_sources = {
            candidate.get("source")
            for candidate in caller_id_agent.last_payload.get("name_candidates", [])
        }
        self.assertIn("caller_id_corrected", caller_id_sources)
        self.assertEqual(result["patient_names"][0]["value"], "Morgan Example")
        self.assertEqual(result["patient_names"][0]["source"], "relationship_subject")

    def test_subject_fallback_topology_request_subject_beats_caller_staff_name(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "split_identity_subject_fallback_dual_correction"
        os.environ["CANDIDATE_AGENT_EXECUTION"] = "parallel_http"
        subject_agent = SubjectSelectingNameAgent()
        fallback_agent = RecordingAgent(
            "caller_name_fallback",
            {"name_ids": ["name:0"], "name_correction_ids": [], "errors": []},
        )

        result = extract_with_candidate_agents(
            {
                "transcript": (
                    "Hi, this is Bailey with Example Clinic prior authorizations. "
                    "I'm talking about an availability request that we received on Taylor Example."
                ),
                "caller_id": "SYNTHETIC CALLER",
            },
            agents={
                "numbers": FailIfCalledAgent(),
                "subject_name": subject_agent,
                "caller_name_fallback": fallback_agent,
                "spelling_correction": FailIfCalledAgent(),
                "caller_id_correction": FailIfCalledAgent(),
                "dob": FailIfCalledAgent(),
            },
            mode="candidate_agents",
            fallback_to_legacy=False,
        )

        subject_names = {item["raw"]: item for item in subject_agent.last_payload["name_candidates"]}
        self.assertIn("Taylor Example", subject_names)
        self.assertNotIn("An Availability Request", subject_names)
        self.assertEqual(result["patient_names"][0]["value"], "Taylor Example")
        self.assertEqual(result["patient_names"][0]["source"], "relationship_subject")

    def test_subject_fallback_topology_routes_broad_recall_to_both_name_workers(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "split_identity_subject_fallback_dual_correction"
        os.environ["CANDIDATE_AGENT_EXECUTION"] = "parallel_http"
        subject_agent = BroadRecallSelectingNameAgent()
        fallback_agent = RecordingAgent(
            "caller_name_fallback",
            {"name_ids": [], "name_correction_ids": [], "errors": []},
        )

        result = extract_with_candidate_agents(
            {
                "transcript": (
                    "Hello, this is a synthetic office message. "
                    "The message concerns Casey Sample and the appointment."
                ),
                "caller_id": "SYNTHETIC CALLER",
            },
            agents={
                "numbers": FailIfCalledAgent(),
                "subject_name": subject_agent,
                "caller_name_fallback": fallback_agent,
                "spelling_correction": FailIfCalledAgent(),
                "caller_id_correction": FailIfCalledAgent(),
                "dob": FailIfCalledAgent(),
            },
            mode="candidate_agents",
            fallback_to_legacy=False,
        )

        subject_sources = {
            item["raw"]: item["source"]
            for item in subject_agent.last_payload["name_candidates"]
        }
        fallback_sources = {
            item["raw"]: item["source"]
            for item in fallback_agent.last_payload["name_candidates"]
        }
        self.assertEqual(subject_sources["Casey Sample"], "broad_name_recall")
        self.assertEqual(fallback_sources["Casey Sample"], "broad_name_recall")
        self.assertEqual(result["patient_names"][0]["value"], "Casey Sample")
        self.assertEqual(result["patient_names"][0]["source"], "broad_name_recall")
        health = candidate_agent_health(
            agents_loaded=[
                "numbers",
                "subject_name",
                "caller_name_fallback",
                "spelling_correction",
                "caller_id_correction",
                "dob",
            ]
        )
        self.assertTrue(health["candidate_agent_broad_name_recall"])
        self.assertEqual(health["last_name_candidate_counts_by_source"]["broad_name_recall"], 1)
        self.assertGreaterEqual(health["last_name_candidate_total"], 1)

    def test_subject_fallback_topology_caller_id_correction_wins_without_subject(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "split_identity_subject_fallback_dual_correction"
        os.environ["CANDIDATE_AGENT_EXECUTION"] = "parallel_http"
        fallback_agent = RecordingAgent(
            "caller_name_fallback",
            {"name_ids": ["name:0"], "name_correction_ids": [], "errors": []},
        )
        caller_id_agent = CompactSelectingNameCorrectionAgent()

        result = extract_with_candidate_agents(
            {
                "transcript": "Hey, it is Jordan Exampel again.",
                "caller_id": '"JORDAN EXAMPLE" (202) 555-0102',
            },
            agents={
                "numbers": FailIfCalledAgent(),
                "subject_name": FailIfCalledAgent(),
                "caller_name_fallback": fallback_agent,
                "spelling_correction": FailIfCalledAgent(),
                "caller_id_correction": caller_id_agent,
                "dob": FailIfCalledAgent(),
            },
            mode="candidate_agents",
            fallback_to_legacy=False,
        )

        self.assertEqual(fallback_agent.last_payload["name_candidates"][0]["value"], "Jordan Exampel")
        self.assertEqual(caller_id_agent.last_payload["name_candidates"][0]["value"], "Jordan Example")
        self.assertEqual(result["patient_names"][0]["value"], "Jordan Example")
        self.assertEqual(result["patient_names"][0]["source"], "caller_id_corrected")

    def test_subject_fallback_topology_caller_id_correction_survives_im_sorry_false_name(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "split_identity_subject_fallback_dual_correction"
        os.environ["CANDIDATE_AGENT_EXECUTION"] = "parallel_http"
        fallback_agent = RecordingAgent(
            "caller_name_fallback",
            {"name_ids": ["name:0"], "name_correction_ids": [], "errors": []},
        )
        caller_id_agent = CompactSelectingNameCorrectionAgent()

        result = extract_with_candidate_agents(
            {
                "transcript": "Hey, Bailey, I'm sorry. It's Jordan Exampel again. Can you call me?",
                "caller_id": '"JORDAN EXAMPLE" (202) 555-0102',
            },
            agents={
                "numbers": FailIfCalledAgent(),
                "subject_name": FailIfCalledAgent(),
                "caller_name_fallback": fallback_agent,
                "spelling_correction": FailIfCalledAgent(),
                "caller_id_correction": caller_id_agent,
                "dob": FailIfCalledAgent(),
            },
            mode="candidate_agents",
            fallback_to_legacy=False,
        )

        fallback_names = {item["raw"] for item in fallback_agent.last_payload["name_candidates"]}
        self.assertNotIn("Sorry", fallback_names)
        self.assertIn("Jordan Exampel", fallback_names)
        self.assertEqual(caller_id_agent.last_payload["name_candidates"][0]["value"], "Jordan Example")
        self.assertEqual(result["patient_names"][0]["value"], "Jordan Example")
        self.assertEqual(result["patient_names"][0]["source"], "caller_id_corrected")

    def test_subject_fallback_topology_uses_fallback_when_subject_is_empty(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "split_identity_subject_fallback_dual_correction"
        os.environ["CANDIDATE_AGENT_EXECUTION"] = "parallel_http"
        fallback_agent = RecordingAgent(
            "caller_name_fallback",
            {"name_ids": ["name:0"], "name_correction_ids": [], "errors": []},
        )

        result = extract_with_candidate_agents(
            {"transcript": "Hi, this is Bailey Sample.", "caller_id": "SAMPLE,BAILEY"},
            agents={
                "numbers": FailIfCalledAgent(),
                "subject_name": FailIfCalledAgent(),
                "caller_name_fallback": fallback_agent,
                "spelling_correction": FailIfCalledAgent(),
                "caller_id_correction": FailIfCalledAgent(),
                "dob": FailIfCalledAgent(),
            },
            mode="candidate_agents",
            fallback_to_legacy=False,
        )

        self.assertEqual(result["patient_names"][0]["value"], "Bailey Sample")
        self.assertEqual(fallback_agent.last_payload["name_candidates"][0]["source"], "self_identification")
        health = candidate_agent_health(
            agents_loaded=[
                "numbers",
                "subject_name",
                "caller_name_fallback",
                "spelling_correction",
                "caller_id_correction",
                "dob",
            ]
        )
        self.assertIn("subject_name", health["last_agent_skipped"])
        self.assertEqual(health["last_agent_timings_ms"]["subject_name"], 0)

    def test_subject_fallback_topology_spelling_correction_still_wins(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "split_identity_subject_fallback_dual_correction"
        os.environ["CANDIDATE_AGENT_EXECUTION"] = "parallel_http"
        os.environ["CANDIDATE_AGENT_AUTO_ACCEPT_TRANSCRIPT_SPELLING"] = "false"
        subject_agent = RecordingAgent("subject_name", {"name_ids": ["name:0"], "name_correction_ids": [], "errors": []})
        fallback_agent = RecordingAgent(
            "caller_name_fallback",
            {"name_ids": ["name:0"], "name_correction_ids": [], "errors": []},
        )
        spelling_agent = RecordingAgent(
            "spelling_correction",
            {"name_ids": ["name:1"], "name_correction_ids": [], "errors": []},
        )

        result = extract_with_candidate_agents(
            {"transcript": "This is Bailey Sampel, S-A-M-P-L-E.", "caller_id": "Wireless Caller"},
            agents={
                "numbers": FailIfCalledAgent(),
                "subject_name": subject_agent,
                "caller_name_fallback": fallback_agent,
                "spelling_correction": spelling_agent,
                "caller_id_correction": FailIfCalledAgent(),
                "dob": FailIfCalledAgent(),
            },
            mode="candidate_agents",
            fallback_to_legacy=False,
        )

        self.assertEqual(result["patient_names"][0]["raw"], "Bailey Sampel")
        self.assertEqual(result["patient_names"][0]["value"], "Bailey Sample")
        self.assertEqual(result["patient_names"][0]["source"], "transcript_spelling_corrected")
        self.assertEqual(spelling_agent.calls, 1)

    def test_auto_accept_disabled_by_default_still_runs_dob_and_spelling_workers(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "split_identity_dual_correction"
        os.environ["CANDIDATE_AGENT_EXECUTION"] = "parallel_http"
        name_agent = RecordingAgent("name", {"name_ids": ["name:0"], "name_correction_ids": [], "errors": []})
        spelling_agent = RecordingAgent(
            "spelling_correction",
            {"name_ids": ["name:1"], "name_correction_ids": [], "errors": []},
        )
        dob_agent = RecordingAgent("dob", {"dob_ids": ["dob:0"], "errors": []})

        result = extract_with_candidate_agents(
            {"transcript": "This is Bailey Sampel, S-A-M-P-L-E. Date of birth is 1/4/72."},
            agents={
                "numbers": FailIfCalledAgent(),
                "name": name_agent,
                "spelling_correction": spelling_agent,
                "caller_id_correction": FailIfCalledAgent(),
                "dob": dob_agent,
            },
            mode="candidate_agents",
            fallback_to_legacy=False,
        )

        self.assertEqual(spelling_agent.calls, 1)
        self.assertEqual(dob_agent.calls, 1)
        self.assertEqual(result["patient_names"][0]["source"], "transcript_spelling_corrected")
        self.assertEqual(result["dob_candidates"][0]["normalized"], "01/04/1972")
        health = candidate_agent_health(
            agents_loaded=["numbers", "name", "spelling_correction", "caller_id_correction", "dob"]
        )
        self.assertFalse(health["candidate_agent_auto_accept_deterministic"])
        self.assertEqual(health["last_agent_auto_accepted"], [])

    def test_auto_accept_single_high_confidence_dob_skips_dob_worker(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "split_identity"
        os.environ["CANDIDATE_AGENT_EXECUTION"] = "parallel_http"
        os.environ["CANDIDATE_AGENT_AUTO_ACCEPT_DETERMINISTIC"] = "true"

        result = extract_with_candidate_agents(
            {"transcript": "Date of birth is 1/4/72."},
            agents={
                "numbers": FailIfCalledAgent(),
                "name": FailIfCalledAgent(),
                "dob": FailIfCalledAgent(),
            },
            mode="candidate_agents",
            fallback_to_legacy=False,
        )

        self.assertEqual(result["dob_candidates"][0]["normalized"], "01/04/1972")
        health = candidate_agent_health(agents_loaded=["numbers", "name", "dob"])
        self.assertEqual(health["last_agent_auto_accepted"], ["dob"])
        self.assertEqual(set(health["last_agent_skipped"]), {"numbers", "name", "dob"})
        self.assertEqual(health["last_agent_timings_ms"]["dob"], 0)

    def test_auto_accept_multiple_dobs_still_runs_dob_worker(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "split_identity"
        os.environ["CANDIDATE_AGENT_EXECUTION"] = "parallel_http"
        os.environ["CANDIDATE_AGENT_AUTO_ACCEPT_DETERMINISTIC"] = "true"
        dob_agent = RecordingAgent("dob", {"dob_ids": ["dob:0"], "errors": []})

        result = extract_with_candidate_agents(
            {"transcript": "Date of birth is 1/4/72. DOB 2/5/73."},
            agents={
                "numbers": FailIfCalledAgent(),
                "name": FailIfCalledAgent(),
                "dob": dob_agent,
            },
            mode="candidate_agents",
            fallback_to_legacy=False,
        )

        self.assertEqual(dob_agent.calls, 1)
        self.assertEqual(result["dob_candidates"][0]["normalized"], "01/04/1972")
        health = candidate_agent_health(agents_loaded=["numbers", "name", "dob"])
        self.assertEqual(health["last_agent_auto_accepted"], [])

    def test_auto_accept_single_spelling_candidate_skips_spelling_worker(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "split_identity_dual_correction"
        os.environ["CANDIDATE_AGENT_EXECUTION"] = "parallel_http"
        os.environ["CANDIDATE_AGENT_AUTO_ACCEPT_DETERMINISTIC"] = "true"
        name_agent = RecordingAgent("name", {"name_ids": ["name:0"], "name_correction_ids": [], "errors": []})

        result = extract_with_candidate_agents(
            {"transcript": "This is Bailey Sampel, S-A-M-P-L-E.", "caller_id": "Wireless Caller"},
            agents={
                "numbers": FailIfCalledAgent(),
                "name": name_agent,
                "spelling_correction": FailIfCalledAgent(),
                "caller_id_correction": FailIfCalledAgent(),
                "dob": FailIfCalledAgent(),
            },
            mode="candidate_agents",
            fallback_to_legacy=False,
        )

        self.assertEqual(name_agent.calls, 1)
        self.assertEqual(result["patient_names"][0]["value"], "Bailey Sample")
        self.assertEqual(result["patient_names"][0]["source"], "transcript_spelling_corrected")
        health = candidate_agent_health(
            agents_loaded=["numbers", "name", "spelling_correction", "caller_id_correction", "dob"]
        )
        self.assertEqual(health["last_agent_auto_accepted"], ["spelling_correction"])
        self.assertIn("spelling_correction", health["last_agent_skipped"])
        self.assertEqual(health["last_agent_timings_ms"]["spelling_correction"], 0)

    def test_auto_accept_interleaved_spelled_first_and_last_name(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "split_identity_dual_correction"
        os.environ["CANDIDATE_AGENT_EXECUTION"] = "parallel_http"
        os.environ["CANDIDATE_AGENT_AUTO_ACCEPT_DETERMINISTIC"] = "true"
        name_agent = RecordingAgent("name", {"name_ids": ["name:0"], "name_correction_ids": [], "errors": []})

        result = extract_with_candidate_agents(
            {
                "transcript": (
                    "call me back this is Avery A-V-E-R-Y Exampel E-X-A-M-P-L-E "
                    "and my number is 202-555-0123 thank you"
                ),
                "caller_id": "Wireless Caller",
            },
            agents={
                "numbers": CompactSelectingNumbersAgent(),
                "name": name_agent,
                "spelling_correction": FailIfCalledAgent(),
                "caller_id_correction": FailIfCalledAgent(),
                "dob": FailIfCalledAgent(),
            },
            mode="candidate_agents",
            fallback_to_legacy=False,
        )

        self.assertEqual(name_agent.calls, 1)
        self.assertEqual(result["patient_names"][0]["raw"], "Avery Exampel")
        self.assertEqual(result["patient_names"][0]["value"], "Avery Example")
        self.assertEqual(result["patient_names"][0]["source"], "transcript_spelling_corrected")
        self.assertEqual(result["callback_numbers"][0]["formatted"], "(202) 555-0123")
        health = candidate_agent_health(
            agents_loaded=["numbers", "name", "spelling_correction", "caller_id_correction", "dob"]
        )
        self.assertEqual(health["last_agent_auto_accepted"], ["spelling_correction"])
        self.assertIn("spelling_correction", health["last_agent_skipped"])
        self.assertEqual(health["last_agent_timings_ms"]["spelling_correction"], 0)

    def test_auto_accept_full_transcript_interleaved_spelled_name(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "split_identity_dual_correction"
        os.environ["CANDIDATE_AGENT_EXECUTION"] = "parallel_http"
        os.environ["CANDIDATE_AGENT_AUTO_ACCEPT_DETERMINISTIC"] = "true"
        name_agent = RecordingAgent("name", {"name_ids": ["name:0"], "name_correction_ids": [], "errors": []})

        result = extract_with_candidate_agents(
            {"transcript": FULL_SYNTHETIC_TRANSCRIPT, "caller_id": "Wireless Caller"},
            agents={
                "numbers": CompactSelectingNumbersAgent(),
                "name": name_agent,
                "spelling_correction": FailIfCalledAgent(),
                "caller_id_correction": FailIfCalledAgent(),
                "dob": FailIfCalledAgent(),
            },
            mode="candidate_agents",
            fallback_to_legacy=False,
        )

        self.assertEqual(name_agent.calls, 1)
        self.assertEqual(result["patient_names"][0]["raw"], "Avery Exampel")
        self.assertEqual(result["patient_names"][0]["value"], "Avery Example")
        self.assertEqual(result["patient_names"][0]["source"], "transcript_spelling_corrected")
        self.assertEqual(result["callback_numbers"][0]["formatted"], "(202) 555-0123")
        health = candidate_agent_health(
            agents_loaded=["numbers", "name", "spelling_correction", "caller_id_correction", "dob"]
        )
        self.assertEqual(health["last_agent_auto_accepted"], ["spelling_correction"])
        self.assertIn("spelling_correction", health["last_agent_skipped"])
        self.assertEqual(health["last_agent_timings_ms"]["spelling_correction"], 0)

    def test_auto_accept_single_self_identification_skips_name_worker_when_enabled(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "split_identity_dual_correction"
        os.environ["CANDIDATE_AGENT_EXECUTION"] = "parallel_http"
        os.environ["CANDIDATE_AGENT_AUTO_ACCEPT_DETERMINISTIC"] = "true"
        os.environ["CANDIDATE_AGENT_AUTO_ACCEPT_SELF_IDENTIFICATION"] = "true"

        result = extract_with_candidate_agents(
            {"transcript": "and my name is Casey Example.", "caller_id": "Wireless Caller"},
            agents={
                "numbers": FailIfCalledAgent(),
                "name": FailIfCalledAgent(),
                "spelling_correction": FailIfCalledAgent(),
                "caller_id_correction": FailIfCalledAgent(),
                "dob": FailIfCalledAgent(),
            },
            mode="candidate_agents",
            fallback_to_legacy=False,
        )

        self.assertEqual(result["patient_names"][0]["raw"], "Casey Example")
        self.assertEqual(result["patient_names"][0]["source"], "transcript")
        health = candidate_agent_health(
            agents_loaded=["numbers", "name", "spelling_correction", "caller_id_correction", "dob"]
        )
        self.assertEqual(health["last_agent_auto_accepted"], ["name"])
        self.assertIn("name", health["last_agent_skipped"])
        self.assertEqual(health["last_agent_timings_ms"]["name"], 0)
        self.assertTrue(health["candidate_agent_auto_accept_self_identification"])

    def test_auto_accept_self_identification_with_matching_last_first_caller_id(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "split_identity_dual_correction"
        os.environ["CANDIDATE_AGENT_EXECUTION"] = "parallel_http"
        os.environ["CANDIDATE_AGENT_AUTO_ACCEPT_DETERMINISTIC"] = "true"
        os.environ["CANDIDATE_AGENT_AUTO_ACCEPT_SELF_IDENTIFICATION"] = "true"

        result = extract_with_candidate_agents(
            {"transcript": "Hi, this is Jordan Example.", "caller_id": "EXAMPLE,JORDAN"},
            agents={
                "numbers": FailIfCalledAgent(),
                "name": FailIfCalledAgent(),
                "spelling_correction": FailIfCalledAgent(),
                "caller_id_correction": FailIfCalledAgent(),
                "dob": FailIfCalledAgent(),
            },
            mode="candidate_agents",
            fallback_to_legacy=False,
        )

        self.assertEqual(result["patient_names"][0]["raw"], "Jordan Example")
        self.assertEqual(result["patient_names"][0]["source"], "transcript")
        health = candidate_agent_health(
            agents_loaded=["numbers", "name", "spelling_correction", "caller_id_correction", "dob"]
        )
        self.assertEqual(health["last_agent_auto_accepted"], ["name"])

    def test_subject_name_candidate_beats_caller_name_candidate(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "split_identity_dual_correction"
        os.environ["CANDIDATE_AGENT_EXECUTION"] = "parallel_http"
        os.environ["CANDIDATE_AGENT_AUTO_ACCEPT_DETERMINISTIC"] = "true"
        name_agent = SubjectSelectingNameAgent()

        result = extract_with_candidate_agents(
            {
                "transcript": (
                    "Hello, my name is Casey. I'm calling from Example Clinic, Example Clinic, Example Clinic, and Example Clinic. "
                    "I was calling in regards to a client. His name is Morgan Example. If someone can please "
                    "give me a call back at 202-555-0129. Thank you."
                ),
                "caller_id": "Unavailable",
            },
            agents={
                "numbers": CompactSelectingNumbersAgent(),
                "name": name_agent,
                "spelling_correction": FailIfCalledAgent(),
                "caller_id_correction": FailIfCalledAgent(),
                "dob": FailIfCalledAgent(),
            },
            mode="candidate_agents",
            fallback_to_legacy=False,
        )

        self.assertEqual(result["patient_names"][0]["raw"], "Morgan Example")
        self.assertEqual(result["patient_names"][0]["source"], "relationship_subject")
        self.assertTrue(any(item.get("raw") == "Casey" for item in name_agent.last_payload["name_candidates"]))

    def test_facility_patient_subject_is_available_to_name_agent_with_sentence_context(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "split_identity_dual_correction"
        os.environ["CANDIDATE_AGENT_EXECUTION"] = "parallel_http"
        os.environ["CANDIDATE_AGENT_AUTO_ACCEPT_DETERMINISTIC"] = "true"
        name_agent = SubjectSelectingNameAgent()

        result = extract_with_candidate_agents(
            {
                "transcript": (
                    "Morgan with Example Clinic and Example Clinic. Morgan Example, 424 of 60, had a knee revision today. "
                    "Again, this is Morgan from Example Clinic and Example Clinic."
                ),
                "caller_id": "EXAMPLE MORGAN",
            },
            agents={
                "numbers": FailIfCalledAgent(),
                "name": name_agent,
                "spelling_correction": FailIfCalledAgent(),
                "caller_id_correction": FailIfCalledAgent(),
                "dob": FailIfCalledAgent(),
            },
            mode="candidate_agents",
            fallback_to_legacy=False,
        )

        self.assertEqual(result["patient_names"][0]["raw"], "Morgan Example")
        self.assertEqual(result["patient_names"][0]["source"], "relationship_subject")
        payload_names = {item.get("raw"): item for item in name_agent.last_payload["name_candidates"]}
        self.assertIn("Morgan", payload_names)
        self.assertIn("Morgan Example", payload_names)
        self.assertIn("Morgan Example, 424 of 60, had", payload_names["Morgan Example"]["sentence_context"])

    def test_patient_of_provider_subject_is_available_to_name_agent(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "split_identity_dual_correction"
        os.environ["CANDIDATE_AGENT_EXECUTION"] = "parallel_http"
        os.environ["CANDIDATE_AGENT_AUTO_ACCEPT_DETERMINISTIC"] = "true"
        name_agent = SubjectSelectingNameAgent()

        result = extract_with_candidate_agents(
            {
                "transcript": (
                    "Hi, this is Avery calling from Example Home Care. "
                    "I'm a physical therapist for a patient of Dr. Sample, Robin Example."
                ),
                "caller_id": "EXAMPLE HOME CARE",
            },
            agents={
                "numbers": FailIfCalledAgent(),
                "name": name_agent,
                "spelling_correction": FailIfCalledAgent(),
                "caller_id_correction": FailIfCalledAgent(),
                "dob": FailIfCalledAgent(),
            },
            mode="candidate_agents",
            fallback_to_legacy=False,
        )

        self.assertEqual(result["patient_names"][0]["raw"], "Robin Example")
        payload_names = {item.get("raw"): item for item in name_agent.last_payload["name_candidates"]}
        self.assertIn("Avery", payload_names)
        self.assertIn("Robin Example", payload_names)
        self.assertIn("patient of Dr. Sample, Robin Example", payload_names["Robin Example"]["sentence_context"])

    def test_relationship_subject_is_available_to_name_agent(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "split_identity_dual_correction"
        os.environ["CANDIDATE_AGENT_EXECUTION"] = "parallel_http"
        os.environ["CANDIDATE_AGENT_AUTO_ACCEPT_DETERMINISTIC"] = "true"
        name_agent = SubjectSelectingNameAgent()

        result = extract_with_candidate_agents(
            {
                "transcript": (
                    "This is Casey Example. I am calling from my husband Morgan Example. "
                    "Morgan has a concern about his left knee."
                ),
                "caller_id": "Unavailable",
            },
            agents={
                "numbers": FailIfCalledAgent(),
                "name": name_agent,
                "spelling_correction": FailIfCalledAgent(),
                "caller_id_correction": FailIfCalledAgent(),
                "dob": FailIfCalledAgent(),
            },
            mode="candidate_agents",
            fallback_to_legacy=False,
        )

        self.assertEqual(result["patient_names"][0]["raw"], "Morgan Example")
        self.assertEqual(result["patient_names"][0]["source"], "relationship_subject")
        payload_names = {item.get("raw"): item for item in name_agent.last_payload["name_candidates"]}
        self.assertIn("Casey Example", payload_names)
        self.assertIn("Morgan Example", payload_names)
        self.assertIn("my husband Morgan Example", payload_names["Morgan Example"]["sentence_context"])
        health = candidate_agent_health(
            agents_loaded=["numbers", "name", "spelling_correction", "caller_id_correction", "dob"]
        )
        self.assertNotIn("name", health["last_agent_auto_accepted"])
        self.assertNotIn("spelling_correction", health["last_agent_auto_accepted"])

    def test_reverse_relationship_subject_is_available_to_name_agent(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "split_identity_dual_correction"
        os.environ["CANDIDATE_AGENT_EXECUTION"] = "parallel_http"
        os.environ["CANDIDATE_AGENT_AUTO_ACCEPT_DETERMINISTIC"] = "true"
        name_agent = SubjectSelectingNameAgent()

        result = extract_with_candidate_agents(
            {
                "transcript": (
                    "Yes, this is Bailey Example, Casey Sample's daughter. "
                    "He is needing a follow-up appointment from a hospital stay."
                ),
                "caller_id": "Unavailable",
            },
            agents={
                "numbers": FailIfCalledAgent(),
                "name": name_agent,
                "spelling_correction": FailIfCalledAgent(),
                "caller_id_correction": FailIfCalledAgent(),
                "dob": FailIfCalledAgent(),
            },
            mode="candidate_agents",
            fallback_to_legacy=False,
        )

        self.assertEqual(result["patient_names"][0]["raw"], "Casey Sample")
        payload_names = {item.get("raw"): item for item in name_agent.last_payload["name_candidates"]}
        self.assertIn("Bailey Example", payload_names)
        self.assertIn("Casey Sample", payload_names)
        self.assertIn("Casey Sample's daughter", payload_names["Casey Sample"]["sentence_context"])

    def test_facility_for_dob_subject_is_available_to_name_agent(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "split_identity_dual_correction"
        os.environ["CANDIDATE_AGENT_EXECUTION"] = "parallel_http"
        os.environ["CANDIDATE_AGENT_AUTO_ACCEPT_DETERMINISTIC"] = "true"
        name_agent = SubjectSelectingNameAgent()

        result = extract_with_candidate_agents(
            {
                "transcript": (
                    "Hi, this is Casey. I am calling from Example Clinic for Robin Example, "
                    "date of birth 10/11/1981. You can give me a call at Example Clinic. "
                    "The number is 618-555-0100."
                ),
                "caller_id": "Unavailable",
            },
            agents={
                "numbers": CompactSelectingNumbersAgent(),
                "name": name_agent,
                "spelling_correction": FailIfCalledAgent(),
                "caller_id_correction": FailIfCalledAgent(),
                "dob": FailIfCalledAgent(),
            },
            mode="candidate_agents",
            fallback_to_legacy=False,
        )

        self.assertEqual(result["patient_names"][0]["raw"], "Robin Example")
        self.assertEqual(result["callback_numbers"][0]["formatted"], "(618) 555-0100")
        payload_names = {item.get("raw"): item for item in name_agent.last_payload["name_candidates"]}
        self.assertIn("Casey", payload_names)
        self.assertIn("Robin Example", payload_names)
        self.assertIn("Robin Example, date of birth 10/11/1981", payload_names["Robin Example"]["sentence_context"])
        health = candidate_agent_health(
            agents_loaded=["numbers", "name", "spelling_correction", "caller_id_correction", "dob"]
        )
        self.assertIn("dob", health["last_agent_auto_accepted"])
        self.assertNotIn("name", health["last_agent_auto_accepted"])
        self.assertNotIn("spelling_correction", health["last_agent_auto_accepted"])

    def test_pronoun_appositive_subject_is_available_to_name_agent(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "split_identity_dual_correction"
        os.environ["CANDIDATE_AGENT_EXECUTION"] = "parallel_http"
        os.environ["CANDIDATE_AGENT_AUTO_ACCEPT_DETERMINISTIC"] = "true"
        name_agent = SubjectSelectingNameAgent()

        result = extract_with_candidate_agents(
            {
                "transcript": (
                    "Hello, this is Morgan from Example Clinic and Example Clinic. "
                    "I did get your message about him, Robin Example, being there at 8 a.m."
                ),
                "caller_id": "Unavailable",
            },
            agents={
                "numbers": FailIfCalledAgent(),
                "name": name_agent,
                "spelling_correction": FailIfCalledAgent(),
                "caller_id_correction": FailIfCalledAgent(),
                "dob": FailIfCalledAgent(),
            },
            mode="candidate_agents",
            fallback_to_legacy=False,
        )

        self.assertEqual(result["patient_names"][0]["raw"], "Robin Example")
        payload_names = {item.get("raw"): item for item in name_agent.last_payload["name_candidates"]}
        self.assertIn("Morgan", payload_names)
        self.assertIn("Robin Example", payload_names)
        self.assertIn("message about him, Robin Example", payload_names["Robin Example"]["sentence_context"])

    def test_spelling_confirmation_auto_accept_keeps_full_name(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "split_identity_dual_correction"
        os.environ["CANDIDATE_AGENT_EXECUTION"] = "parallel_http"
        os.environ["CANDIDATE_AGENT_AUTO_ACCEPT_DETERMINISTIC"] = "true"
        name_agent = RecordingAgent("name", {"name_ids": ["name:0"], "name_correction_ids": [], "errors": []})

        result = extract_with_candidate_agents(
            {
                "transcript": (
                    "Hi, this is Jordan Example. That's E-X-A-M-P-L-E. I need to speak with Dr. Sample's nurse about "
                    "rescheduling my knee replacement. My number is 202-555-0104. Thanks."
                ),
                "caller_id": "EXAMPLE JORDAN",
            },
            agents={
                "numbers": CompactSelectingNumbersAgent(),
                "name": name_agent,
                "spelling_correction": FailIfCalledAgent(),
                "caller_id_correction": FailIfCalledAgent(),
                "dob": FailIfCalledAgent(),
            },
            mode="candidate_agents",
            fallback_to_legacy=False,
        )

        self.assertEqual(result["patient_names"][0]["raw"], "Jordan Example")
        self.assertEqual(result["patient_names"][0]["value"], "Jordan Example")
        self.assertEqual(result["patient_names"][0]["source"], "transcript_spelling_corrected")
        health = candidate_agent_health(
            agents_loaded=["numbers", "name", "spelling_correction", "caller_id_correction", "dob"]
        )
        self.assertIn("spelling_correction", health["last_agent_auto_accepted"])

    def test_auto_accept_self_identification_is_disabled_by_default(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "split_identity_dual_correction"
        os.environ["CANDIDATE_AGENT_EXECUTION"] = "parallel_http"
        os.environ["CANDIDATE_AGENT_AUTO_ACCEPT_DETERMINISTIC"] = "true"
        name_agent = RecordingAgent("name", {"name_ids": ["name:0"], "name_correction_ids": [], "errors": []})

        result = extract_with_candidate_agents(
            {"transcript": "and my name is Casey Example.", "caller_id": "Wireless Caller"},
            agents={
                "numbers": FailIfCalledAgent(),
                "name": name_agent,
                "spelling_correction": FailIfCalledAgent(),
                "caller_id_correction": FailIfCalledAgent(),
                "dob": FailIfCalledAgent(),
            },
            mode="candidate_agents",
            fallback_to_legacy=False,
        )

        self.assertEqual(name_agent.calls, 1)
        self.assertEqual(result["patient_names"][0]["raw"], "Casey Example")
        health = candidate_agent_health(
            agents_loaded=["numbers", "name", "spelling_correction", "caller_id_correction", "dob"]
        )
        self.assertFalse(health["candidate_agent_auto_accept_self_identification"])
        self.assertEqual(health["last_agent_auto_accepted"], [])

    def test_auto_accept_self_identification_does_not_skip_name_when_caller_id_is_unrelated_name_shaped(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "split_identity_dual_correction"
        os.environ["CANDIDATE_AGENT_EXECUTION"] = "parallel_http"
        os.environ["CANDIDATE_AGENT_AUTO_ACCEPT_DETERMINISTIC"] = "true"
        os.environ["CANDIDATE_AGENT_AUTO_ACCEPT_SELF_IDENTIFICATION"] = "true"
        name_agent = RecordingAgent("name", {"name_ids": ["name:0"], "name_correction_ids": [], "errors": []})

        result = extract_with_candidate_agents(
            {"transcript": "and my name is Casey Example.", "caller_id": "ROBIN SAMPLE"},
            agents={
                "numbers": FailIfCalledAgent(),
                "name": name_agent,
                "spelling_correction": FailIfCalledAgent(),
                "caller_id_correction": RecordingAgent(
                    "caller_id_correction",
                    {"name_ids": [], "name_correction_ids": [], "errors": []},
                ),
                "dob": FailIfCalledAgent(),
            },
            mode="candidate_agents",
            fallback_to_legacy=False,
        )

        self.assertEqual(name_agent.calls, 1)
        self.assertEqual(result["patient_names"][0]["raw"], "Casey Example")
        health = candidate_agent_health(
            agents_loaded=["numbers", "name", "spelling_correction", "caller_id_correction", "dob"]
        )
        self.assertNotIn("name", health["last_agent_auto_accepted"])

    def test_spelling_auto_accept_disabled_by_env_runs_spelling_worker(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "split_identity_dual_correction"
        os.environ["CANDIDATE_AGENT_EXECUTION"] = "parallel_http"
        os.environ["CANDIDATE_AGENT_AUTO_ACCEPT_DETERMINISTIC"] = "true"
        os.environ["CANDIDATE_AGENT_AUTO_ACCEPT_TRANSCRIPT_SPELLING"] = "false"
        name_agent = RecordingAgent("name", {"name_ids": ["name:0"], "name_correction_ids": [], "errors": []})
        spelling_agent = RecordingAgent(
            "spelling_correction",
            {"name_ids": ["name:1"], "name_correction_ids": [], "errors": []},
        )

        result = extract_with_candidate_agents(
            {"transcript": "This is Bailey Sampel, S-A-M-P-L-E.", "caller_id": "Wireless Caller"},
            agents={
                "numbers": FailIfCalledAgent(),
                "name": name_agent,
                "spelling_correction": spelling_agent,
                "caller_id_correction": FailIfCalledAgent(),
                "dob": FailIfCalledAgent(),
            },
            mode="candidate_agents",
            fallback_to_legacy=False,
        )

        self.assertEqual(name_agent.calls, 1)
        self.assertEqual(spelling_agent.calls, 1)
        self.assertEqual(result["patient_names"][0]["value"], "Bailey Sample")
        health = candidate_agent_health(
            agents_loaded=["numbers", "name", "spelling_correction", "caller_id_correction", "dob"]
        )
        self.assertNotIn("spelling_correction", health["last_agent_auto_accepted"])
        self.assertFalse(health["candidate_agent_auto_accept_transcript_spelling"])

    def test_auto_accept_multiple_spelling_candidates_still_runs_spelling_worker(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "split_identity_dual_correction"
        os.environ["CANDIDATE_AGENT_EXECUTION"] = "parallel_http"
        os.environ["CANDIDATE_AGENT_AUTO_ACCEPT_DETERMINISTIC"] = "true"
        spelling_agent = CompactSelectingNameCorrectionAgent()

        result = extract_with_candidate_agents(
            {"transcript": "This is Bailey Sampel, S-A-M-P-L-E. My name is Example, K-O-E-S-P-E-R."},
            agents={
                "numbers": FailIfCalledAgent(),
                "name": RecordingAgent("name", {"name_ids": ["name:0"], "name_correction_ids": [], "errors": []}),
                "spelling_correction": spelling_agent,
                "caller_id_correction": FailIfCalledAgent(),
                "dob": FailIfCalledAgent(),
            },
            mode="candidate_agents",
            fallback_to_legacy=False,
        )

        self.assertTrue(hasattr(spelling_agent, "last_payload"))
        self.assertEqual(result["patient_names"][0]["source"], "transcript_spelling_corrected")
        health = candidate_agent_health(
            agents_loaded=["numbers", "name", "spelling_correction", "caller_id_correction", "dob"]
        )
        self.assertEqual(health["last_agent_auto_accepted"], [])

    def test_split_identity_skips_numbers_and_dob_when_only_name_candidates_exist(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "split_identity"
        os.environ["CANDIDATE_AGENT_EXECUTION"] = "parallel_http"
        name_agent = CompactSelectingNameAgent()

        result = extract_with_candidate_agents(
            {"transcript": "This is Jordan Example calling."},
            agents={
                "numbers": FailIfCalledAgent(),
                "name": name_agent,
                "dob": FailIfCalledAgent(),
            },
            mode="candidate_agents",
            fallback_to_legacy=False,
        )

        self.assertEqual(result["patient_names"][0]["value"], "Jordan Example")
        self.assertEqual(result["callback_numbers"], [])
        self.assertEqual(result["dob_candidates"], [])
        health = candidate_agent_health(agents_loaded=["numbers", "name", "dob"])
        self.assertEqual(set(health["last_agent_skipped"]), {"numbers", "dob"})
        self.assertEqual(health["last_agent_timings_ms"]["numbers"], 0)
        self.assertEqual(health["last_agent_timings_ms"]["dob"], 0)

    def test_split_identity_skips_name_and_dob_when_only_number_candidates_exist(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "split_identity"
        numbers_agent = CompactSelectingNumbersAgent()

        result = extract_with_candidate_agents(
            {"transcript": "Please call me back at 202-555-0108."},
            agents={
                "numbers": numbers_agent,
                "name": FailIfCalledAgent(),
                "dob": FailIfCalledAgent(),
            },
            mode="candidate_agents",
            fallback_to_legacy=False,
        )

        self.assertEqual(result["callback_numbers"][0]["normalized"], "2025550108")
        self.assertEqual(result["patient_names"], [])
        self.assertEqual(result["dob_candidates"], [])
        health = candidate_agent_health(agents_loaded=["numbers", "name", "dob"])
        self.assertEqual(set(health["last_agent_skipped"]), {"name", "dob"})

    def test_split_identity_skips_all_agents_for_semantic_only_callback_request(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "split_identity"

        result = extract_with_candidate_agents(
            {"transcript": "Please call me back at the number on file."},
            agents={
                "numbers": FailIfCalledAgent(),
                "name": FailIfCalledAgent(),
                "dob": FailIfCalledAgent(),
            },
            mode="candidate_agents",
            fallback_to_legacy=False,
        )

        self.assertEqual(list(result.keys()), FINAL_SCHEMA_KEYS)
        self.assertEqual(result["callback_numbers"], [])
        self.assertEqual(result["patient_names"], [])
        self.assertEqual(result["dob_candidates"], [])
        health = candidate_agent_health(agents_loaded=["numbers", "name", "dob"])
        self.assertEqual(set(health["last_agent_skipped"]), {"numbers", "name", "dob"})

    def test_split_identity_parallel_execution_submits_only_relevant_agents(self):
        os.environ["CANDIDATE_AGENT_TOPOLOGY"] = "split_identity"
        os.environ["CANDIDATE_AGENT_EXECUTION"] = "parallel_http"
        numbers_agent = RecordingAgent(
            "numbers",
            {"callback_ids": ["number:0"], "fax_ids": [], "uncertain_ids": [], "errors": []},
        )
        dob_agent = RecordingAgent("dob", {"dob_ids": ["dob:0"], "errors": []})

        result = extract_with_candidate_agents(
            {"transcript": "Date of birth is 1/4/72. Please call me back at 202-555-0108."},
            agents={
                "numbers": numbers_agent,
                "name": FailIfCalledAgent(),
                "dob": dob_agent,
            },
            mode="candidate_agents",
        )

        self.assertEqual(numbers_agent.calls, 1)
        self.assertEqual(dob_agent.calls, 1)
        self.assertEqual(result["callback_numbers"][0]["normalized"], "2025550108")
        self.assertEqual(result["dob_candidates"][0]["normalized"], "01/04/1972")
        health = candidate_agent_health(agents_loaded=["numbers", "name", "dob"])
        self.assertEqual(health["last_agent_skipped"], ["name"])

    def test_identity_topology_skips_numbers_when_no_number_candidates_exist(self):
        os.environ.pop("CANDIDATE_AGENT_TOPOLOGY", None)

        result = extract_with_candidate_agents(
            {"transcript": "This is Jordan Example calling."},
            agents={
                "numbers": FailIfCalledAgent(),
                "identity": CompactSelectingIdentityAgent(),
            },
            mode="candidate_agents",
        )

        self.assertEqual(result["patient_names"][0]["value"], "Jordan Example")
        self.assertEqual(result["callback_numbers"], [])
        health = candidate_agent_health(agents_loaded=["numbers", "identity"])
        self.assertEqual(health["last_agent_skipped"], ["numbers"])

    def test_shadow_mode_returns_legacy_output(self):
        legacy = {key: [] for key in FINAL_SCHEMA_KEYS}
        legacy["possible_errors"] = ["legacy baseline"]

        result = extract_with_candidate_agents(
            {"transcript": "Please call me back at 202-555-0108."},
            legacy_extractor=lambda _request: legacy,
            agents={"numbers": SelectingNumbersAgent(), "identity": EmptyIdentityAgent()},
            mode="shadow_candidate_agents",
        )

        self.assertIs(result, legacy)

    def test_active_mode_falls_back_to_legacy_when_enabled(self):
        legacy = {key: [] for key in FINAL_SCHEMA_KEYS}
        legacy["possible_errors"] = ["legacy fallback"]

        result = extract_with_candidate_agents(
            {"transcript": "Please call me back at 202-555-0108."},
            legacy_extractor=lambda _request: legacy,
            agents={"numbers": FailingAgent(), "identity": EmptyIdentityAgent()},
            mode="candidate_agents",
            fallback_to_legacy=True,
        )

        self.assertIs(result, legacy)


if __name__ == "__main__":
    unittest.main()

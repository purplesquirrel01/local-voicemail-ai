import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import watcher


class HostAwareEndpointRoutingTests(unittest.TestCase):
    def settings(self):
        return SimpleNamespace(
            whisper_ready_timeout_seconds=0.1,
            gemma_ready_timeout_seconds=0.1,
            parakeet_ready_timeout_seconds=0.1,
        )

    def test_gemma_prefers_host_without_local_whisper_in_flight(self):
        settings = self.settings()
        whisper_39 = "http://192.0.2.39:8765/transcribe/voicemail"
        whisper_47 = "http://192.0.2.47:8765/transcribe/voicemail"
        gemma_39 = "http://192.0.2.39:8787"
        gemma_47 = "http://192.0.2.47:8787"
        host_load = watcher.HostLoadTracker(enabled=True)
        whisper_pool = watcher.WhisperEndpointPool(
            (whisper_39, whisper_47),
            host_load_tracker=host_load,
        )
        gemma_pool = watcher.ServiceEndpointPool(
            "Gemma",
            (gemma_39, gemma_47),
            "gemma_ready_timeout_seconds",
            host_load_tracker=host_load,
            kind="gemma",
            cross_kinds=("whisper",),
        )
        gemma_pool._remote_load = lambda _url, _settings, _request_id: 0

        whisper_pool.mark_started(whisper_39)
        self.assertEqual(
            gemma_pool.least_busy_order(settings, "msg001"),
            (gemma_47, gemma_39),
        )

        whisper_pool.mark_finished(whisper_39)
        self.assertEqual(
            gemma_pool.least_busy_order(settings, "msg001"),
            (gemma_39, gemma_47),
        )

    def test_whisper_prefers_host_without_local_gemma_in_flight(self):
        settings = self.settings()
        whisper_39 = "http://192.0.2.39:8765/transcribe/voicemail"
        whisper_47 = "http://192.0.2.47:8765/transcribe/voicemail"
        gemma_39 = "http://192.0.2.39:8787"
        gemma_47 = "http://192.0.2.47:8787"
        host_load = watcher.HostLoadTracker(enabled=True)
        whisper_pool = watcher.WhisperEndpointPool(
            (whisper_39, whisper_47),
            host_load_tracker=host_load,
        )
        whisper_pool._voicemail_queue_depth = lambda _url, _settings, _request_id: 0
        gemma_pool = watcher.ServiceEndpointPool(
            "Gemma",
            (gemma_39, gemma_47),
            "gemma_ready_timeout_seconds",
            host_load_tracker=host_load,
            kind="gemma",
            cross_kinds=("whisper",),
        )

        gemma_pool.mark_started(gemma_39)
        self.assertEqual(
            whisper_pool.least_busy_order(settings, "msg002"),
            (whisper_47, whisper_39),
        )

        gemma_pool.mark_finished(gemma_39)
        self.assertEqual(
            whisper_pool.least_busy_order(settings, "msg002"),
            (whisper_39, whisper_47),
        )

    def test_host_matching_ignores_port_and_path(self):
        host_load = watcher.HostLoadTracker(enabled=True)
        host_load.mark_started("http://192.0.2.39:8765/transcribe/voicemail", "whisper")

        self.assertEqual(host_load.in_flight("http://192.0.2.39:8787", "whisper"), 1)
        self.assertEqual(host_load.in_flight("http://192.0.2.47:8787", "whisper"), 0)

    def test_disabled_host_tracker_preserves_independent_pool_ordering(self):
        settings = self.settings()
        whisper_39 = "http://192.0.2.39:8765/transcribe/voicemail"
        whisper_47 = "http://192.0.2.47:8765/transcribe/voicemail"
        gemma_39 = "http://192.0.2.39:8787"
        gemma_47 = "http://192.0.2.47:8787"
        host_load = watcher.HostLoadTracker(enabled=False)
        whisper_pool = watcher.WhisperEndpointPool(
            (whisper_39, whisper_47),
            host_load_tracker=host_load,
        )
        gemma_pool = watcher.ServiceEndpointPool(
            "Gemma",
            (gemma_39, gemma_47),
            "gemma_ready_timeout_seconds",
            host_load_tracker=host_load,
            kind="gemma",
            cross_kinds=("whisper",),
        )
        gemma_pool._remote_load = lambda _url, _settings, _request_id: 0

        whisper_pool.mark_started(whisper_39)
        self.assertEqual(
            gemma_pool.least_busy_order(settings, "msg003"),
            (gemma_39, gemma_47),
        )

    def test_parakeet_pool_without_host_tracker_ignores_whisper_and_gemma_load(self):
        settings = self.settings()
        parakeet_39 = "http://192.0.2.39:8766/transcribe"
        parakeet_47 = "http://192.0.2.47:8766/transcribe"
        parakeet_pool = watcher.ServiceEndpointPool(
            "Parakeet",
            (parakeet_39, parakeet_47),
            "parakeet_ready_timeout_seconds",
        )
        parakeet_pool._remote_load = lambda _url, _settings, _request_id: 0

        self.assertEqual(
            parakeet_pool.least_busy_order(settings, "msg004"),
            (parakeet_39, parakeet_47),
        )

    def test_settings_host_aware_routing_defaults_on_and_can_be_disabled(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(watcher.Settings.from_env().host_aware_routing)

        with patch.dict(os.environ, {"WATCHER_HOST_AWARE_ROUTING": "false"}, clear=True):
            self.assertFalse(watcher.Settings.from_env().host_aware_routing)


if __name__ == "__main__":
    unittest.main()

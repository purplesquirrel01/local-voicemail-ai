"""Install a built wheel in a new environment and verify its application surface."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

PROBE = r"""
import importlib
import importlib.metadata as metadata
import importlib.resources as resources
from pathlib import Path
import sys
from unittest.mock import patch

root = Path(sys.prefix).resolve()
names = ['watcher', 'voicemail_portal', 'verification', 'candidate_extractor',
         'extraction_orchestrator', 'final_resolver', 'lvt_entrypoints',
         'voicemail_common', 'voicemail_watcher', 'voicemail_portal_app',
         'voicemail_verification', 'agent_constraints', 'prompts', 'lvt_assets']
for name in names:
    module = importlib.import_module(name)
    assert Path(module.__file__).resolve().is_relative_to(root), name
entries = {e.name: e for e in metadata.distribution('local-voicemail-transcription').entry_points
           if e.group == 'console_scripts'}
assert set(entries) == {'lvt-watcher', 'lvt-portal', 'lvt-whisper-api', 'lvt-parakeet-api', 'lvt-gemma-api'}
assert all(callable(e.load()) for e in entries.values())
assert 'callback' in resources.files('prompts').joinpath('numbers_agent.md').read_text()
assert '<svg' in resources.files('lvt_assets').joinpath('voicemail-portal-icon.svg').read_text()
for name in ('lvt_installer', 'lvt_combined_installer', 'lvt_bootstrap'):
    assert importlib.util.find_spec(name) is None, name
import lvt_entrypoints as entry
with patch('uvicorn.run') as server, patch.object(sys, 'argv', ['lvt-portal']):
    entry.portal_main()
    entry.whisper_main()
    entry.parakeet_main()
    entry.gemma_main()
    assert server.call_count == 4
    assert all(c.kwargs['host'] == '127.0.0.1' for c in server.call_args_list)
print('14 installed module/package imports, 5 console entries, packaged prompts/icon, and 4 mocked service dispatches passed')
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('wheel', type=Path)
    parser.add_argument('--venv', type=Path, required=True)
    args = parser.parse_args()
    wheel, environment = args.wheel.resolve(), args.venv.resolve()
    if not wheel.is_file() or wheel.suffix != '.whl':
        parser.error('wheel must name one built .whl file')
    if environment.exists():
        parser.error('verification environment must not already exist')
    subprocess.run([sys.executable, '-m', 'venv', str(environment)], check=True)
    python = environment / 'bin/python'
    subprocess.run([str(python), '-m', 'pip', 'install', f'{wheel}[portal]'], check=True)
    subprocess.run([str(python), '-m', 'pip', 'check'], check=True)
    subprocess.run([str(python), '-I', '-c', PROBE], cwd=environment, check=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

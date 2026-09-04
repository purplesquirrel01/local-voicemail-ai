from pathlib import Path
from tools.audit_secrets import scan_for_secrets

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def test_source_secret_audit_rejects_private_keys_and_committed_credentials(tmp_path: Path) -> None:
    (tmp_path / "unsafe.env").write_text("LVT_API_KEY=" + "a" * 48 + "\n", encoding="utf-8")
    private_marker = "-----BEGIN " + "PRIVATE KEY-----"
    (tmp_path / "private.pem").write_text(f"{private_marker}\nnot-real-key-material\n", encoding="utf-8")

    findings = scan_for_secrets(tmp_path)

    assert {(item.path.name, item.kind) for item in findings} == {
        ("unsafe.env", "credential-assignment"),
        ("private.pem", "private-key"),
    }

def test_source_secret_audit_ignores_generated_test_and_release_workspaces(tmp_path: Path) -> None:
    for relative in (
        ".pytest-orchestrator/synthetic.env",
        ".test-tmp/synthetic.env",
        ".build-tmp/synthetic.env",
        "local-test-release-synthetic/synthetic.env",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("LVT_API_KEY=" + "a" * 48 + "\n", encoding="utf-8")

    assert scan_for_secrets(tmp_path) == []

def test_source_secret_audit_accepts_the_public_source_tree() -> None:
    assert scan_for_secrets(PROJECT_ROOT) == []

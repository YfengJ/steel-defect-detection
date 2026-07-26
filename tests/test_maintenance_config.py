from __future__ import annotations

import ast
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_ci_runs_declared_checks_with_read_only_permissions() -> None:
    workflow = yaml.safe_load(
        (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    )
    job = workflow["jobs"]["syntax-check"]
    commands = "\n".join(
        str(step.get("run", "")) for step in job["steps"] if isinstance(step, dict)
    )

    assert workflow["permissions"] == {"contents": "read"}
    assert job["timeout-minutes"] <= 20
    assert "requirements-dev.txt" in commands
    assert "pytest tests" in commands
    assert "compileall" in commands
    actions = [
        str(step.get("uses", "")) for step in job["steps"] if isinstance(step, dict)
    ]
    assert "actions/checkout@v6" in actions
    assert "actions/setup-python@v6" in actions


def test_security_policy_warns_about_untrusted_checkpoints() -> None:
    policy = (REPO_ROOT / "SECURITY.md").read_text(encoding="utf-8").lower()

    assert "trusted" in policy
    assert "checkpoint" in policy or "model weight" in policy


def test_vendored_runtime_keeps_upstream_distribution_identity() -> None:
    setup = (REPO_ROOT / "setup.py").read_text(encoding="utf-8")
    tree = ast.parse(setup)
    setup_call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "setup"
    )
    metadata = {
        keyword.arg: ast.literal_eval(keyword.value)
        for keyword in setup_call.keywords
        if keyword.arg in {"name", "python_requires", "url"}
    }

    assert metadata["name"] == "ultralytics"
    assert metadata["python_requires"] == ">=3.10,<3.13"
    assert metadata["url"] == "https://github.com/ultralytics/ultralytics"
    assert "packages the vendored Ultralytics runtime" in setup

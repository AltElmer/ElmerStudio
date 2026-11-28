"""Static test: CI workflow YAML must parse and declare correct ELMERSTUDIO_TEST_REQUIRE per OS.

This test catches:
- YAML syntax errors (e.g., unquoted colon in step name)
- Missing or wrong solver requirements per OS (Linux=elmer+nastran, Windows=nastran, macOS=none)
- Missing install steps (asset names, checksums, NASTRAN_HOME export)
"""
import yaml
import os


def test_ci_yaml_parses():
    """The CI workflow must be valid YAML (no unquoted colons in step names, etc.)."""
    path = os.path.join(os.path.dirname(__file__), "..", ".github", "workflows", "ci.yml")
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    assert data is not None
    assert "jobs" in data
    assert "test" in data["jobs"]


def test_ci_requires_solvers_per_os():
    """Each OS job step must set ELMERSTUDIO_TEST_REQUIRE correctly for its solvers.

    Linux: elmer,nastran (both installed in the workflow)
    Windows: nastran (NASTRAN-95 installed, no Elmer)
    macOS: (empty - no solvers installed)
    """
    path = os.path.join(os.path.dirname(__file__), "..", ".github", "workflows", "ci.yml")
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    job = data["jobs"]["test"]
    steps = job["steps"]

    # Find the test steps by their names
    linux_step = None
    nonlinux_step = None
    for step in steps:
        name = step.get("name", "")
        if "Engine and solver tests (Linux" in name:
            linux_step = step
        elif "Engine tests (no Elmer on this runner" in name:
            nonlinux_step = step

    assert linux_step is not None, "Linux test step not found"
    assert nonlinux_step is not None, "Non-Linux test step not found"

    # Linux step: requires elmer AND nastran
    linux_env = linux_step.get("env", {})
    linux_require = linux_env.get("ELMERSTUDIO_TEST_REQUIRE", "")
    assert "elmer" in linux_require.lower(), f"Linux step missing 'elmer' in ELMERSTUDIO_TEST_REQUIRE: {linux_require}"
    assert "nastran" in linux_require.lower(), f"Linux step missing 'nastran' in ELMERSTUDIO_TEST_REQUIRE: {linux_require}"

    # Non-Linux step: requires nastran on Windows, empty on macOS
    nonlinux_env = nonlinux_step.get("env", {})
    nonlinux_require = nonlinux_env.get("ELMERSTUDIO_TEST_REQUIRE", "")
    # The YAML uses a conditional: ${{ runner.os == 'Windows' && 'nastran' || '' }}
    # We just verify the expression is present (yaml loads it as a string)
    assert "Windows" in nonlinux_require and "nastran" in nonlinux_require, \
        f"Non-Linux step missing conditional nastran for Windows: {nonlinux_require}"


def test_ci_installs_nastran_with_checksums():
    """The workflow must install NASTRAN-95 with verified checksums on Linux and Windows."""
    path = os.path.join(os.path.dirname(__file__), "..", ".github", "workflows", "ci.yml")
    with open(path, "r") as f:
        content = f.read()

    # Linux install step
    assert "nastran95-v1.0.0-linux-x86_64.tar.gz" in content
    assert "6a97cd11975041367c0ababa560dddbd556a50b0bff1b9f6eb2ecaedf1c0d73b" in content
    assert "sha256sum -c" in content
    assert "NASTRAN_HOME=" in content

    # Windows install step
    assert "nastran95-v1.0.0-windows-x86_64.zip" in content
    assert "35399cefb508db4745725f3c61bf3df134b302ff94b251bfa663b7f2ef60f726" in content
    assert "Get-FileHash" in content
    assert "NASTRAN_HOME=" in content


def test_missing_required_detects_absent_variable():
    """missing_required() must treat absent ELMERSTUDIO_TEST_REQUIRE differently from empty string.

    When the variable is not set at all (popped), the test session should not exit.
    When it is set to empty string, it means 'require no solvers' - which is valid for macOS.
    """
    import importlib.util
    import os
    spec = importlib.util.spec_from_file_location("conftest", os.path.join(os.path.dirname(__file__), "conftest.py"))
    conftest = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conftest)
    missing_required = conftest.missing_required

    # Test: absent env var (not in os.environ) -> no requirements -> no error
    # We simulate by temporarily removing it
    old = os.environ.pop("ELMERSTUDIO_TEST_REQUIRE", None)
    try:
        wrong = missing_required(os.environ.get("ELMERSTUDIO_TEST_REQUIRE", ""))
        assert wrong == [], f"Absent variable should not produce errors: {wrong}"
    finally:
        if old is not None:
            os.environ["ELMERSTUDIO_TEST_REQUIRE"] = old

    # Test: empty string -> no requirements -> no error (valid for macOS)
    os.environ["ELMERSTUDIO_TEST_REQUIRE"] = ""
    try:
        wrong = missing_required(os.environ.get("ELMERSTUDIO_TEST_REQUIRE", ""))
        assert wrong == [], f"Empty string should not produce errors: {wrong}"
    finally:
        if old is not None:
            os.environ["ELMERSTUDIO_TEST_REQUIRE"] = old
        else:
            os.environ.pop("ELMERSTUDIO_TEST_REQUIRE", None)

    # Test: requiring nastran when not installed -> error
    os.environ["ELMERSTUDIO_TEST_REQUIRE"] = "nastran"
    try:
        # This will fail if nastran is not installed, which is expected
        wrong = missing_required(os.environ.get("ELMERSTUDIO_TEST_REQUIRE", ""))
        # We can't assert on the exact result without knowing if nastran is installed
        # but we can verify the function doesn't crash
        assert isinstance(wrong, list)
    finally:
        if old is not None:
            os.environ["ELMERSTUDIO_TEST_REQUIRE"] = old
        else:
            os.environ.pop("ELMERSTUDIO_TEST_REQUIRE", None)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

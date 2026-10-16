"""Pipeline CLIs must not crash on non-ASCII output when stdout is cp1252 (Windows redirects)."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("script", ["run_phase2", "run_phase3", "run_snapshot", "ingest"])
def test_script_survives_cp1252_stdout(script):
    code = (
        f"import sys; sys.path.insert(0, r'{ROOT / 'scripts'}'); import {script}; "
        "print('\u2705 done')"
    )
    env = {**os.environ, "PYTHONIOENCODING": "cp1252"}
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, env=env, cwd=ROOT)
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    assert "done" in proc.stdout.decode("utf-8", "replace")

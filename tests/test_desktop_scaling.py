"""The desktop at the display scalings and font sizes people use: the packaged app's own self-test
(``--selftest-ui``: every ribbon command reachable at 1600/1024/760 px in both layouts, no clipped text in the
ribbon or status bar, every control named, the recovery checkpoint) run through the real entry point at 100, 125,
150 and 200 % (QT_SCALE_FACTOR, which scales like a display setting) and with a larger base font.

The checks themselves are proven to catch a defect in test_desktop.py (deliberately clipped label, a command
removed from a collapsed group's menu)."""
import os
import subprocess
import sys
import tempfile

import pytest

pytest.importorskip("PySide6")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _selftest(scale: str, font: str | None = None):
    out = os.path.join(tempfile.mkdtemp(prefix="es_scale_"), "selftest.txt")
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", QT_SCALE_FACTOR=scale,
               ELMERSTUDIO_RECOVERY_DIR=tempfile.mkdtemp(prefix="es_recovery_"))
    if font:
        env["ELMERSTUDIO_FONT_PT"] = font
    r = subprocess.run([sys.executable, "-m", "elmerstudio", "--example", "heat_sink", "--quit-after", "1",
                        "--selftest-ui", "--selftest-out", out], cwd=ROOT, env=env, capture_output=True, text=True,
                       timeout=300)
    txt = open(out).read() if os.path.exists(out) else ""
    return txt, r.stdout + r.stderr


@pytest.mark.parametrize("scale,font", [("1", None), ("1.25", None), ("1.5", None), ("2", None), ("1", "11")])
def test_desktop_at_display_scaling(scale, font):
    txt, log = _selftest(scale, font)
    problems = [ln for ln in log.splitlines() if ln.startswith("UI:")]
    assert txt.startswith("OK"), log[-3000:]
    assert "ui=ok" in txt, "\n".join(problems) or log[-3000:]

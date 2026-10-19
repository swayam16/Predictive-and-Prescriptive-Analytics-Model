"""The Streamlit demo renders against the committed artifacts without errors."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytest.importorskip("streamlit")


@pytest.mark.skipif(not (ROOT / "artifacts" / "live_scores.csv.gz").exists(), reason="run `churnpolicy run` first")
def test_app_renders():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=120).run()
    assert not at.exception
    assert at.metric[0].value == "500"
    at.sidebar.slider[0].set_value(200).run()
    assert not at.exception
    assert at.metric[0].value == "200"

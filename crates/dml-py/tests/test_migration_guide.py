"""The code in docs/user-guide/migrating-from-doubleml.md runs, in order."""

import re
import textwrap
import warnings
from pathlib import Path

import pytest

GUIDE = Path(__file__).resolve().parents[3] / "docs" / "user-guide" / "migrating-from-doubleml.md"


@pytest.mark.skipif(not GUIDE.exists(), reason="the documentation is not available")
def test_guide_runs():
    blocks = re.findall(r"^( *)```python\n(.*?)^\1```", GUIDE.read_text(encoding="utf-8"), flags=re.M | re.S)
    assert len(blocks) >= 8
    namespace = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _, code in blocks:
            exec(textwrap.dedent(code), namespace)

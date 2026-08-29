"""Regression pin for the generated graphs.

If this fails, the graph changed. That is not automatically wrong — but it must
be a decision someone made, not a side effect. Regenerate with
``python scripts/update_goldens.py`` and review the diff.
"""

from __future__ import annotations

import json

import pytest

from tests.workflows.cases import GOLDEN_DIR, cases


@pytest.mark.parametrize("name,graph", list(cases()), ids=lambda value: value)
def test_graph_matches_golden(name: str, graph: dict) -> None:
    path = GOLDEN_DIR / f"{name}.json"
    assert path.is_file(), (
        f"no golden file for {name}; run python scripts/update_goldens.py"
    )
    expected = json.loads(path.read_text())
    assert graph == expected, (
        f"the {name} graph changed. If that was intended, run "
        "python scripts/update_goldens.py and review the diff."
    )

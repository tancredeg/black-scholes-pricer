"""The demo notebook runs end to end, offline, in under two minutes."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "demo.ipynb"


@pytest.mark.slow
def test_demo_notebook_executes_quickly_without_errors():
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    start = time.perf_counter()
    nbclient.NotebookClient(
        notebook,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK.parent)}},
    ).execute()
    assert time.perf_counter() - start < 120
    errors = [
        o
        for c in notebook.cells
        if c.cell_type == "code"
        for o in c.outputs
        if o.output_type == "error"
    ]
    assert not errors

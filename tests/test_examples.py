"""Tests for the example notebooks.

The notebooks are the library's tutorials and are not executed in CI — several train
models or fetch data. What is pinned here is cheaper and still load-bearing: every
code cell must at least be valid Python, which catches API drift that leaves a
notebook referencing removed names or broken syntax.
"""

import json
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"
NOTEBOOK_PATHS = sorted(EXAMPLES_DIR.glob("*.ipynb"))


@pytest.mark.parametrize("path", NOTEBOOK_PATHS, ids=lambda path: path.stem)
def test_notebook_code_cells_compile(path: Path) -> None:
	"""Test that every code cell of the notebook is valid Python."""
	notebook = json.loads(path.read_text())
	sources = []
	for cell in notebook["cells"]:
		if cell["cell_type"] != "code":
			continue
		# IPython magics and shell escapes are not Python; strip those lines.
		sources.append(
			"".join(line for line in cell["source"] if not line.lstrip().startswith(("%", "!")))
		)
	compile("\n\n".join(sources), str(path), "exec")


def test_notebooks_exist() -> None:
	"""Test that the examples directory is where this test thinks it is."""
	assert len(NOTEBOOK_PATHS) >= 20

docs: lint
    uv run pdoc3 --force --html --output-dir docs/ --config latex_math=True frx download mup_sweep

lint: fmt
    ruff check frx/ download.py mup_sweep.py

fmt:
    isort .
    ruff format --preview .

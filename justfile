docs: lint
    uv run pdoc3 --force --html --output-dir docs/ --config latex_math=True frx download sweep

lint: fmt
    ruff check frx/ download.py sweep.py

fmt:
    isort .
    ruff format --preview .

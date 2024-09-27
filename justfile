docs: lint
    uv run pdoc3 --force --html --output-dir docs/ --config latex_math=True frx train download

lint: fmt
    ruff check frx/ train.py

fmt:
    isort .
    ruff format --preview .

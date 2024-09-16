docs: lint
    uv run pdoc3 --force --output-dir docs/md --config latex_math=True src.frx train download
    uv run pdoc3 --http :8081 --force --html --output-dir docs/html --config latex_math=True src.frx train download

lint: fmt
    ruff check .

fmt:
    isort .
    ruff format --preview .

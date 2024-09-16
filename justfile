lint: fmt
    ruff check .

fmt:
    isort .
    ruff format --preview .

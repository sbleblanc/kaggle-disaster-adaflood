update-dependencies:
	uv pip compile pyproject.toml --output-file requirements.txt
	uv pip sync requirements.txt
	uv pip install -e .
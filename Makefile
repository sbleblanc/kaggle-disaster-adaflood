update-dependencies:
	uv pip compile pyproject.toml --universal --output-file requirements.txt
	uv pip sync requirements.txt
	uv pip install -e .
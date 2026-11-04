clean:
	find . \( -name node_modules -o -name .git \) -prune -o \
		-type d \( -name __pycache__ -o -name dist -o -name .mypy_cache -o -name .pytest_cache -o -name .ruff_cache \) \
		-exec rm -rf {} + 2>/dev/null
	find . \( -name node_modules -o -name .git \) -prune -o \
		-type f \( -name '*.pyc' -o -name '*.log' -o -name '*.coverage' \) \
		-exec rm -f {} + 2>/dev/null
	rm -rf logs
	clear

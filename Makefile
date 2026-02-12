MODULE ?= test/d9d_test

.PHONY: test test-local test-distributed

test-local:
	@echo "Running local tests for $(MODULE)..."
	@python -m pytest $(MODULE) -m local

test-distributed:
	@echo "Running distributed tests for $(MODULE)..."
	@torchrun --nnodes 1 --nproc-per-node 8 --local-ranks-filter=0 --log_dir logs/dist_test --tee 3 -m pytest --instafail $(MODULE) -m distributed
	@echo "Please see the logs/dist_test for logs across all ranks if distributed tests did not pass"

test: test-local test-distributed

lint:
	@echo "Formatting"
	@ruff format
	@echo "Auto-Fixing Imports"
	@ruff check --fix
	@echo "Running linting"
	@ruff check

mypy:
	@echo "Running mypy"
	@mypy

mkdocs:
	@echo "Starting docs server"
	@poetry run mkdocs serve -a 0.0.0.0:8081 -w docs

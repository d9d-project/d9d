MODULE ?= test/d9d_test

.PHONY: test test-local test-distributed

test-local:
	@echo "Running local tests for $(MODULE)..."
	@uv run python -m pytest $(MODULE) -m local

test-distributed:
	@echo "Running distributed tests for $(MODULE)..."
	@uv run torchrun --nnodes 1 --nproc-per-node 8 --local-ranks-filter=0 --log_dir logs/dist_test --tee 3 -m pytest --instafail $(MODULE) -m distributed
	@echo "Please see the logs/dist_test for logs across all ranks if distributed tests did not pass"

test: test-local test-distributed

lint:
	@echo "Formatting"
	@uv run ruff format
	@echo "Auto-Fixing Imports"
	@uv run ruff check --fix
	@echo "Running linting"
	@uv run ruff check
	@echo "Running type checking"
	@uv run ty check

mkdocs:
	@echo "Starting docs server"
	@uv run zensical serve -a 0.0.0.0:8081

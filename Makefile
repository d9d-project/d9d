MODULE ?= test/d9d_test

.PHONY: test
test:
	@echo "Running local tests for $(MODULE)..."
	@python -m pytest $(MODULE) -m local
	@echo "Running distributed tests for $(MODULE)..."
	@torchrun --nnodes 1 --nproc-per-node 8 -m pytest -x $(MODULE) -m distributed

lint:
	@echo "Auto-Fixing Imports"
	@ruff check --fix
	@echo "Running linting"
	@ruff check
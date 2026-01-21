MODULE ?= test/d9d_test

.PHONY: test
test:
	@echo "Running local tests for $(MODULE)..."
	@python -m pytest $(MODULE) -m local
	@echo "Running distributed tests for $(MODULE)..."
	@torchrun --nnodes 1 --nproc-per-node 8 --local-ranks-filter=0 --log_dir logs/dist_test --tee 3 -m pytest -x $(MODULE) -m distributed
	@echo "Please see the logs/dist_test for logs across all ranks if distributed tests did not pass"

lint:
	@echo "Auto-Fixing Imports"
	@ruff check --fix
	@echo "Running linting"
	@ruff check
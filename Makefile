.PHONY: test

test:
	@echo "Running local tests..."
	@python -m pytest test/d9d_test -m local
	@echo "Running distributed tests..."
	@torchrun --nnodes 1 --nproc-per-node 8 -m pytest -x test/d9d_test -m distributed

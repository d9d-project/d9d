MODULE ?= packages/d9d-dev/src/d9d_dev packages/d9d-kernels-dev/src/d9d_kernels_dev

.PHONY: test test-local test-distributed test-distributed-shared-gpus

test-local:
	@echo "Running local tests for $(MODULE)..."
	@uv run python -m pytest $(MODULE) -m local

TORCHRUN_ENV =

test-distributed:
	@echo "Running distributed tests for $(MODULE)..."
	@$(TORCHRUN_ENV) uv run torchrun --nnodes 1 --nproc-per-node 8 --local-ranks-filter=0 --log_dir logs/dist_test --tee 3 -m pytest --instafail $(MODULE) -m distributed
	@echo "Please see the logs/dist_test for logs across all ranks if distributed tests did not pass"

# Runs the 8 ranks of the distributed tests on fewer GPUs, so that several ranks share one GPU.
# NCCL puts several ranks on one GPU only with NCCL_MULTI_RANK_GPU_ENABLE, and NCCL 2.30.7 failed in nvlsAllocateMem
# with NVLink SHARP on. Every rank holds its own communicators, so small buffers keep them in the memory of the GPU.
# MPS runs the kernels of the ranks of one GPU at the same time instead of in turns.
test-distributed-shared-gpus: TORCHRUN_ENV = NCCL_MULTI_RANK_GPU_ENABLE=1 NCCL_NVLS_ENABLE=0 NCCL_BUFFSIZE=262144 \
	NCCL_MAX_CTAS=1 NCCL_CUMEM_ENABLE=0 scripts/with_mps.sh
test-distributed-shared-gpus: test-distributed

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

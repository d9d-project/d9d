# Troubleshooting

## About

This page shows how to read the logs of a d9d job, the first step for any problem. If your problem is not here, ask in [Discord](https://discord.gg/sNRjDbxVrg) or open a [GitHub issue](https://github.com/d9d-project/d9d/issues).

## Reading the Logs

d9d logs to standard output on every rank. Each line starts with `[d9d]` and the coordinates of the rank in the base domain, e.g. `[d9d] [pp:0-dp:3-cp:0-tp:0]`.

`torchrun` can write the output of each rank to its own file. `--log-dir` sets the directory, and `--tee 3` also keeps the output in the console. `--local-ranks-filter 0` prints only local rank 0 to the console:

```bash
torchrun --nproc-per-node 8 --log-dir logs --tee 3 --local-ranks-filter 0 train.py config.yaml
```

When one rank fails, the other ranks often fail later with a timeout or a communication error. Find the rank that failed first and read its log.

To debug communication, set `NCCL_DEBUG=INFO` for NCCL logs and `TORCH_DISTRIBUTED_DEBUG=DETAIL` for checks of collective calls in PyTorch.

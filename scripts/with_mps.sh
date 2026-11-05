#!/bin/sh
# Runs a command under a private CUDA MPS daemon and stops the daemon when the command ends.
# A private pipe directory keeps other CUDA processes of the node off this daemon.
export CUDA_MPS_PIPE_DIRECTORY="/tmp/d9d-mps-$USER/pipe" CUDA_MPS_LOG_DIRECTORY="/tmp/d9d-mps-$USER/log"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
# A daemon left by a killed run makes the next run fail.
echo quit | nvidia-cuda-mps-control >/dev/null 2>&1
nvidia-cuda-mps-control -d || exit 1
trap 'echo quit | nvidia-cuda-mps-control' EXIT
# dash skips the EXIT trap when a signal ends it, so a signal exits through the trap.
trap 'exit 130' INT
trap 'exit 143' TERM
"$@"

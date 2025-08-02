#!/bin/sh

# DDP
CUDA_VISIBLE_DEVICES="0"

# set to 'true' to use torchrun, or 'false' for single-process training
USE_DDP=false

#! common arguments for run.py, edit this according to your experiment!
EXP_ARGS="--mode train --model-name VocoMorphUnet --device gpu"

if [ "$USE_DDP" = "true" ]; then
  # DDP-specific environment variables

  export MASTER_ADDR="localhost" #! change this!!
  export MASTER_PORT=29500
  export NNODES=1
  export NPROC_PER_NODE=1
  export NODE_RANK=0 #! change this!!

  # launch with torchrun
  torchrun \
    --nproc-per-node=$NPROC_PER_NODE \
    --nnodes=$NNODES \
    --node-rank=$NODE_RANK \
    --master-addr=$MASTER_ADDR \
    --master-port=$MASTER_PORT \
    run.py $EXP_ARGS
else
  # launch with a single process
  python run.py $EXP_ARGS
fi

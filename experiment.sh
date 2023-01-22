#!/bin/bash

name=$1
shift

wandb login
rm -rf log/$name && python3 -m src.pipeline.run_nerf --config src/configs/$name.txt $@ | tee text_log/$name.txt
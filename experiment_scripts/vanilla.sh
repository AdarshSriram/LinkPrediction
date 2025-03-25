#!/bin/bash
dataset=$1
savepath="results/vanilla/${dataset}.txt"
python3 models.py --dataset $dataset --model gcn --runs 2 --epochs 200 --results $savepath

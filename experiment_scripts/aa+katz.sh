#!/bin/bash
dataset=$1
savepath="results/aa+katz/${dataset}.txt"
runs=3
#k_lst=(10 30 50 100)
k_lst=(200)
pct_lst=(0.25)

for k in "${k_lst[@]}"; do
    for pct in "${pct_lst[@]}"; do
        echo "k: ${k} | pct: ${pct} | runs: ${runs}"
        python3 models.py --dataset $dataset --model gcn --feature-aug --feature-aug-k $k --random-pct $pct --embedding-aug aa --runs $runs --results $savepath
    done
done

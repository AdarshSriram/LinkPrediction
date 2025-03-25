#!/bin/bash
dataset=$1
savepath="results/featureaug/katz/${dataset}.txt"
runs=$2
#k_lst=(10 30 50 100)
k_lst=(200 100)
pct_lst=(0.0 0.1 0.2 0.25 0.4 0.5)

for k in "${k_lst[@]}"; do
    for pct in "${pct_lst[@]}"; do
        echo "k: ${k} | pct: ${pct} | runs: ${runs}"
        python3 models.py --dataset $dataset --model gcn --feature-aug --feature-aug-k $k --random-pct $pct --runs $runs --results $savepath
    done
done

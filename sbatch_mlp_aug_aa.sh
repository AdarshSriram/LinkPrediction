#!/bin/bash
#SBATCH -J mlpaug  # Job name
#SBATCH -o results/aug/AA/%j.log  # Name of stdout output file (%j expands to jobId)
#SBATCH -e results/aug/AA/augcmn_aa.err  # Name of stderr output file
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vg222@cornell.edu
#SBATCH -N 1  # Total number of CPU nodes requested
#SBATCH -n 4  # Total number of CPU cores requrested
#SBATCH -t 24:00:00  # Run time (hh:mm:ss)
#SBATCH --mem=50000  # CPU Memory pool for all cores
#SBATCH --partition=cuvl --gres=gpu:1
#SBATCH --get-user-env
# Put the command you want to run here. For example:
python3 models.py --dataset collab --model gcn --embedding-aug aa

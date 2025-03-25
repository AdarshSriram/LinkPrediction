#!/bin/bash
#SBATCH -J sgcind # Job name
#SBATCH -o results/aa+katz/%j.log  # Name of stdout output file (%j expands to jobId)
#SBATCH -e results/aa+katz/aug_katz%j.err  # Name of stderr output file
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vg222@cornell.edu
#SBATCH -N 1  # Total number of CPU nodes requested
#SBATCH -n 4  # Total number of CPU cores requrested
#SBATCH -t 12:00:00  # Run time (hh:mm:ss)
#SBATCH --mem=40000  # CPU Memory pool for all cores
#SBATCH  --gres=gpu:1
#SBATCH --get-user-env
# Put the command you want to run here. For example:
bash experiment_scripts/aa+katz.sh collab

#!/bin/bash
#SBATCH --job-name=stackexchange_incremental
#SBATCH --time=24:00:00
#SBATCH --mem=24G
#SBATCH --gpus-per-node=a40
#SBATCH --exclude=spot,heistotron
#SBATCH --output=Cobweb-Clustering-Language/slurm/slurm_outputs/stackexchange.out
#SBATCH --error=Cobweb-Clustering-Language/slurm/slurm_errors/stackexchange.err
#SBATCH --account="overcap"
#SBATCH --partition="overcap"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --qos="short"
#SBATCH --exclude="clippy"

export PYTHONUNBUFFERED=TRUE
source ~/.bashrc
conda activate rag-cobweb
cd ~/flash/Cobweb-Clustering-Language
export PYTHONPATH=$(pwd)

echo "Starting StackExchange Incremental Topic Modeling at $(date)"

srun python src/benchmarks/incremental_benchmark.py stackexchange --first-batch-size 2000 --batch-size 125 --max-docs 5000 --enable-refit

echo "StackExchange Incremental Topic Modeling completed at $(date)"

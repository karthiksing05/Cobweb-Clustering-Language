#!/bin/bash
#SBATCH --job-name=nytimes_incremental
#SBATCH --time=24:00:00
#SBATCH --mem=24G
#SBATCH --gpus-per-node=a40
#SBATCH --exclude=spot,heistotron
#SBATCH --output=Cobweb-Clustering-Language/slurm/slurm_outputs/nytimes_incremental.out
#SBATCH --error=Cobweb-Clustering-Language/slurm/slurm_errors/nytimes_incremental.err
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

echo "Starting NYTimes Annotated Corpus Incremental Topic Modeling at $(date)"

srun python src/benchmarks/incremental_benchmark.py nytimes --first-batch-size 500 --batch-size 250 --max-docs 10000

echo "NYTimes Annotated Corpus Incremental Topic Modeling completed at $(date)"

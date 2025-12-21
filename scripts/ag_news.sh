#!/bin/bash
#SBATCH --job-name=ag_news_topic_modeling
#SBATCH --time=5:00:00
#SBATCH --mem=24G
#SBATCH --gpus-per-node=a40
#SBATCH --exclude=spot,heistotron
#SBATCH --output=Cobweb-Clustering-Language/slurm/slurm_outputs/ag_news.out
#SBATCH --error=Cobweb-Clustering-Language/slurm/slurm_errors/ag_news.err
#SBATCH --partition="tail-lab"
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

echo "Starting AG News Topic Modeling at $(date)"

srun python src/benchmarks/benchmark.py agnews --max-docs 10000

echo "AG News Topic Modeling completed at $(date)"
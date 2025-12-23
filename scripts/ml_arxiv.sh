#!/bin/bash
#SBATCH --job-name=ml_arxiv_topic_modeling
#SBATCH --time=24:00:00
#SBATCH --mem=24G
#SBATCH --gpus-per-node=a40
#SBATCH --exclude=spot,heistotron
#SBATCH --output=Cobweb-Clustering-Language/slurm/slurm_outputs/ml_arxiv.out
#SBATCH --error=Cobweb-Clustering-Language/slurm/slurm_errors/ml_arxiv.err
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

echo "Starting ML ArXiv Topic Modeling at $(date)"

srun python src/benchmarks/benchmark.py ml_arxiv --max-docs 10000

echo "ML ArXiv Topic Modeling completed at $(date)"

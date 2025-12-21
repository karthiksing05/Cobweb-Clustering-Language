#!/bin/bash
#SBATCH --job-name=reuters_topic_modeling
#SBATCH --time=5:00:00
#SBATCH --mem=24G
#SBATCH --gpus-per-node=a40
#SBATCH --exclude=spot,heistotron
#SBATCH --output=Cobweb-Clustering-Language/slurm/slurm_outputs/reuters.out
#SBATCH --error=Cobweb-Clustering-Language/slurm/slurm_errors/reuters.err
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

echo "Starting Reuters Topic Modeling at $(date)"

srun python src/benchmarks/benchmark.py reuters

echo "Reuters Topic Modeling completed at $(date)"
#!/bin/bash
#SBATCH --job-name=twenty_newsgroups_topic_modeling
#SBATCH --time=5:00:00
#SBATCH --mem=24G
#SBATCH --gpus-per-node=a40
#SBATCH --output=Cobweb-Clustering-Language/slurm/slurm_outputs/twenty_newsgroups.out
#SBATCH --error=Cobweb-Clustering-Language/slurm/slurm_errors/twenty_newsgroups.err
#SBATCH --partition="overcap"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --qos="short"
#SBATCH --cpus-per-task=4

export PYTHONUNBUFFERED=TRUE
source ~/.bashrc
conda activate rag-cobweb
cd ~/flash/Cobweb-Clustering-Language
export PYTHONPATH=$(pwd)

echo "Starting 20 NewsGroups Topic Modeling at $(date)"

srun python src/benchmarks/benchmark.py 20newsgroups --test_hierarchical --reverse_levels

echo "20 NewsGroups Topic Modeling completed at $(date)"
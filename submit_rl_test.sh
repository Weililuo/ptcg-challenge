#!/bin/bash
#SBATCH --job-name=ptcg_dagger_test
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=00:30:00
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --export=ALL
#SBATCH --output=slurm-dagger-test-%j.out

cd /nas/longleaf/home/weililuo/ptcg_project

echo "=== Assigned Node & CPU Info ==="
echo "Node: $SLURM_NODELIST"
nproc

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# 微测试版：只跑 1 轮，极少对局数，用来验证闭环
~/.conda/envs/ptcg/bin/python3 run_dagger_loop.py \
  --rounds 1 \
  --matches 40 \
  --workers 8 \
  --arena-games 32 \
  --arena-workers 8 \
  --boost-rounds 10 \
  --lr 0.01 \
  --threshold 0.30 \
  --min-expert-winrate 0.20 \
  --min-champion-winrate 0.40 \
  --expert-frac 0.70 \
  --nemesis-boost 1.0 \
  --promote \
  --viz
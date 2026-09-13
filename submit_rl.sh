#!/bin/bash
#SBATCH --job-name=ptcg_dagger_sweetspot
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96g
#SBATCH --time=16:00:00
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --export=ALL
#SBATCH --output=slurm-dagger-sweetspot-%j.out

cd /nas/longleaf/home/weililuo/ptcg_project

echo "=== Assigned Node & CPU Info ==="
echo "Node: $SLURM_NODELIST"
nproc

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

[ -f /nas/longleaf/home/weililuo/ptcg_project/selfplay/arena/arena_rounds.csv ] && mv /nas/longleaf/home/weililuo/ptcg_project/selfplay/arena/arena_rounds.csv /nas/longleaf/home/weililuo/ptcg_project/selfplay/arena/arena_rounds.csv.bak

# Full-environment pool: 8 official expert decks + self-mirror (70/30).
# Arena is now mixed, so --threshold applies to a blended win rate:
#   0.70 x expert + 0.30 x champion-mirror, whose floor is
#   0.70*0.30 + 0.30*0.50 = 0.36. 0.40 keeps a little headroom above that.
# Re-calibrate after watching a few rounds of arena_rounds.csv.
~/.conda/envs/ptcg/bin/python3 run_dagger_loop.py \
  --rounds 150 \
  --matches 1000 \
  --workers 28 \
  --arena-games 400 \
  --arena-workers 28 \
  --boost-rounds 50 \
  --lr 0.01 \
  --threshold 0.40 \
  --min-expert-winrate 0.30 \
  --min-champion-winrate 0.50 \
  --expert-frac 0.70 \
  --nemesis-boost 1.0 \
  --promote \
  --viz
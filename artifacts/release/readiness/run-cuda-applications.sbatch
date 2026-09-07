#!/bin/bash
#SBATCH --job-name=curobo-app-parity
#SBATCH --partition=unlimited
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=cuda-applications-%j.log
set -euo pipefail
cd /common/home/ad2046/curobo-metal-cuda-replay/applications-20260907
python_bin=/common/home/ad2046/curobo-metal-cuda-replay/.venv/bin/python
upstream=/common/home/ad2046/curobo-metal-cuda-replay/upstream
unset PYTHONPATH PYTHONHOME PYTHONOPTIMIZE
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
"$python_bin" run.py run --suite applications --python "$python_bin" --backend cuda --expect-device cuda --upstream-source "$upstream" --output cuda-results
"$python_bin" run.py compare --suite applications --metal metal-report.json --cuda cuda-results/report.json --output comparison.json

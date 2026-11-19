#!/bin/bash
# Keep one protected-QOS runner of scripts/v2_queue.py queued until every step is done or failed.
cd /home/caroyf/myproject/ACMEC
source ~/miniconda3/etc/profile.d/conda.sh; conda activate hfpa; source activate.sh
while true; do
  left=$(python scripts/v2_queue.py --status | grep -c -E "^(todo|running)")
  [ "$left" -eq 0 ] && { echo "all steps finished"; break; }
  if ! squeue -u caroyf -h -o "%q" | grep -q protected; then
    j=$(sbatch --parsable --job-name=v2q slurm/gpu_p.sbatch scripts/v2_queue.py --budget 95 2>&1)
    echo "$(date -u +%H:%M) submitted $j ($left steps left)"
  fi
  sleep 120
done

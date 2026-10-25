#!/usr/bin/env bash
# Reproduce our rows of the paper's main results table (Table 2): 9 benchmarks,
# 1- and 5-shot (PerMIS is 1-shot only: each episode has one support frame).
#
#   GPUS="0 1 2 3" bash run_final.sh              # all 17 jobs, then the table
#   GPUS="0"       bash run_final.sh isic lung    # a subset of benchmarks
#   MAX_EPISODES=50 bash run_final.sh isic        # quick smoke test
#
# One job per (benchmark, k); each job runs every fold of the benchmark and
# writes results/k{k}/{benchmark}/results.json. Jobs are spread round-robin
# over $GPUS and a finished job is skipped on re-runs, so the script resumes.
# Seed 1, all folds, all episodes: the main table's protocol (PerMIS episodes are
# fixed, so the seed does not affect them). Expect several GPU-days in total
# (LVIS-92i: 10 folds x 2300 episodes; 5-shot LVIS peaks at ~19 GB).
#
# Requires: $DINO_WEIGHTS (DINOv3 ViT-L/16 checkpoint), $DATA_ROOT
# (benchmark root, see README), and python with requirements.txt installed.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"
GPUS="${GPUS:-0}"
SHOTS="${SHOTS:-1 5}"
MAX_EPISODES="${MAX_EPISODES:-0}"
RESULTS="${RESULTS:-results}"
DATASETS="${*:-lvis coco isic suim isaid lung pascal_part paco_part permis}"

: "${DINO_WEIGHTS:?set DINO_WEIGHTS to the DINOv3 ViT-L/16 checkpoint}"
: "${DATA_ROOT:?set DATA_ROOT to the benchmark root}"

read -ra gpu_list <<< "$GPUS"
mkdir -p "$RESULTS/logs"
pids=()
i=0
for k in $SHOTS; do
  for ds in $DATASETS; do
    [[ $ds == permis && $k != 1 ]] && continue   # PerMIS has one support per episode
    out="$RESULTS/k$k/$ds/results.json"
    if [[ -f "$out" ]]; then
      echo "skip $ds k=$k ($out exists)"
      continue
    fi
    gpu="${gpu_list[$((i % ${#gpu_list[@]}))]}"
    i=$((i + 1))
    log="$RESULTS/logs/${ds}_k${k}.log"
    echo "launch $ds k=$k on GPU $gpu -> $log"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" eval.py --dataset "$ds" --shots "$k" \
        --seed 1 --max-episodes "$MAX_EPISODES" --output "$out" > "$log" 2>&1 &
    pids+=($!)
  done
done

fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
if [[ $fail -ne 0 ]]; then
  echo "some jobs failed; see $RESULTS/logs/*.log. Re-run to resume." >&2
  exit 1
fi

# ---- aggregate: mean over folds per cell, mean over the row's benchmarks for Avg ----
"$PYTHON" - "$RESULTS" <<'EOF'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
cols = [("lvis", "LVIS-92i"), ("coco", "COCO-20i"), ("isic", "ISIC"), ("suim", "SUIM"),
        ("isaid", "iSAID"), ("lung", "X-Ray"), ("pascal_part", "PASCAL"), ("paco_part", "PACO"),
        ("permis", "PerMIS")]
# baseline rows, copied from the paper's Table 2 (not run by this script):
# Matcher/GF-SAM (marked †) as reported by INSID3, the others our runs of their public code;
# the last entry is the Avg as printed in the paper
baselines = {
    1: {"Matcher†": [33.0, 52.7, 38.6, 44.1, 33.3, 70.8, 42.9, 34.7, 63.8, 46.0],
        "GF-SAM†":  [35.2, 58.7, 48.7, 53.1, 47.1, 51.0, 44.5, 36.3, 54.1, 47.6],
        "FSSDINO":  [33.8, 51.3, 53.0, 52.5, 47.2, 76.7, 50.3, 35.2, 48.5, 49.8],
        "INSID3":   [41.5, 58.3, 54.0, 53.4, 51.7, 78.6, 49.1, 38.5, 67.0, 54.7],
        "FROST":    [32.3, 48.1, 54.4, 58.1, 49.1, 84.8, 54.3, 36.8, 66.6, 53.8]},
    5: {"Matcher†": [40.0, 60.7, 35.0, 50.6, 34.3, 71.2, 45.8, 33.6, None, 46.4],
        "GF-SAM†":  [44.2, 66.8, 55.2, 58.1, 52.4, 52.9, 51.2, 41.9, None, 52.8],
        "FSSDINO":  [43.4, 60.6, 63.6, 62.8, 58.5, 78.6, 60.8, 45.8, None, 59.2],
        "INSID3":   [47.5, 64.3, 63.9, 63.1, 60.1, 79.4, 58.4, 47.0, None, 60.5],
        "FROST":    [47.5, 61.8, 66.7, 67.2, 61.0, 86.4, 68.0, 51.3, None, 63.7]},
}

def fmt(v):
    return "--" if v is None else f"{v:.1f}"

def avg(vals, k):
    # PerMIS counts only at 1-shot; a row with a missing expected cell gets no Avg
    need = [v for (key, _), v in zip(cols, vals) if k == 1 or key != "permis"]
    return None if None in need else sum(need) / len(need)

lines = ["| Method | " + " | ".join(n for _, n in cols) + " | Avg |",
         "|---|" + "---:|" * (len(cols) + 1)]
for k in (1, 5):
    lines.append(f"| **{k}-shot** |" + " |" * (len(cols) + 1))
    for name, vals in baselines[k].items():
        lines.append(f"| {name} | " + " | ".join(fmt(v) for v in vals) + " |")
    ours = []
    for key, _ in cols:
        f = root / f"k{k}" / key / "results.json"
        ours.append(json.load(f.open())["miou"] if f.exists() else None)
    lines.append("| **Ours** | " + " | ".join(fmt(v) for v in ours) + f" | {fmt(avg(ours, k))} |")
table = "\n".join(lines)
(root / "table.md").write_text(table + "\n")
print(table)
print(f"\nwritten to {root/'table.md'}")
EOF

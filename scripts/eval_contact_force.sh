#!/usr/bin/env bash
# Evaluate the pretrained Contact-Force models on the test split of every object
# (paper Table II; the average is the Contact-Force column of Table I).
#
# Usage: bash scripts/eval_contact_force.sh [extra evaluate.py arguments, e.g. --device cpu]
set -euo pipefail
cd "$(dirname "$0")/.."

OBJECTS=(bottle cat dog donut doritos flipflop pillow)
for o in "${OBJECTS[@]}"; do
  python evaluate.py --config "configs/contact_force/$o.yaml" \
    --checkpoint "checkpoints/contact_force/$o.pth" "$@"
done

python - "${OBJECTS[@]}" <<'EOF'
import json, sys
rows = [(o, json.load(open(f"outputs/contact_force/{o}/metrics_test.json"))["overall_mm"]) for o in sys.argv[1:]]
print(f"\n{'Object':<10}{'RMSE (mm)':>12}{'MAE (mm)':>12}{'MSE (mm^2)':>12}")
for o, m in rows:
    print(f"{o:<10}{m['rmse']:>12.3f}{m['mae']:>12.3f}{m['mse']:>12.3f}")
avg = {k: sum(m[k] for _, m in rows) / len(rows) for k in ("rmse", "mae", "mse")}
print(f"{'Average':<10}{avg['rmse']:>12.3f}{avg['mae']:>12.3f}{avg['mse']:>12.3f}")
EOF

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
SEEDS=${SEEDS:-"0 1 2"}
for S in $SEEDS; do
  SEED=$S OUT="runs/a800_synthetic_nonlinear_seed${S}" bash scripts/run_a800_synthetic_two_gpu.sh
done
python - <<'PY'
from pathlib import Path
import pandas as pd
rows=[]
for p in sorted(Path('runs').glob('a800_synthetic_nonlinear_seed*/comparison.csv')):
    df=pd.read_csv(p)
    df['seed_dir']=p.parent.name
    rows.append(df)
out=Path('runs/a800_synthetic_nonlinear_all_seeds.csv')
out.parent.mkdir(exist_ok=True)
pd.concat(rows, ignore_index=True).to_csv(out,index=False)
print('Wrote', out)
PY

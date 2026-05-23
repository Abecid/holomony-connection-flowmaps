#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
SEEDS=${SEEDS:-"0 1 2"}
OUT_ROOT=${OUT_ROOT:-runs/synthetic_nonlinear}
export OUT_ROOT
SEEDS="$SEEDS" OUT_ROOT="$OUT_ROOT" bash scripts/run_synthetic.sh
python - <<'PY'
import os
from pathlib import Path
import pandas as pd
rows=[]
out_root = os.environ.get('OUT_ROOT', 'runs/synthetic_nonlinear')
root = Path(out_root)
for p in sorted(root.parent.glob(f'{root.name}_seed*/comparison.csv')):
    df=pd.read_csv(p)
    df['seed_dir']=p.parent.name
    rows.append(df)
out=Path(f'{out_root}_all_seeds.csv')
out.parent.mkdir(exist_ok=True)
pd.concat(rows, ignore_index=True).to_csv(out,index=False)
print('Wrote', out)
PY

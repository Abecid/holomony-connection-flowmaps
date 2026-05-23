from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Aggregate final affine-MNIST metrics across model run dirs.")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for r in args.runs:
        p = Path(r)
        csv_path = p / "metrics.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        row = df.iloc[-1].to_dict()
        row["run"] = str(p)
        row["model"] = p.name
        rows.append(row)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

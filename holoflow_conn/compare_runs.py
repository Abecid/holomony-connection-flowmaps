from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect final eval rows from run directories.")
    p.add_argument("--runs", nargs="+", required=True, help="Run directories containing metrics.csv or eval_metrics.csv.")
    p.add_argument("--out", type=str, default="runs/comparison.csv")
    return p.parse_args()


def final_row(run_dir: Path) -> dict:
    eval_csv = run_dir / "eval_metrics.csv"
    if eval_csv.exists():
        row = pd.read_csv(eval_csv).iloc[-1].to_dict()
    else:
        metrics = run_dir / "metrics.csv"
        if not metrics.exists():
            raise FileNotFoundError(f"No metrics.csv/eval_metrics.csv in {run_dir}")
        row = pd.read_csv(metrics).iloc[-1].to_dict()
    row["run_dir"] = str(run_dir)
    if "model" not in row:
        # infer from path / config fallback
        row["model"] = run_dir.name
    return row


def main() -> None:
    args = parse_args()
    rows = [final_row(Path(r)) for r in args.runs]
    df = pd.DataFrame(rows)
    preferred = [
        "model", "run_dir", "step", "local_mse", "ab_ba_endpoint_mse", "ab_ba_comm_mse",
        "ab_ba_comm_true_norm", "ab_ba_comm_pred_norm", "ab_ba_comm_cosine",
        "id_hol_mse", "id_hol_true_norm", "id_hol_pred_norm", "id_hol_cosine",
        "ood_ab_ba_endpoint_mse", "ood_ab_ba_comm_mse", "ood_hol_mse",
        "flatness_norm2",
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df = df[cols]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df[cols[:min(len(cols), 18)]].to_string(index=False))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()

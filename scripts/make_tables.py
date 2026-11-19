# -*- coding: utf-8 -*-
"""Regenerate the per-fold and mean +/- std tables from results/hcemor/muenlhd_result.xlsx.

Usage:
    python scripts/make_tables.py [--results results/hcemor/muenlhd_result.xlsx] [--out results/tables]

Input columns (as produced by the experiments): fold, close, rand, pre_mic, pre_mac,
rec_mic, rec_mac, f1_mic, f1_mac, auc.
Output: one CSV per setting plus a combined markdown table with mean and std over folds.
"""
import argparse
import os
import sys

METRICS = ["pre_mic", "pre_mac", "rec_mic", "rec_mac", "f1_mic", "f1_mac", "auc"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/hcemor/muenlhd_result.xlsx")
    ap.add_argument("--out", default="results/tables")
    args = ap.parse_args()

    try:
        import pandas as pd
    except ImportError:
        sys.exit("pandas is required: pip install pandas openpyxl")

    if not os.path.isfile(args.results):
        sys.exit("not found: " + args.results)

    df = pd.read_excel(args.results)
    df.columns = [str(c).strip() for c in df.columns]
    os.makedirs(args.out, exist_ok=True)

    lines = ["| setting | fold | " + " | ".join(METRICS) + " |",
             "|---|---|" + "---|" * len(METRICS)]
    for (close, rand), g in df.groupby(["close", "rand"], sort=True):
        for _, row in g.sort_values("fold").iterrows():
            vals = " | ".join("%.4f" % float(row[m]) for m in METRICS)
            lines.append("| %s/rand%s | %s | %s |" % (close, rand, row["fold"], vals))
        mean = g[METRICS].mean()
        std = g[METRICS].std()
        mv = " | ".join("%.4f" % mean[m] for m in METRICS)
        sv = " | ".join("%.4f" % std[m] for m in METRICS)
        lines.append("| %s/rand%s | **mean** | %s |" % (close, rand, mv))
        lines.append("| %s/rand%s | **std** | %s |" % (close, rand, sv))
        g.to_csv(os.path.join(args.out, "%s_rand%s_per_fold.csv" % (close, rand)), index=False)

    md = "\n".join(lines)
    with open(os.path.join(args.out, "tables.md"), "w", encoding="utf-8") as f:
        f.write(md + "\n")
    print(md)


if __name__ == "__main__":
    main()

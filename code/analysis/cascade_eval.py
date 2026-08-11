#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A confidence-gated cascade that could actually be deployed.

`paired_routing.py` reports the oracle and stratum policies, which use gold-derived
difficulty and therefore bound what a perfect router could achieve. This script
closes the loop with a REALIZABLE policy: answer everything directly, then escalate
only the least-confident fraction to reasoning, using the direct-answer confidence
from `log_confidence.py` as the sole routing signal. No gold is consulted.

Per split, over ten seeds, it reports:
  - the all-direct and all-reasoning baselines, and the oracle / stratum ceilings;
  - the cascade at escalation fractions of 10-50%;
  - mean confidence per subset, as a diagnostic: if confidence does not separate
    the easy strata (A, C) from the hard ones (B, D), the router has no signal to
    work with and the cascade cannot beat a coin flip.

The last line per fraction is the one that matters: what share of the ORACLE GAIN
the deployable cascade recovers, and at what share of the reasoning latency.

Pairs per question by row order -- valid because every file comes from the same
per-seed pkl -- and asserts that the gold sequences agree before doing so.

Run `log_confidence.py` first.

Usage
  python cascade_eval.py --technique distill_sft --alpha-suffix ""
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT_ROOT = Path("outputs_stage2")
SEEDS = [8, 12, 17, 23, 25, 31, 37, 44, 52, 61]
FRACTIONS = [0.10, 0.20, 0.30, 0.40, 0.50]
SUBSETS = ["A", "B", "C", "H"]          # H is the legacy code for stratum D
LEGACY_COLUMNS = {"acertou": "correct", "acertou_conf": "correct_conf"}
LEGACY_TIMING_KEYS = {"secs_per_questao": "secs_per_question"}
LEGACY_DIR_NAMES = {"pure_sft": "sft_puro"}


def load_seed(run: Path, split: str):
    direct = pd.read_csv(run / f"infer_{split}_answer_only.csv").rename(columns=LEGACY_COLUMNS)
    reason = pd.read_csv(run / f"infer_{split}_reasoning.csv").rename(columns=LEGACY_COLUMNS)
    conf = pd.read_csv(run / f"conf_{split}.csv").rename(columns=LEGACY_COLUMNS)
    n = min(len(direct), len(reason), len(conf))
    gold = direct["gold"].to_numpy()[:n]
    assert (reason["gold"].to_numpy()[:n] == gold).all() and \
           (conf["gold"].to_numpy()[:n] == gold).all(), \
        f"PAIRING BROKEN: gold mismatch in {run.name} {split}"

    timing = json.load(open(run / "infer_timing.json"))
    spq = {}
    for key, v in timing.items():
        v = {LEGACY_TIMING_KEYS.get(k, k): val for k, val in v.items()}
        spq[key] = v["secs_per_question"]

    return pd.DataFrame({
        "subset": direct["subset_code"].astype(str).str.upper().to_numpy()[:n],
        "direct": direct["correct"].to_numpy()[:n],
        "reason": reason["correct"].to_numpy()[:n],
        "conf": conf["conf"].to_numpy()[:n],
        "lat_direct": spq.get(f"{split}_answer_only", np.nan),
        "lat_reason": spq.get(f"{split}_reasoning", np.nan),
    })


def metrics(df, route_mask):
    """Accuracy and latency when `route_mask` selects the reasoning outcome."""
    acc = np.where(route_mask, df["reason"], df["direct"]).mean()
    lat = np.where(route_mask, df["lat_reason"], df["lat_direct"]).mean()
    return acc, lat, route_mask.mean()


def eval_split(per_seed):
    policies = (["all_direct", "all_reasoning", "oracle", "stratum"]
                + [f"cascade_{int(f*100)}" for f in FRACTIONS])
    acc = {k: [] for k in policies}
    lat = {k: [] for k in policies}
    frac = {k: [] for k in policies}
    conf_by_subset = {s: [] for s in SUBSETS}

    for df in per_seed:
        n = len(df)
        subset = df["subset"].to_numpy()
        for s in SUBSETS:
            m = subset == s
            if m.any():
                conf_by_subset[s].append(df["conf"].to_numpy()[m].mean())

        fixed = [
            ("all_direct", np.zeros(n, bool)),
            ("all_reasoning", np.ones(n, bool)),
            ("oracle", (df["direct"].to_numpy() == 0) & (df["reason"].to_numpy() == 1)),
            ("stratum", np.isin(subset, ["B", "D", "H"])),
        ]
        for key, mask in fixed:
            a, l, f = metrics(df, mask)
            acc[key].append(a); lat[key].append(l); frac[key].append(f)

        # cascade: escalate the LOWEST-confidence fraction
        order = np.argsort(df["conf"].to_numpy())          # ascending
        for f in FRACTIONS:
            k = int(n * f)
            mask = np.zeros(n, bool)
            mask[order[:k]] = True
            a, l, fr = metrics(df, mask)
            key = f"cascade_{int(f*100)}"
            acc[key].append(a); lat[key].append(l); frac[key].append(fr)

    reasoning_latency = np.mean(lat["all_reasoning"])
    table = pd.DataFrame([{
        "policy": k,
        "accuracy": round(float(np.mean(acc[k])), 4),
        "pct_reasoning_calls": round(100 * float(np.mean(frac[k])), 1),
        "pct_reasoning_latency": round(100 * float(np.mean(lat[k])) / reasoning_latency, 1),
    } for k in policies])
    diagnostic = {s: round(float(np.mean(v)), 3) for s, v in conf_by_subset.items() if v}
    return table, diagnostic


def resolve_run(root, technique, seed, epochs, suffix):
    for name in [technique] + ([LEGACY_DIR_NAMES[technique]]
                               if technique in LEGACY_DIR_NAMES else []):
        d = root / f"{name}_seed{seed}_ep{epochs}{suffix}"
        if d.exists():
            return d
    return root / f"{technique}_seed{seed}_ep{epochs}{suffix}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="outputs_stage2")
    ap.add_argument("--technique", default="distill_sft")
    ap.add_argument("--alpha-suffix", default="")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--out", default=None, help="optional CSV path")
    args = ap.parse_args()
    root = Path(args.root)

    collected = []
    for split in ("hard", "abc"):
        per_seed = [load_seed(resolve_run(root, args.technique, s, args.epochs,
                                          args.alpha_suffix), split)
                    for s in SEEDS]
        table, diagnostic = eval_split(per_seed)
        table.insert(0, "split", split)
        collected.append(table)

        print(f"\n=== {args.technique}{args.alpha_suffix or ' (alpha=1.0)'} "
              f"| split={split} | {len(SEEDS)} seeds ===")
        print(table.drop(columns="split").to_string(index=False))
        print(f"mean confidence by subset (does it separate easy A/C from hard B/D?): "
              f"{diagnostic}")

        baseline = table.loc[table.policy == "all_direct", "accuracy"].values[0]
        oracle = table.loc[table.policy == "oracle", "accuracy"].values[0]
        gain = oracle - baseline if oracle > baseline else np.nan
        for f in FRACTIONS:
            row = table.loc[table.policy == f"cascade_{int(f*100)}"].iloc[0]
            recovered = ((row.accuracy - baseline) / gain * 100
                         if gain and gain > 0 else np.nan)
            print(f"  cascade@{int(f*100)}%: acc={row.accuracy:.3f} "
                  f"lat={row.pct_reasoning_latency:.0f}% of all-reasoning, "
                  f"recovers {recovered:.0f}% of the oracle gain")

    if args.out:
        pd.concat(collected, ignore_index=True).to_csv(args.out, index=False)
        print(f"\nsaved={args.out}")


if __name__ == "__main__":
    main()

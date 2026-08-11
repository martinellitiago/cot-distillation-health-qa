#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline routing analysis: how much of the ~77x reasoning cost is avoidable?

Reasoning is not uniformly better than answering directly -- it redistributes
accuracy, helping exactly where direct answering fails and hurting where it
already succeeds. So the deployable question is not "reason or not" but "reason
WHERE". This script answers it without training a router, by pairing each
question's direct and reasoning outcome and evaluating four policies:

  all_direct     answer everything directly (cheap; weak on B and hard)
  all_reasoning  reason on everything (the full ~77x cost)
  oracle         reason ONLY where it flips a wrong direct answer to correct.
                 This is the ACHIEVABLE CEILING of a perfect router, not a
                 deployable policy: it uses the outcome to decide.
  stratum        a rule that uses the paired partition as a difficulty signal:
                 reason on B and hard, answer A and C directly. Also gold-derived,
                 so it bounds the achievable benefit rather than describing a
                 deployed system.

Both non-trivial policies use gold-derived difficulty, so they measure HEADROOM.
The gap between them and `all_reasoning` is what a real learned router could aim
at; the gap between them and `all_direct` is what it would have to earn.

Results are reported PER SPLIT. Pooling abc and hard dilutes the effect with the
A and C items, where reasoning does not help -- the same aggregation tautology
that hides the transfer effect everywhere else in this project.

Usage
  python paired_routing.py --root outputs_stage2 --technique distill_sft
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEEDS = [8, 12, 17, 23, 25, 31, 37, 44, 52, 61]
LEGACY_COLUMNS = {"acertou": "correct"}
LEGACY_TIMING_KEYS = {"secs_per_questao": "secs_per_question"}


def load_regime(run: Path, split: str, regime: str):
    f = run / f"infer_{split}_{regime}.csv"
    if not f.exists():
        raise FileNotFoundError(f"missing {f}")
    return pd.read_csv(f).rename(columns=LEGACY_COLUMNS).reset_index(drop=True)


def seconds_per_question(run: Path):
    timing = json.load(open(run / "infer_timing.json"))
    out = {}
    for key, v in timing.items():
        v = {LEGACY_TIMING_KEYS.get(k, k): val for k, val in v.items()}
        out[key] = v["secs_per_question"]
    return out


def one_seed(run: Path):
    """Per-question arrays, keeping the split so routing can be reported per split."""
    spq = seconds_per_question(run)
    frames = []
    for split in ("abc", "hard"):
        direct = load_regime(run, split, "answer_only").rename(
            columns={"correct": "direct"})
        reason = load_regime(run, split, "reasoning").rename(
            columns={"correct": "reason"})
        n = min(len(direct), len(reason))
        # the two regimes must describe the SAME questions in the SAME order,
        # otherwise pairing them per question is meaningless
        assert (direct["gold"].to_numpy()[:n] == reason["gold"].to_numpy()[:n]).all(), \
            f"PAIRING BROKEN: gold mismatch in {run.name} {split}"
        frames.append(pd.DataFrame({
            "split": split,
            "subset": direct["subset_code"].astype(str).str.upper().to_numpy()[:n],
            "direct": direct["direct"].to_numpy()[:n],
            "reason": reason["reason"].to_numpy()[:n],
            "lat_direct": spq.get(f"{split}_answer_only", np.nan),
            "lat_reason": spq.get(f"{split}_reasoning", np.nan),
        }))
    return pd.concat(frames, ignore_index=True)


def policies(df):
    """Each policy picks, per question, direct or reasoning; latency uses that
    question's own split-specific per-regime latency."""
    direct = df["direct"].to_numpy()
    reason = df["reason"].to_numpy()
    subset = df["subset"].to_numpy()
    lat_d = df["lat_direct"].to_numpy()
    lat_r = df["lat_reason"].to_numpy()

    out = {
        "all_direct": (direct.mean(), lat_d.mean(), 0.0),
        "all_reasoning": (reason.mean(), lat_r.mean(), 1.0),
    }
    rescued = (direct == 0) & (reason == 1)         # oracle: reason only where it saves
    out["oracle"] = (np.where(rescued, reason, direct).mean(),
                     np.where(rescued, lat_r, lat_d).mean(),
                     rescued.mean())
    to_reason = np.isin(subset, ["B", "D", "H"])    # stratum rule (H = legacy code for D)
    out["stratum"] = (np.where(to_reason, reason, direct).mean(),
                      np.where(to_reason, lat_r, lat_d).mean(),
                      to_reason.mean())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="outputs_stage2")
    ap.add_argument("--technique", default="distill_sft")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--alpha-suffix", default="", help="'' for alpha=1.0, '_a01', '_a03'")
    ap.add_argument("--out", default="routing_analysis.csv")
    args = ap.parse_args()
    root = Path(args.root)

    per_seed = [one_seed(root / f"{args.technique}_seed{s}_ep{args.epochs}{args.alpha_suffix}")
                for s in SEEDS]

    ORDER = ("all_direct", "oracle", "stratum", "all_reasoning")
    rows = []
    for scope in ("hard", "abc", "combined"):
        agg = {p: {"acc": [], "lat": [], "frac": []} for p in ORDER}
        for df in per_seed:
            scoped = df if scope == "combined" else df[df["split"] == scope]
            for p, (acc, lat, frac) in policies(scoped).items():
                agg[p]["acc"].append(acc)
                agg[p]["lat"].append(lat)
                agg[p]["frac"].append(frac)
        reasoning_latency = np.mean(agg["all_reasoning"]["lat"])
        for p in ORDER:
            rows.append({
                "scope": scope, "policy": p,
                "accuracy": round(float(np.mean(agg[p]["acc"])), 4),
                "secs_per_question": round(float(np.mean(agg[p]["lat"])), 4),
                "pct_reasoning_calls": round(100 * float(np.mean(agg[p]["frac"])), 1),
                "pct_reasoning_latency": round(
                    100 * float(np.mean(agg[p]["lat"])) / reasoning_latency, 1),
            })

    res = pd.DataFrame(rows)
    res.to_csv(args.out, index=False)
    print(f"Technique={args.technique}{args.alpha_suffix or ' (alpha=1.0)'}  "
          f"{len(SEEDS)} seeds")
    print("Reported PER SPLIT. `combined` is diluted by A and C and only shows that")
    print("blanket reasoning is dominated -- the gain lives in hard and in subset B.")
    print("`oracle` and `stratum` use gold-derived difficulty: they bound the")
    print("ACHIEVABLE benefit, they are not deployable policies.\n")
    with pd.option_context("display.width", 160):
        print(res.to_string(index=False))


if __name__ == "__main__":
    main()

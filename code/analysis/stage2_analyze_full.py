#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 2 analysis -- rebuilds every reported table from the question-level
predictions. Nothing is transcribed from memory; every number comes from a file.

Produces
  accuracy.csv                        technique x alpha x regime x split x subset,
                                      seed means with two-level bootstrap intervals
  paired_tests.csv                    paired t across seeds (confirmatory) plus
                                      exact within-seed McNemar (diagnostic), on
                                      the whole split AND on the focus subset
  tradeoff_questions_vs_baseline.csv  questions gained/lost per subset vs pure SFT
  compute_efficiency.csv              tokens, tokens/s, s/question, cost frontier
  training_summary.csv                final loss, steps
  loss_curve_by_condition.csv         convergence
  confounder_checks.csv               truncation / English reasoning / emission

METHODOLOGICAL RULES (do not violate)
  - never transcribe a number from memory: everything is read from a file
  - subset B is the thermometer; the aggregate hides the trade-off
  - question-level pairing requires the SAME question order: this script ASSERTS
    that the gold sequence matches between paired runs before any McNemar
  - `mcnemar_p_min_*` is the MINIMUM p across seeds. It is DESCRIPTIVE (the best
    seed), never a global test. The confirmatory test is the paired t across
    seeds, with Wilcoxon as sensitivity.
  - this script enumerates ALL condition pairs and applies NO multiplicity
    correction, by design: it is exploratory output. The paper pre-declares four
    primary hypotheses (content causality, format neutrality, technique x regime,
    the alpha effect) and treats everything else as exploratory.

Usage
  python stage2_analyze_full.py                       # defaults
  python stage2_analyze_full.py --figures             # + diagnostic figures
  python stage2_analyze_full.py --root outputs_stage2 --out analysis_stage2
"""

import argparse
import itertools
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy import stats as st
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[warning] scipy unavailable: McNemar and t tests will be skipped")

# project palette
INK, ORANGE, GOLD = "#3A3226", "#D97757", "#C9A227"
EXTRA = ["#8A7B66", "#B5651D", "#6B5B4A"]

# Run directory names. `sft_puro` is the original Portuguese identifier used by
# the archived runs; it is accepted here and normalised to `pure_sft`.
RUN_RE = re.compile(
    r"^(?P<tech>pure_sft|sft_puro|distill_sft|step_by_step)"
    r"_seed(?P<seed>\d+)"
    r"_ep(?P<ep>\d+)"
    r"(?:_frac(?P<frac>\d+))?"
    r"(?:_a(?P<alpha>[\d]+))?$"
)
LEGACY_TECHNIQUE_NAMES = {"sft_puro": "pure_sft"}

# Column names written by earlier (Portuguese) versions of the run script, so the
# released analysis reproduces the archived runs unchanged.
LEGACY_COLUMNS = {
    "acertou": "correct",
    "truncou": "truncated",
    "racioc_chars": "reasoning_chars",
    "racioc_en": "reasoning_en",
}
LEGACY_TIMING_KEYS = {
    "n_questoes": "n_questions",
    "secs_per_questao": "secs_per_question",
}

SPLITS = ["abc", "hard"]
REGIMES = ["answer_only", "reasoning"]


# ---------------------------------------------------------
# 0. DISCOVERY AND LOADING
# ---------------------------------------------------------
def parse_alpha(tag):
    """Recover alpha from a run-directory tag: '01' -> 0.1, '03' -> 0.3.

    AUDIT NOTE: the run script writes f'_a{alpha:g}'.replace('.', ''), which is
    lossy -- 0.05 becomes 'a005' and 1.5 becomes 'a15', neither invertible. This
    parser therefore only accepts the single-digit 0.x family actually used
    (0.1, 0.3, 0.5) and raises otherwise instead of returning a wrong value
    silently, as it previously did (0.05 -> 0.5, 1.5 -> 15.0). run_config.json
    holds the authoritative value and takes precedence in discover_runs()."""
    if tag is None:
        return 1.0
    if tag.startswith("0"):
        stripped = tag.lstrip("0")
        if len(stripped) != 1:
            raise ValueError(
                f"ambiguous alpha tag '_a{tag}': not invertible. "
                f"Read alpha from run_config.json instead.")
        return float("0." + stripped)
    raise ValueError(
        f"ambiguous alpha tag '_a{tag}': values >= 1 are not round-trippable. "
        f"Read alpha from run_config.json instead.")


def discover_runs(root: Path, require_done=True):
    runs = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        m = RUN_RE.match(d.name)
        if not m:
            continue
        if require_done and not (d / "DONE").exists():
            print(f"[skipping: no DONE marker] {d.name}")
            continue
        info = m.groupdict()
        cfg = {}
        cfg_path = d / "run_config.json"
        if cfg_path.exists():
            cfg = json.load(open(cfg_path))
        runs.append({
            "dir": d,
            "run_name": d.name,
            "tech": LEGACY_TECHNIQUE_NAMES.get(info["tech"], info["tech"]),
            "seed": int(info["seed"]),
            "epochs": int(info["ep"]),
            "frac": int(info["frac"]) / 100 if info["frac"] else 1.0,
            "alpha": cfg.get("alpha", parse_alpha(info["alpha"])),
            "mnt_reasoning": cfg.get("mnt_reasoning"),
            "mnt_answer": cfg.get("mnt_answer"),
            "config": cfg,
        })
    return runs


def condition_label(r):
    """The experimental condition WITHOUT the seed -- the correct unit of analysis."""
    label = r["tech"]
    if r["alpha"] != 1.0:
        label += f"_a{r['alpha']:g}"
    if r["frac"] != 1.0:
        label += f"_frac{int(r['frac']*100)}"
    return label


def check_ceilings(runs):
    """A condition comparison is only clean if every run shares the same generation
    ceiling; a differing ceiling truncates one arm and depresses its accuracy."""
    for field in ("mnt_reasoning", "mnt_answer"):
        values = {r[field] for r in runs if r.get(field) is not None}
        if len(values) > 1:
            raise SystemExit(
                f"INCONSISTENT CEILING: runs use different {field} values "
                f"{sorted(values)}. Accuracies are not comparable across them -- "
                f"split the analysis by ceiling.")
        if values:
            print(f"  {field} = {values.pop()} across all runs")


def load_question_level(runs):
    """One row per (run, split, regime, question)."""
    frames = []
    for r in runs:
        for split in SPLITS:
            for regime in REGIMES:
                f = r["dir"] / f"infer_{split}_{regime}.csv"
                if not f.exists():
                    continue          # pure_sft has no reasoning regime: expected
                df = pd.read_csv(f).rename(columns=LEGACY_COLUMNS)
                # row order == the order of the per-seed pkl, which is what makes
                # question-level pairing valid; asserted later against `gold`
                df["q_idx"] = np.arange(len(df))
                df["cond"] = condition_label(r)
                df["tech"] = r["tech"]
                df["alpha"] = r["alpha"]
                df["seed"] = r["seed"]
                df["split"] = split
                df["regime"] = regime
                df["run_name"] = r["run_name"]
                frames.append(df)
    if not frames:
        raise SystemExit("No infer_*.csv found. Did the runs finish?")
    return pd.concat(frames, ignore_index=True)


def load_timing(runs):
    rows = []
    for r in runs:
        f = r["dir"] / "infer_timing.json"
        if not f.exists():
            continue
        for key, v in json.load(open(f)).items():
            split, regime = key.split("_", 1)
            v = {LEGACY_TIMING_KEYS.get(k, k): val for k, val in v.items()}
            rows.append({"cond": condition_label(r), "tech": r["tech"], "alpha": r["alpha"],
                         "seed": r["seed"], "split": split, "regime": regime, **v})
    return pd.DataFrame(rows)


def load_training(runs):
    rows, curves = [], []
    for r in runs:
        summary = r["dir"] / "train_summary.json"
        if summary.exists():
            js = json.load(open(summary))
            rows.append({"cond": condition_label(r), "tech": r["tech"], "alpha": r["alpha"],
                         "seed": r["seed"], "train_loss": js.get("train_loss"),
                         "steps": js.get("steps")})
        curve = r["dir"] / "train_loss_curve.csv"
        if curve.exists():
            cv = pd.read_csv(curve)
            cv["cond"] = condition_label(r)
            cv["seed"] = r["seed"]
            curves.append(cv)
    return (pd.DataFrame(rows),
            pd.concat(curves, ignore_index=True) if curves else pd.DataFrame())


# ---------------------------------------------------------
# 1. ACCURACY + TWO-LEVEL BOOTSTRAP
# ---------------------------------------------------------
def two_level_bootstrap(correct_by_seed, n_boot=10000, rng=None):
    """Resample SEEDS with replacement, then QUESTIONS within each chosen seed.

    correct_by_seed: dict seed -> array of 0/1 per question.

    The top level has only ten seeds, which bounds how tight these intervals can
    honestly be -- that limit is a property of the design, not of the estimator."""
    rng = rng or np.random.default_rng(0)
    seeds = list(correct_by_seed.keys())
    if not seeds:
        return (np.nan,) * 3
    means = np.empty(n_boot)
    for b in range(n_boot):
        chosen = rng.choice(seeds, size=len(seeds), replace=True)
        means[b] = np.mean([
            rng.choice(correct_by_seed[s], size=len(correct_by_seed[s]), replace=True).mean()
            for s in chosen])
    point = np.mean([correct_by_seed[s].mean() for s in seeds])
    return point, np.percentile(means, 2.5), np.percentile(means, 97.5)


def accuracy_tables(q, out):
    results = []
    group_cols = ["cond", "tech", "alpha", "split", "regime"]

    for keys, g in q.groupby(group_cols):
        correct_by_seed = {s: gg["correct"].to_numpy(float) for s, gg in g.groupby("seed")}
        point, lo, hi = two_level_bootstrap(correct_by_seed)
        per_seed = [v.mean() for v in correct_by_seed.values()]
        results.append(dict(zip(group_cols, keys)) | {
            "subset": "ALL", "n_seeds": len(correct_by_seed),
            "n_q": int(len(g) / max(len(correct_by_seed), 1)),
            "acc": round(point, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
            "acc_std_seeds": (round(float(np.std(per_seed, ddof=1)), 4)
                              if len(per_seed) > 1 else np.nan),
        })

    # per subset (A/B/C inside abc; the hard split carries the legacy code H)
    for keys, g in q.groupby(group_cols + ["subset_code"]):
        *main, subset_code = keys
        correct_by_seed = {s: gg["correct"].to_numpy(float) for s, gg in g.groupby("seed")}
        point, lo, hi = two_level_bootstrap(correct_by_seed, n_boot=4000)
        results.append(dict(zip(group_cols, main)) | {
            "subset": str(subset_code), "n_seeds": len(correct_by_seed),
            "n_q": int(len(g) / max(len(correct_by_seed), 1)),
            "acc": round(point, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
            "acc_std_seeds": np.nan,
        })

    df = pd.DataFrame(results).sort_values(["split", "regime", "subset", "acc"],
                                           ascending=[True, True, True, False])
    df.to_csv(out / "accuracy.csv", index=False)
    return df


# ---------------------------------------------------------
# 2. PAIRED TESTS
# ---------------------------------------------------------
def mcnemar_exact(a, b):
    """Exact McNemar via the binomial on discordant pairs. a, b: arrays of 0/1."""
    n01 = int(((a == 0) & (b == 1)).sum())   # b right, a wrong
    n10 = int(((a == 1) & (b == 0)).sum())   # a right, b wrong
    n = n01 + n10
    if n == 0:
        return 1.0, n01, n10
    return st.binomtest(min(n01, n10), n, 0.5).pvalue, n01, n10


def paired_tests(q, out, subset_focus="B"):
    """For every pair of conditions within the same (split, regime):
       - paired t across seeds, on the whole split AND on the focus subset
       - Wilcoxon signed-rank as sensitivity
       - exact McNemar per seed (item-level diagnostic)

    The gold sequence is ASSERTED identical before anything is compared.

    AUDIT FIX: earlier versions computed the paired t only on the whole split, so
    a subset-restricted effect (distill_sft over step_by_step on subset B under
    reasoning) was invisible here and had to be computed by a separate script.
    The focus-subset columns below close that gap."""
    if not HAS_SCIPY:
        return pd.DataFrame()

    rows = []
    for (split, regime), g in q.groupby(["split", "regime"]):
        conditions = sorted(g["cond"].unique())
        for c1, c2 in itertools.combinations(conditions, 2):
            g1, g2 = g[g["cond"] == c1], g[g["cond"] == c2]
            seeds = sorted(set(g1["seed"]) & set(g2["seed"]))
            if not seeds:
                continue

            acc1, acc2 = [], []
            acc1_focus, acc2_focus = [], []
            mcnemar_all, mcnemar_focus = [], []
            n01_focus = n10_focus = 0
            skipped_seeds = []

            for s in seeds:
                a = g1[g1["seed"] == s].sort_values("q_idx")
                b = g2[g2["seed"] == s].sort_values("q_idx")
                if len(a) != len(b):
                    skipped_seeds.append(s)
                    continue
                # GUARANTEE: same question order => same gold sequence
                if not (a["gold"].to_numpy() == b["gold"].to_numpy()).all():
                    raise AssertionError(
                        f"PAIRING BROKEN in {split}/{regime} {c1} vs {c2} seed {s}: "
                        "the gold sequences differ -- these runs do not share a "
                        "question order, so no paired test on them is valid.")
                av, bv = a["correct"].to_numpy(), b["correct"].to_numpy()
                acc1.append(av.mean())
                acc2.append(bv.mean())
                mcnemar_all.append(mcnemar_exact(av, bv)[0])

                focus = a["subset_code"].astype(str).str.upper().to_numpy() == subset_focus
                if focus.sum() > 0:
                    acc1_focus.append(av[focus].mean())
                    acc2_focus.append(bv[focus].mean())
                    p_focus, i01, i10 = mcnemar_exact(av[focus], bv[focus])
                    mcnemar_focus.append(p_focus)
                    n01_focus += i01
                    n10_focus += i10

            if skipped_seeds:
                print(f"[pairing] {c1} vs {c2} ({split}/{regime}): dropped seeds "
                      f"{skipped_seeds} for differing length")
            if len(acc1) < 2:
                continue

            t_all = st.ttest_rel(acc2, acc1)              # positive => c2 > c1
            w_all = st.wilcoxon(acc2, acc1) if len(acc1) >= 3 else None
            row = {
                "split": split, "regime": regime,
                "cond_1": c1, "cond_2": c2, "n_seeds": len(acc1),
                "acc_1": round(float(np.mean(acc1)), 4),
                "acc_2": round(float(np.mean(acc2)), 4),
                "delta_2_minus_1": round(float(np.mean(acc2) - np.mean(acc1)), 4),
                "paired_t_p": round(float(t_all.pvalue), 4),
                "wilcoxon_p": round(float(w_all.pvalue), 4) if w_all is not None else np.nan,
                "seeds_2_wins": int(sum(y > x for x, y in zip(acc1, acc2))),
            }
            if len(acc1_focus) >= 2:
                t_focus = st.ttest_rel(acc2_focus, acc1_focus)
                w_focus = st.wilcoxon(acc2_focus, acc1_focus) if len(acc1_focus) >= 3 else None
                row |= {
                    f"acc_1_{subset_focus}": round(float(np.mean(acc1_focus)), 4),
                    f"acc_2_{subset_focus}": round(float(np.mean(acc2_focus)), 4),
                    f"delta_{subset_focus}": round(
                        float(np.mean(acc2_focus) - np.mean(acc1_focus)), 4),
                    f"paired_t_p_{subset_focus}": round(float(t_focus.pvalue), 4),
                    f"wilcoxon_p_{subset_focus}": (round(float(w_focus.pvalue), 4)
                                                   if w_focus is not None else np.nan),
                    f"seeds_2_wins_{subset_focus}": int(
                        sum(y > x for x, y in zip(acc1_focus, acc2_focus))),
                }
            # DESCRIPTIVE ONLY -- the minimum across seeds, never a global test
            row |= {
                "mcnemar_p_min_all_DESCRIPTIVE": (round(float(np.min(mcnemar_all)), 4)
                                                  if mcnemar_all else np.nan),
                f"mcnemar_p_min_{subset_focus}_DESCRIPTIVE": (
                    round(float(np.min(mcnemar_focus)), 4) if mcnemar_focus else np.nan),
                f"discord_{subset_focus}_2_wins": n01_focus,
                f"discord_{subset_focus}_1_wins": n10_focus,
            }
            rows.append(row)

    df = pd.DataFrame(rows).sort_values(["split", "regime", "paired_t_p"])
    df.to_csv(out / "paired_tests.csv", index=False)
    return df


# ---------------------------------------------------------
# 3. QUESTION-LEVEL TRADE-OFF VS BASELINE
# ---------------------------------------------------------
def tradeoff_vs_baseline(q, out, baseline="pure_sft", regime_base="answer_only"):
    """Questions gained/lost per subset against the baseline, paired per question
    and averaged across seeds. The aggregate accuracy hides this entirely."""
    rows = []
    base = q[(q["cond"] == baseline) & (q["regime"] == regime_base)]
    if base.empty:
        print(f"[tradeoff] baseline {baseline}/{regime_base} absent -- skipping")
        return pd.DataFrame()

    for (split, cond, regime), g in q.groupby(["split", "cond", "regime"]):
        if cond == baseline:
            continue
        deltas_by_subset = {}
        seeds_ok = 0
        for s, gg in g.groupby("seed"):
            bb = base[(base["split"] == split) & (base["seed"] == s)].sort_values("q_idx")
            gg = gg.sort_values("q_idx")
            if len(bb) != len(gg) or len(bb) == 0:
                continue
            if not (bb["gold"].to_numpy() == gg["gold"].to_numpy()).all():
                raise AssertionError(f"PAIRING BROKEN in the trade-off: {cond} seed {s}")
            seeds_ok += 1
            for subset_code in sorted(gg["subset_code"].astype(str).unique()):
                m = gg["subset_code"].astype(str) == subset_code
                gain = int((gg["correct"].to_numpy()[m] - bb["correct"].to_numpy()[m]).sum())
                deltas_by_subset.setdefault(subset_code, []).append(gain)
        for subset_code, deltas in sorted(deltas_by_subset.items()):
            rows.append({"split": split, "cond": cond, "regime": regime,
                         "vs": f"{baseline}/{regime_base}", "subset": subset_code,
                         "n_seeds": seeds_ok,
                         "delta_questions_mean": round(float(np.mean(deltas)), 2),
                         "delta_questions_min": int(np.min(deltas)),
                         "delta_questions_max": int(np.max(deltas))})
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(out / "tradeoff_questions_vs_baseline.csv", index=False)
    return df


# ---------------------------------------------------------
# 4. COMPUTE / EFFICIENCY
# ---------------------------------------------------------
def compute_efficiency(q, timing, out):
    """Accuracy joined to cost. The accuracy-cost frontier is the efficiency
    argument of the paper: reasoning buys accuracy the direct mode cannot reach,
    at a latency that only makes sense if you route selectively."""
    per_seed = (q.groupby(["cond", "tech", "alpha", "split", "regime", "seed"])
                  .agg(acc=("correct", "mean"),
                       new_tokens_med=("n_new_tokens", "median"),
                       new_tokens_mean=("n_new_tokens", "mean"),
                       reasoning_chars=("reasoning_chars", "mean"))
                  .reset_index())
    agg = (per_seed.groupby(["cond", "tech", "alpha", "split", "regime"])
                   .agg(n_seeds=("seed", "nunique"), acc=("acc", "mean"),
                        acc_std=("acc", lambda x: x.std(ddof=1) if len(x) > 1 else np.nan),
                        new_tokens_med=("new_tokens_med", "mean"),
                        reasoning_chars=("reasoning_chars", "mean"))
                   .reset_index())
    if not timing.empty:
        tagg = (timing.groupby(["cond", "split", "regime"])
                      .agg(tokens_per_sec=("tokens_per_sec", "mean"),
                           secs_per_question=("secs_per_question", "mean"),
                           gen_secs_total_mean=("gen_secs_total", "mean"))
                      .reset_index())
        agg = agg.merge(tagg, on=["cond", "split", "regime"], how="left")
        agg["secs_per_correct"] = (agg["secs_per_question"] / agg["acc"]).round(4)
    agg = agg.sort_values(["split", "regime", "acc"], ascending=[True, True, False]).round(4)
    agg.to_csv(out / "compute_efficiency.csv", index=False)
    return agg


# ---------------------------------------------------------
# 5. TRAINING
# ---------------------------------------------------------
def training_summary(train_df, curves, out):
    if train_df.empty:
        return pd.DataFrame()
    agg = (train_df.groupby(["cond", "tech", "alpha"])
                   .agg(n_seeds=("seed", "nunique"),
                        train_loss_mean=("train_loss", "mean"),
                        train_loss_std=("train_loss",
                                        lambda x: x.std(ddof=1) if len(x) > 1 else np.nan),
                        steps=("steps", "mean"))
                   .reset_index().round(4))
    agg.to_csv(out / "training_summary.csv", index=False)
    if not curves.empty:
        curves = curves.copy()
        curves["epoch_bin"] = curves["epoch"].round(1)
        (curves.groupby(["cond", "epoch_bin"])["loss"].mean().reset_index()
               .to_csv(out / "loss_curve_by_condition.csv", index=False))
    return agg


# ---------------------------------------------------------
# 6. CONFOUNDERS
# ---------------------------------------------------------
def confounder_checks(q, out):
    """The three known contaminants: truncated at the ceiling, reasoned in English
    (fell back to the native mode), emitted no answer at all. If any is high the
    accuracy is contaminated and must not be read at face value."""
    qq = q.copy()
    qq["emitted"] = qq["pred"].notna()
    check = (qq.groupby(["cond", "split", "regime"])
               .agg(n=("correct", "size"),
                    truncated_pct=("truncated", "mean"),
                    reasoning_en_pct=("reasoning_en", "mean"),
                    emitted_pct=("emitted", "mean"))
               .reset_index().round(4))
    check["ALERT"] = ""
    check.loc[check["truncated_pct"] > 0.10, "ALERT"] += "TRUNCATION>10% "
    check.loc[(check["regime"] == "reasoning") & (check["reasoning_en_pct"] > 0.15),
              "ALERT"] += "ENGLISH>15% "
    check.loc[check["emitted_pct"] < 0.95, "ALERT"] += "EMISSION<95% "
    check.to_csv(out / "confounder_checks.csv", index=False)
    return check


# ---------------------------------------------------------
# 7. DIAGNOSTIC FIGURES (the paper's figures live in figures/make_paper_figures.py)
# ---------------------------------------------------------
def make_diagnostic_figures(acc_df, eff_df, curves, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[figures] matplotlib unavailable -- skipping")
        return

    palette = [ORANGE, GOLD, INK] + EXTRA
    plt.rcParams.update({"axes.edgecolor": INK, "text.color": INK,
                         "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})

    b = acc_df[(acc_df["subset"] == "B") & (acc_df["split"] == "abc")]
    if not b.empty:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        y = np.arange(len(b))
        ax.barh(y, b["acc"], xerr=[b["acc"] - b["ci_lo"], b["ci_hi"] - b["acc"]],
                color=[palette[i % len(palette)] for i in range(len(b))],
                edgecolor=INK, capsize=3)
        ax.set_yticks(y)
        ax.set_yticklabels((b["cond"] + " / " + b["regime"]).tolist(), fontsize=8)
        ax.set_xlabel("Accuracy on subset B (95% two-level bootstrap CI)")
        ax.set_title("Subset B -- the thermometer")
        fig.tight_layout()
        fig.savefig(out / "fig_subset_b.png", dpi=150)
        plt.close(fig)

    e = eff_df[eff_df["split"] == "abc"]
    if not e.empty and "new_tokens_med" in e:
        fig, ax = plt.subplots(figsize=(7, 5))
        for i, (_, r) in enumerate(e.iterrows()):
            ax.scatter(r["new_tokens_med"], r["acc"], s=90,
                       color=palette[i % len(palette)], edgecolor=INK, zorder=3)
            ax.annotate(f'{r["cond"]}/{r["regime"][:4]}', (r["new_tokens_med"], r["acc"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=7)
        ax.set_xlabel("Generated tokens (median per question)")
        ax.set_ylabel("Accuracy (ABC test)")
        ax.set_title("Accuracy versus generation cost")
        fig.tight_layout()
        fig.savefig(out / "fig_cost_frontier.png", dpi=150)
        plt.close(fig)

    if not curves.empty:
        lc = pd.read_csv(out / "loss_curve_by_condition.csv")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for i, (cond, g) in enumerate(lc.groupby("cond")):
            ax.plot(g["epoch_bin"], g["loss"], label=cond,
                    color=palette[i % len(palette)], lw=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training loss (mean across seeds)")
        ax.legend(fontsize=7)
        ax.set_title("Convergence by condition")
        fig.tight_layout()
        fig.savefig(out / "fig_loss.png", dpi=150)
        plt.close(fig)

    print(f"[figures] written to {out}/fig_*.png")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="outputs_stage2")
    ap.add_argument("--out", default="analysis_stage2")
    ap.add_argument("--subset-focus", default="B",
                    help="thermometer subset for the focused tests (default B)")
    ap.add_argument("--baseline", default="pure_sft")
    ap.add_argument("--include-incomplete", action="store_true",
                    help="include runs without a DONE marker -- for peeking only, "
                         "never for a final table")
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()

    root, out = Path(args.root), Path(args.out)
    out.mkdir(exist_ok=True)

    runs = discover_runs(root, require_done=not args.include_incomplete)
    if not runs:
        raise SystemExit(f"No runs found in {root}/ (with DONE). "
                         "Use --include-incomplete to peek at partial runs.")
    print(f"Runs: {len(runs)} | conditions: {sorted({condition_label(r) for r in runs})} "
          f"| seeds: {sorted({r['seed'] for r in runs})}")
    check_ceilings(runs)
    print()

    q = load_question_level(runs)
    timing = load_timing(runs)
    train_df, curves = load_training(runs)

    print("=" * 78)
    print("1) ACCURACY (seed means, two-level bootstrap CI) -> accuracy.csv")
    print("=" * 78)
    acc_df = accuracy_tables(q, out)
    with pd.option_context("display.width", 160, "display.max_rows", 100):
        print(acc_df[acc_df["subset"].isin(["ALL", "A", "B", "C"])].to_string(index=False))

    print("\n" + "=" * 78)
    print(f"2) PAIRED TESTS (t across seeds; whole split AND subset "
          f"{args.subset_focus}) -> paired_tests.csv")
    print("=" * 78)
    pt = paired_tests(q, out, subset_focus=args.subset_focus)
    if not pt.empty:
        with pd.option_context("display.width", 240, "display.max_rows", 100):
            print(pt.to_string(index=False))

    print("\n" + "=" * 78)
    print(f"3) QUESTION TRADE-OFF vs {args.baseline} -> tradeoff_questions_vs_baseline.csv")
    print("=" * 78)
    td = tradeoff_vs_baseline(q, out, baseline=args.baseline)
    if not td.empty:
        with pd.option_context("display.width", 160, "display.max_rows", 100):
            print(td.to_string(index=False))

    print("\n" + "=" * 78)
    print("4) EFFICIENCY (accuracy x tokens x time) -> compute_efficiency.csv")
    print("=" * 78)
    eff = compute_efficiency(q, timing, out)
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(eff.to_string(index=False))

    print("\n" + "=" * 78)
    print("5) TRAINING -> training_summary.csv / loss_curve_by_condition.csv")
    print("=" * 78)
    tr = training_summary(train_df, curves, out)
    if not tr.empty:
        print(tr.to_string(index=False))

    print("\n" + "=" * 78)
    print("6) CONFOUNDERS -> confounder_checks.csv")
    print("=" * 78)
    check = confounder_checks(q, out)
    with pd.option_context("display.width", 160, "display.max_rows", 100):
        print(check.to_string(index=False))
    alerts = check[check["ALERT"] != ""]
    if not alerts.empty:
        print("\n!! CONTAMINATION ALERTS (read these accuracies with caution):")
        print(alerts.to_string(index=False))
    else:
        print("\nNo contamination alerts: every condition is within bounds.")

    if args.figures:
        make_diagnostic_figures(acc_df, eff, curves, out)

    print(f"\nEverything written to {out}/ -- no table was typed from memory; "
          "all of it came from the run files.")


if __name__ == "__main__":
    main()

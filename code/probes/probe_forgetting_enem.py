#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Catastrophic-forgetting probe: how much GENERAL knowledge does fine-tuning cost?

Medical fine-tuning can buy in-domain accuracy by eroding everything else. This
probe measures that directly, on ENEM (the Brazilian national secondary-school
examination, maritaca-ai/enem) -- a domain the models were never trained on here.

It evaluates the untrained Qwen3-4B as the reference, then every saved LoRA
adapter from Stage 2 (3 techniques x 3 alphas x 10 seeds). If the rationale-based
techniques lose LESS than pure SFT, that replicates the RaDis finding: training on
rationales preserves general ability that answer-only fine-tuning destroys.

Two design points that decide whether the number means anything:

  * Inference uses the DIRECT-ANSWER format, the same one the medical fine-tuning
    used. Probing in a format the adapters never saw would confound forgetting
    with format mismatch.
  * The prompt is built with enable_thinking=True, consistent with Stage-2
    training and inference after the masking fix. An earlier version of this
    probe used False, which makes the Qwen3 template inject an empty
    <think></think> -- a shape these adapters never encountered in training.

Sampling is stratified by ENEM subject area and cached to a parquet, so every
adapter is scored on exactly the same questions. Items that carry an image are
dropped: the models see text only.

Runs are resumable -- an evaluation whose CSV already exists is skipped unless
--force is given, so a interrupted sweep picks up where it stopped.

Usage
  python probe_forgetting_enem.py --n 500                  # base + every adapter
  python probe_forgetting_enem.py --smoke                  # base + 2 adapters
  python probe_forgetting_enem.py --only distill_sft       # filter by condition
"""
import argparse
import gc
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import transformers
transformers.logging.set_verbosity_error()
from unsloth import FastLanguageModel

MODEL_NAME = "unsloth/Qwen3-4B"
STAGE2_ROOT = Path("outputs_stage2")
OUT_DIR = Path("outputs_forgetting")
MAX_SEQ_LEN = 2048
SAMPLE_SEED = 42

ANSWER_RE = re.compile(r"Resposta\s*[:\-]?\s*\**\s*([A-E])", re.I)
SEED_RE = re.compile(r"_seed(\d+)")


def log(m):
    print(f">>> {m}", flush=True)


# ---------------------------------------------------------
# 1. THE ENEM SAMPLE
# ---------------------------------------------------------
def load_enem(n_total, seed=SAMPLE_SEED):
    """Stratified sample, cached. Every adapter must be scored on the SAME items,
    otherwise the deltas compare models across different question sets."""
    cache = OUT_DIR / "enem_sample.parquet"
    if cache.exists():
        out = pd.read_parquet(cache)
        log(f"reusing the existing ENEM sample: {len(out)} questions ({cache})")
        return out

    from datasets import load_dataset, get_dataset_config_names

    log("downloading maritaca-ai/enem ...")
    try:
        configs = get_dataset_config_names("maritaca-ai/enem")
        log(f"configs: {configs}")
    except Exception:
        configs = []

    frames = []
    for c in configs:
        try:
            ds = load_dataset("maritaca-ai/enem", c, split="train")
            d = ds.to_pandas()
            d["config"] = c
            frames.append(d)
            log(f"  {c}: {len(d)} questions")
        except Exception as e:
            log(f"  {c}: failed ({e})")
    if not frames:
        frames = [load_dataset("maritaca-ai/enem", split="train").to_pandas()]

    raw = pd.concat(frames, ignore_index=True)
    log(f"raw total: {len(raw)} | columns: {list(raw.columns)}")

    # the published schema has shifted between releases, so map columns by name
    col_q = next((c for c in raw.columns if c.lower() in
                  ("question", "questao", "enunciado", "pergunta", "text")), None)
    col_a = next((c for c in raw.columns if c.lower() in
                  ("alternatives", "options", "alternativas", "choices")), None)
    col_l = next((c for c in raw.columns if c.lower() in
                  ("label", "answer", "gabarito", "resposta", "correct")), None)
    col_area = next((c for c in raw.columns if any(k in c.lower() for k in
                    ("area", "disciplin", "subject", "categor", "domain", "config"))), None)

    if not (col_q and col_a and col_l):
        raise RuntimeError(f"could not map the columns. Available: {list(raw.columns)}\n"
                           f"example row: {raw.iloc[0].to_dict()}")
    log(f"columns -> question={col_q} | options={col_a} | answer={col_l} | area={col_area}")

    # drop items that depend on a figure: the model sees text only
    for c in raw.columns:
        if "image" in c.lower() or "figure" in c.lower():
            try:
                keep = raw[c].isna() | (raw[c].astype(str).str.strip()
                                        .isin(["", "[]", "None", "False"]))
                if 0 < keep.sum() < len(raw):
                    raw = raw[keep].copy()
                    log(f"dropped image items via '{c}' -> {len(raw)} remain")
            except Exception:
                pass

    if col_area and raw[col_area].notna().any():
        parts = []
        for area, p in raw[col_area].value_counts(normalize=True).items():
            pool = raw[raw[col_area] == area]
            k = min(int(round(n_total * p)), len(pool))
            if k > 0:
                parts.append(pool.sample(n=k, random_state=seed))
        sample = pd.concat(parts, ignore_index=True)
    else:
        sample = raw.sample(n=min(n_total, len(raw)), random_state=seed)

    if len(sample) > n_total:
        sample = sample.sample(n=n_total, random_state=seed)
    sample = sample.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    out = pd.DataFrame({
        "stem": sample[col_q].astype(str),
        "options_raw": sample[col_a],
        "gold": sample[col_l].astype(str).str.strip().str.upper().str[0],
        "area": sample[col_area].astype(str) if col_area else "?",
    })
    out = out[out["gold"].isin(list("ABCDE"))].reset_index(drop=True)
    log(f"final sample: {len(out)} questions")
    if col_area:
        log(f"by area: {out['area'].value_counts().to_dict()}")
    OUT_DIR.mkdir(exist_ok=True)
    out.to_parquet(cache, index=False)
    return out


def format_options(options):
    """ENEM ships options as a list, a dict, or the repr of either."""
    if isinstance(options, dict):
        return "\n".join(f"{k.strip().upper()}) {v}" for k, v in sorted(options.items()))
    if isinstance(options, (list, np.ndarray)):
        letters = "ABCDE"
        return "\n".join(f"{letters[i]}) {a}" for i, a in enumerate(options[:5]))
    text = str(options)
    try:
        import ast
        return format_options(ast.literal_eval(text))
    except Exception:
        return text


def build_prompt(row, tokenizer):
    user = (f"Questão:\n{row['stem']}\n\n"
            f"Alternativas:\n{format_options(row['options_raw'])}\n\n"
            "Responda apenas com 'Resposta: X' (X = A-E). Não explique.")
    # enable_thinking=True keeps the prompt identical in shape to Stage-2 training
    # and inference; see the module docstring for why this matters.
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True)


# ---------------------------------------------------------
# 2. INFERENCE
# ---------------------------------------------------------
@torch.no_grad()
def evaluate(model, tokenizer, df, batch_size=32, max_new_tokens=64):
    FastLanguageModel.for_inference(model)
    tokenizer.padding_side = "left"
    prompts = [build_prompt(r, tokenizer) for _, r in df.iterrows()]
    generations = []
    n_batches = (len(prompts) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(prompts), batch_size), total=n_batches,
                  desc="    enem", ncols=80, leave=False):
        batch = prompts[i:i + batch_size]
        encoded = tokenizer(batch, return_tensors="pt", padding=True,
                            truncation=True, max_length=MAX_SEQ_LEN).to(model.device)
        out = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False,
                             temperature=None, top_p=None,
                             pad_token_id=tokenizer.pad_token_id)
        for j in range(len(batch)):
            new = out[j][encoded["input_ids"].shape[1]:]
            generations.append(tokenizer.decode(new, skip_special_tokens=True))

    # last match, consistent with the rest of the pipeline
    preds = []
    for g in generations:
        matches = ANSWER_RE.findall(str(g))
        preds.append(matches[-1].upper() if matches else None)

    res = pd.DataFrame({"gold": df["gold"].values, "pred": preds,
                        "area": df["area"].values, "gen": generations})
    res["correct"] = (res["pred"] == res["gold"]).astype(int)
    return res


def evaluate_and_save(model_path, run_name, enem, batch_size, force=False):
    """Score one model (base or adapter), skipping work already on disk."""
    out_csv = OUT_DIR / f"enem_{run_name}.csv"
    if out_csv.exists() and not force:
        acc = pd.read_csv(out_csv)["correct"].mean()
        log(f"[SKIP] {run_name} (already scored: acc={acc:.3f})")
        return float(acc)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path, max_seq_length=MAX_SEQ_LEN, dtype=None, load_in_4bit=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    res = evaluate(model, tokenizer, enem, batch_size)
    res.to_csv(out_csv, index=False)
    acc = res["correct"].mean()
    log(f"{run_name}: acc={acc:.3f} | emitted={res['pred'].notna().sum()}/{len(res)}")
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return float(acc)


# ---------------------------------------------------------
# 3. MAIN
# ---------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--only", default=None, help="substring filter, e.g. distill_sft")
    ap.add_argument("--smoke", action="store_true", help="base + 2 adapters, for a quick check")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    enem = load_enem(args.n)

    adapters = sorted(d for d in STAGE2_ROOT.iterdir()
                      if d.is_dir() and (d / "adapter").exists() and (d / "DONE").exists())
    if args.only:
        adapters = [d for d in adapters if args.only in d.name]
    if args.smoke:
        adapters = adapters[:2]
    log(f"adapters to score: {len(adapters)} | plus the untrained base")

    per_run = {"base": evaluate_and_save(MODEL_NAME, "base", enem,
                                         args.batch_size, args.force)}

    for d in adapters:
        log("=" * 60)
        log(d.name)
        log("=" * 60)
        try:
            per_run[d.name] = evaluate_and_save(str(d / "adapter"), d.name, enem,
                                                args.batch_size, args.force)
        except Exception as e:
            # one bad adapter should not abandon a sweep of seventy
            log(f"[ERROR] {d.name}: {e} -- continuing")

    # summarise by CONDITION (the run name minus _seedNN) -- never by substring
    rows = []
    for name, acc in per_run.items():
        if name == "base" or acc is None:
            continue
        m = SEED_RE.search(name)
        rows.append({"cond": SEED_RE.sub("", name),
                     "seed": int(m.group(1)) if m else -1, "acc": acc})
    df = pd.DataFrame(rows)
    base = per_run.get("base", np.nan)

    print()
    print("=" * 74)
    print(f"FORGETTING PROBE -- ENEM | base acc = {base:.4f} (n={len(enem)} questions)")
    print("=" * 74)
    if not df.empty:
        summary = (df.groupby("cond")
                     .agg(n_seeds=("seed", "nunique"), acc_mean=("acc", "mean"),
                          acc_std=("acc", lambda x: x.std(ddof=1) if len(x) > 1 else np.nan),
                          acc_min=("acc", "min"), acc_max=("acc", "max"))
                     .reset_index())
        summary["delta_vs_base"] = (summary["acc_mean"] - base).round(4)
        summary = summary.sort_values("delta_vs_base", ascending=False).round(4)
        print(summary.to_string(index=False))
        summary.to_csv(OUT_DIR / "forgetting_by_condition.csv", index=False)
    df.to_csv(OUT_DIR / "forgetting_by_run.csv", index=False)
    json.dump({k: (float(v) if v is not None else None) for k, v in per_run.items()},
              open(OUT_DIR / "forgetting_summary.json", "w"), indent=2)

    print(f"\nwritten to {OUT_DIR}/")
    print("Reading it: a NEGATIVE delta is forgetting. If the rationale-based "
          "techniques fall less than pure SFT, that replicates RaDis -- rationale "
          "supervision preserves general ability that answer-only training erodes.")


if __name__ == "__main__":
    main()

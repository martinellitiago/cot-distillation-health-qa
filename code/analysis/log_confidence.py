#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Log the direct-answer confidence of each Stage-2 run, for the cascade router.

The routing analyses in `paired_routing.py` use gold-derived difficulty, so they
measure the CEILING a perfect router could reach. This script produces the signal
a REAL router could use: the model's own confidence, available at inference time
with no gold and no reasoning pass.

Method: load each saved adapter and, for every test question, run ONE forward
pass over the direct-answer prompt with "Resposta:" already appended, so the very
next token is the option letter. Read the logits at that position, restrict them
to the five option letters, softmax over those five, and take the maximum. That
is a well-defined multiple-choice self-confidence, and it costs one forward pass
rather than three hundred generated tokens.

Writes conf_{split}.csv per run, in the SAME ROW ORDER as the existing
infer_{split}_answer_only.csv (both read the same per-seed pkl), so
`cascade_eval.py` can pair confidence with the reasoning-mode correctness that
already exists without re-running anything.

Run this before cascade_eval.py.

Usage
  for tech in distill_sft step_by_step pure_sft; do
    for asuf in "" "_a01" "_a03"; do
      python log_confidence.py --technique $tech --alpha-suffix "$asuf"
    done
  done
(pure_sft has no alpha variants; run it once with an empty suffix)
"""
import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
transformers.logging.set_verbosity_error()
from unsloth import FastLanguageModel

SPLITS_DIR = Path("data/splits_stage2")
OUT_ROOT = Path("outputs_stage2")
MAX_SEQ_LEN = 2048
SEEDS = [8, 12, 17, 23, 25, 31, 37, 44, 52, 61]
LETTERS = ["A", "B", "C", "D", "E"]
LEGACY_DIR_NAMES = {"pure_sft": "sft_puro"}


def format_options(options):
    import ast
    if isinstance(options, str):
        options = ast.literal_eval(options)
    return "\n".join(f"{k.strip().upper()}) {v}" for k, v in sorted(options.items()))


def direct_prompt(tokenizer, stem, options_fmt):
    """The direct-answer prompt from stage2_run.py, with the answer preamble forced
    so that the next token position holds the option letter."""
    user = (f"Questão:\n{stem}\n\nAlternativas:\n{options_fmt}\n\n"
            "Responda apenas com 'Resposta: X' (X = A-E). Não explique.")
    p = tokenizer.apply_chat_template([{"role": "user", "content": user}],
                                      tokenize=False, add_generation_prompt=True,
                                      enable_thinking=True)
    return p + "Resposta:"


def letter_token_ids(tokenizer):
    """Token id for ' A' ... ' E' -- the leading space gives the mid-sequence form,
    which is what actually follows 'Resposta:'."""
    return [tokenizer(" " + letter, add_special_tokens=False)["input_ids"][-1]
            for letter in LETTERS]


@torch.no_grad()
def confidences(model, tokenizer, df, letter_ids, batch_size=16):
    FastLanguageModel.for_inference(model)
    tokenizer.padding_side = "left"
    prompts = [direct_prompt(tokenizer, r["enunciado"], format_options(r["alternativas"]))
               for _, r in df.iterrows()]
    preds, confs = [], []
    ids = torch.tensor(letter_ids, device=model.device)
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        encoded = tokenizer(batch, return_tensors="pt", padding=True,
                            truncation=True, max_length=MAX_SEQ_LEN).to(model.device)
        out = model(**encoded)
        # left padding means the last position is the real next-token slot
        last = out.logits[:, -1, :]
        option_logits = last[:, ids]                          # [B, 5]
        probs = torch.softmax(option_logits.float(), dim=-1)  # normalised over A-E only
        p, idx = probs.max(dim=-1)
        preds.extend(LETTERS[j] for j in idx.tolist())
        confs.extend(p.tolist())
    return preds, confs


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
    ap.add_argument("--technique", required=True)
    ap.add_argument("--alpha-suffix", default="")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    for seed in SEEDS:
        run = resolve_run(OUT_ROOT, args.technique, seed, args.epochs, args.alpha_suffix)
        adapter = run / "adapter"
        if not adapter.exists():
            print(f"[skip] no adapter: {adapter}")
            continue

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(adapter), max_seq_length=MAX_SEQ_LEN,
            dtype=None, load_in_4bit=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        ids = letter_token_ids(tokenizer)

        for split in ("abc", "hard"):
            df = pd.read_pickle(SPLITS_DIR / f"df_test_{split}_seed{seed}.pkl")
            preds, confs = confidences(model, tokenizer, df, ids, args.batch_size)
            gold = df["resposta"].astype(str).str.strip().str.upper().values
            res = pd.DataFrame({
                "gold": gold,
                "pred_conf": preds,
                "conf": np.round(confs, 5),
                "correct_conf": (np.array(preds) == gold).astype(int),
                "subset_code": (df["subset_code"].astype(str).values
                                if "subset_code" in df else "?"),
            })
            res.to_csv(run / f"conf_{split}.csv", index=False)
            # sanity: this accuracy should sit close to the generated direct-answer
            # accuracy. If it does not, the letter token ids are wrong.
            print(f"{run.name} {split}: conf-pred acc={res['correct_conf'].mean():.3f} "
                  f"mean_conf={res['conf'].mean():.3f}")

        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    print("done. conf_{abc,hard}.csv written per run.")


if __name__ == "__main__":
    main()

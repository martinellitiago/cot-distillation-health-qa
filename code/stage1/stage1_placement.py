#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 1 -- what drives reasoning transfer: rationale content or format?

Trains a Qwen3-4B student with LoRA on teacher rationales under five target
layouts and evaluates transfer on subset B (teacher right, untrained student
wrong). The comparison isolates two factors:

  content   coherent rationale vs. a cross-question DERANGEMENT (each item gets
            a fluent, same-length, same-style rationale belonging to a DIFFERENT
            question). Only relevance is destroyed, so a drop is attributable to
            content rather than to length or style.
  format    rationale inside the model's native <think> span vs. as plain body
            text vs. after the label (Wadhwa et al., EMNLP 2024).

Modes
  base            no training; inference-only reference
  correct         <think>rationale</think> + "Resposta: X"   (tagged)
  correct_notag   rationale as body text  + "Resposta: X"    (tag-free)
  correct_after   "Resposta: X" + <think>rationale</think>   (post-label)
  shuffle         deranged rationale, tagged                 (control for `correct`)
  shuffle_notag   deranged rationale, tag-free               (control for `correct_notag`)
  none            answer-only target

Guarantees enforced at runtime (do not remove):
  * completion-only masking with an EXACT-PREFIX assert  -> "MASKING BROKEN"
  * true derangement, no item receives its own rationale -> "DERANGEMENT BROKEN"

Reproduces the paper's Stage-1 numbers with
  python stage1_placement.py \
    --think-modes base correct correct_notag correct_after shuffle shuffle_notag \
    --subset AB --eval-subset B \
    --seeds 8 12 17 23 25 31 37 44 52 61 \
    --epochs 2 --token-size 250 \
    --max-seq-length 2048 --max-new-tokens 768 \
    --inference-prompt-style reasoning \
    --train-only-rationale-examples \
    --lora-random-state 3407
"""

import os
import gc
import re
import ast
import math
import json
import argparse
import random
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import Trainer
from trl import SFTConfig
from unsloth import FastLanguageModel


MODEL_NAME = "unsloth/Qwen3-4B"
MODEL_SAVE_ROOT = "./models_stage1"
BASE_OUTPUT_DIR = "./outputs_stage1"
RATIONALE_DATA_DIR = "./distill_thinks"
MAX_SEQ_LENGTH = 2048
THINK_COL = "think_prof_trunc"

TRAINED_MODES = {"correct", "shuffle", "none", "correct_after", "correct_notag", "shuffle_notag"}
INFERENCE_ONLY_MODES = {"base"}
ALL_MODES = sorted(TRAINED_MODES | INFERENCE_ONLY_MODES)

# Modes whose target carries a rationale. Every one of these must appear in the
# skip rule and the position map below; a mode missing from either is silently
# mislabelled, which is how a condition ends up measuring something else.
RATIONALE_MODES = {"correct", "shuffle", "correct_after", "correct_notag", "shuffle_notag"}
DERANGED_MODES = {"shuffle", "shuffle_notag"}
POSITION_BY_MODE = {"correct_after": "after", "correct_notag": "notag", "shuffle_notag": "notag"}


def log(m):
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============ prompt / target builders ============
def normalize_options(options):
    """Answer options are stored as a dict (or its repr); render them as A) ... E)."""
    if isinstance(options, str):
        options = ast.literal_eval(options)
    if not isinstance(options, dict):
        raise ValueError(f"options must be a dict, got: {type(options)}")
    return "\n".join(f"{str(k).strip().upper()}) {v}" for k, v in sorted(options.items()))


def clean_reasoning_text(text):
    if text is None:
        return ""
    if isinstance(text, float) and math.isnan(text):
        return ""
    text = str(text).strip()
    if not text:
        return ""
    return re.sub(r"(?i)</?think>", "", text).strip()


def smart_truncate_think(body, max_tokens=250):
    """Keep the LAST `max_tokens` whitespace words -- the decision usually sits at
    the tail -- then trim to a sentence boundary.

    AUDIT FIX: the sentence-boundary trim could return "" when the truncation
    window held a single trailing period, silently demoting a rationale item to
    answer-only (or, under --train-only-rationale-examples, dropping it from
    training). Measured over the 4260 archived teacher rationales: 0 collapses at
    max_tokens=250 (the published setting), 1 at 150, 5 at 100, 36 at 50. The
    guard below makes the failure impossible rather than merely improbable."""
    original = body.strip()
    body = re.sub(r"\*\*\s*$", "", body).strip()
    tokens = body.split()
    if len(tokens) > max_tokens:
        body = " ".join(tokens[-max_tokens:])
        first_period = body.find(".")
        if first_period != -1:
            trimmed = body[first_period + 1:].strip()
            if trimmed:                     # only accept the trim if it leaves text
                body = trimmed

    if body and body[-1] not in ".!?":
        last_period = body.rfind(".")
        if last_period > len(body) * 0.5:
            body = body[: last_period + 1]
        else:
            body = body.rstrip(" *") + "."

    body = body.strip()
    if not body and original:
        log("  [warn] truncation emptied a non-empty rationale; keeping the tail verbatim")
        body = " ".join(original.split()[-max_tokens:]).strip()
    return body


def get_reasoning_body(row, think_col=THINK_COL, max_tokens=250):
    body = clean_reasoning_text(row.get(think_col, ""))
    if not body:
        return ""
    return smart_truncate_think(body, max_tokens=max_tokens)


def build_think_block(row, think_col=THINK_COL, max_tokens=250):
    body = get_reasoning_body(row, think_col=think_col, max_tokens=max_tokens)
    return f"<think>\n{body}\n</think>" if body else ""


def build_user_content(row, prompt_style="answer_only", include_rationale=False, max_think_tokens=250):
    """The prompt is Portuguese because the examinations and the distilled
    rationales are Portuguese, and keeping the student in-language is the point of
    the study. These strings are data, not UI text, and are left untranslated."""
    base = (
        f"Questão:\n{row['enunciado']}\n\n"
        f"Alternativas:\n{normalize_options(row['alternativas'])}\n\n"
    )

    if include_rationale:
        rationale = get_reasoning_body(row, max_tokens=max_think_tokens)
        if rationale:
            base += f"Raciocínio de apoio do professor:\n{rationale}\n\n"

    if prompt_style == "answer_only":
        return base + "Responda apenas com 'Resposta: X'."

    if prompt_style == "reasoning":
        return base + (
            "Raciocine de forma breve e objetiva antes de responder. "
            "Finalize obrigatoriamente com 'Resposta: X'."
        )

    if prompt_style == "rationale_prompt":
        rationale = get_reasoning_body(row, max_tokens=max_think_tokens)
        if rationale and "Raciocínio de apoio do professor:" not in base:
            base += f"Raciocínio de apoio do professor:\n{rationale}\n\n"
        return base + (
            "Use o raciocínio de apoio como contexto. "
            "Raciocine de forma breve e objetiva e finalize obrigatoriamente com 'Resposta: X'."
        )

    raise ValueError(f"invalid prompt_style: {prompt_style}")


def build_answer_content(row):
    return f"Resposta: {str(row['resposta']).strip().upper()}"


def build_assistant_content(think_block, row, position="before"):
    """Assemble the training target.

    position:
      'before' -> <think>r</think> + answer   (tagged; the original Stage-1 design)
      'after'  -> answer + <think>r</think>   (post-label; Wadhwa et al., EMNLP 2024)
      'notag'  -> r without tags + answer     (reasoning in the response BODY)

    Why 'notag' exists: under the historical masking bug, 46% of `correct`
    generations emitted an EMPTY <think></think> and put the reasoning in the
    body -- and that accidental mode scored HIGHER (0.416) than a correctly
    masked `before` (0.385). Hypothesis: Qwen3's <think> is a special token tied
    to the model's NATIVE reasoning behaviour (English, its own style) and
    collides with a distilled rationale (Portuguese, the teacher's style).
    Outside the tag it is ordinary text, so there is no collision. Stage 1 tests
    that hypothesis directly instead of relying on the accident.

    shuffle_notag reuses this same 'notag' position; the derangement happens
    earlier, when the rationale assignment is built in build_dataset()."""
    answer = build_answer_content(row)
    if not think_block:
        return answer
    if position == "after":
        return f"{answer}\n\n{think_block}"
    if position == "notag":
        body = re.sub(r"</?think>", "", think_block).strip()
        return f"{body}\n\n{answer}"
    return f"{think_block}\n\n{answer}"


# ============ subset filters ============
def filter_by_subset(df, subset, label="df"):
    """Subsets come from the paired teacher x untrained-student partition:
    A = both right, B = teacher right / student wrong (the TRANSFER subset),
    C = teacher wrong / student right, D = both wrong (the HARD stratum).
    Note the data files store the hard stratum under the legacy code "H"."""
    if subset == "all":
        out = df.reset_index(drop=True).copy()
        log(f"{label} subset all: {len(out)}/{len(df)} rows")
        return out

    if "subset_code" not in df.columns:
        raise ValueError(f"{label} has no subset_code column")

    wanted = set(str(subset).upper())
    before = len(df)
    out = df[df["subset_code"].astype(str).str.upper().isin(wanted)].reset_index(drop=True).copy()
    log(f"{label} subset {subset}: {len(out)}/{before} rows")
    return out


# ============ completion-only tokenization ============
def tokenize_example(tokenizer, user_content, assistant_content, max_len):
    """Build input_ids/labels with loss ONLY on the completion.

    HISTORICAL BUG (fixed 14 Jul), kept documented for transparency: Qwen3's chat
    template with enable_thinking=False injects an EMPTY '<think>\\n\\n</think>'
    block at the end of the prompt, but does NOT inject it into the full sequence
    when the assistant content already contains <think>. Prompt and full therefore
    diverged, and masking the first len(prompt) tokens of the full sequence ate the
    START OF THE TARGET. In `correct_after` that swallowed the whole 'Resposta: X'
    (the loss never saw the answer, so the model generated nothing); in `before` it
    ate the opening rationale tokens (the model never learned to OPEN the <think>).

    Fix: build the prompt with enable_thinking=True (which injects nothing) and the
    full sequence as prompt + target, so the prompt is an EXACT prefix -- asserted
    below rather than assumed."""
    messages = [{"role": "user", "content": user_content}]

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,      # does NOT inject an empty <think></think>
    )

    full_text = prompt_text + assistant_content + "<|im_end|>\n"

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    full_ids = tokenizer(full_text, add_special_tokens=False).input_ids

    # GUARANTEE: the prompt must be an exact prefix of the full sequence
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise RuntimeError(
            "MASKING BROKEN: the prompt is not a prefix of the full sequence. "
            "Masking would eat the start of the target.")

    # AUDIT GUARD: right-truncation removes the TAIL, and in the 'before' and
    # 'notag' layouts the answer IS the tail -- the example would then train a
    # rationale with no answer, without raising, because loss tokens still exist.
    # Measured over the archived data the longest example reaches ~1.3k tokens
    # against max_len=2048, so this never fired; the check keeps it that way for
    # anyone running with a smaller budget.
    if len(full_ids) > max_len:
        raise RuntimeError(
            f"Example exceeds max_len={max_len}; right-truncation would drop the "
            f"answer at the end of the target. Raise --max-seq-length or shorten "
            f"the rationale with --token-size.")

    labels = list(full_ids)
    for i in range(min(len(prompt_ids), len(labels))):
        labels[i] = -100

    n_loss_tokens = sum(x != -100 for x in labels)
    if n_loss_tokens == 0:
        raise RuntimeError("Example has no loss tokens. Raise --max-seq-length.")

    return {
        "input_ids": full_ids,
        "labels": labels,
        "n_loss_tokens": n_loss_tokens,
        "n_prompt_tokens": len(prompt_ids),
    }


def derange(eligible, rng):
    """Assign each eligible item a rationale belonging to a DIFFERENT item.

    Returns the permutation `perm` such that item `eligible[k]` receives the
    rationale of item `perm[k]`. Raises if any item would receive its own.

    Extracted as a function so tests/test_rationale_derangement.py can exercise
    it on CPU, over many seeds, without loading a model. The causal claim of
    Stage 1 rests entirely on this being a true derangement: if an item keeps its
    own rationale, the 'shuffled' arm is quietly a little bit coherent."""
    perm = rng.permutation(eligible)
    # Repair fixed points. Scanning upwards, swapping with position k+1 cannot
    # leave a fixed point at k+1 (it receives eligible[k] != eligible[k+1], and
    # k+1 is revisited anyway); only the wrap-around at k = n-1 can leave one.
    for k in range(len(eligible)):
        if perm[k] == eligible[k]:
            j = (k + 1) % len(eligible)
            perm[k], perm[j] = perm[j], perm[k]
    n_self = sum(int(perm[k] == eligible[k]) for k in range(len(eligible)))
    if n_self:
        raise RuntimeError(
            f"DERANGEMENT BROKEN: {n_self} item(s) received their own rationale "
            f"(wrap-around edge in the repair). Switch to Sattolo or "
            f"reject-and-resample for this seed.")
    return perm


def build_dataset(df, tokenizer, think_mode, args):
    df = filter_by_subset(df, args.subset, label="Train")

    if think_mode not in TRAINED_MODES:
        raise ValueError(f"build_dataset called for a non-trainable mode: {think_mode}")

    records = []
    for _, row in df.iterrows():
        block = build_think_block(row, max_tokens=args.token_size)
        records.append({"row": row,
                        "teacher_correct": bool(row["teacher_correct"]),
                        "block": block})

    # only teacher-correct items that actually carry a rationale are eligible
    eligible = [i for i, r in enumerate(records) if r["teacher_correct"] and r["block"]]
    assignment = {i: records[i]["block"] for i in eligible}

    # `correct_after` reuses exactly the same rationale as `correct`; only the
    # POSITION changes. The deranged modes reassign rationales across questions.
    if think_mode in DERANGED_MODES and len(eligible) > 1:
        # split seed + shuffle seed => reproducible, and different per split
        rng = np.random.default_rng(args.shuffle_seed + args.current_split_seed)
        perm = derange(eligible, rng)
        assignment = {eligible[k]: records[perm[k]]["block"] for k in range(len(eligible))}

    rows = []
    example_audit = []
    n_with_think = n_answer_only = n_skipped = 0
    loss_token_counts = []

    for i, record in enumerate(records):
        block = assignment.get(i, "") if think_mode != "none" else ""

        # If the point is to train on rationales, do not dilute the arm with
        # answer-only targets -- unless that mixed design is what you want.
        if args.train_only_rationale_examples and think_mode in RATIONALE_MODES and not block:
            n_skipped += 1
            continue

        user_c = build_user_content(
            record["row"],
            prompt_style="answer_only",
            include_rationale=False,
            max_think_tokens=args.token_size,
        )
        position = POSITION_BY_MODE.get(think_mode, "before")
        assistant_c = build_assistant_content(block, record["row"], position=position)
        example = tokenize_example(tokenizer, user_c, assistant_c, args.max_seq_length)
        rows.append({"input_ids": example["input_ids"], "labels": example["labels"]})

        n_with_think += int(bool(block))
        n_answer_only += int(not block)
        loss_token_counts.append(example["n_loss_tokens"])
        example_audit.append({
            "had_think_target": bool(block),
            "teacher_correct": bool(record["teacher_correct"]),
            "n_loss_tokens": int(example["n_loss_tokens"]),
        })

    if not rows:
        raise RuntimeError(
            "Training set empty after filtering. Disable --train-only-rationale-examples "
            "or check the data.")

    audit = {
        "mode": think_mode,
        "subset": args.subset,
        "n_rows_after_subset": len(records),
        "n_eligible_teacher_correct_with_think": len(eligible),
        "n_used": len(rows),
        "n_skipped": n_skipped,
        "n_think_targets": n_with_think,
        "n_plain_answer_targets": n_answer_only,
        "loss_tokens_mean": float(np.mean(loss_token_counts)),
        "loss_tokens_min": int(np.min(loss_token_counts)),
        "loss_tokens_max": int(np.max(loss_token_counts)),
        "train_only_rationale_examples": bool(args.train_only_rationale_examples),
    }

    log(f"Training examples used: {audit['n_used']} | {n_with_think} with a rationale in "
        f"the loss | {n_answer_only} answer-only | skipped={n_skipped} | "
        f"mode={think_mode}, subset={args.subset}")
    log(f"Tokenization: loss_tokens_mean={audit['loss_tokens_mean']:.1f} | "
        f"min={audit['loss_tokens_min']} | max={audit['loss_tokens_max']}")

    return Dataset.from_list(rows), audit, pd.DataFrame(example_audit)


class PadCollator:
    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, features):
        maxlen = max(len(f["input_ids"]) for f in features)
        input_ids, labels, attention = [], [], []
        for f in features:
            ids, lab = f["input_ids"], f["labels"]
            pad = maxlen - len(ids)
            input_ids.append(ids + [self.pad_id] * pad)
            labels.append(lab + [-100] * pad)          # padding never enters the loss
            attention.append([1] * len(ids) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention),
            "labels": torch.tensor(labels),
        }


# ============ inference ============
def extract_predicted(generated_text):
    """Read the predicted letter from a generation, taking the LAST match.

    AUDIT FIX: this used to take the FIRST 'Resposta: X'. In the 'before' and
    'notag' layouts the rationale precedes the answer, so a letter mentioned
    mid-reasoning ("...isso sugeriria Resposta: B, mas...") beat the model's
    actual conclusion. Measured over the 9840 archived generations: 3.4-5.6% of
    trained generations disagree between the two rules, and on those the last
    match is right 37.0% of the time against 15.6% for the first. The old rule
    biased trained conditions DOWN by about 1 percentage point and left the
    untrained baseline untouched (it does not reason before answering), so it was
    both conservative and differential.

    REPRODUCIBILITY: the Stage-1 accuracies printed in the paper were computed
    with the old first-match rule. Re-running this script reproduces every
    conclusion with trained accuracies about 1 pp higher; see the README section
    "Known deviation: answer extraction" for the full before/after table."""
    if generated_text is None:
        return None

    matches = re.findall(r"Resposta\s*[:\-]?\s*\**\s*([A-E])", str(generated_text), re.I)
    if matches:
        return matches[-1].upper()

    # fallbacks operate on the tail only, so they cannot pick a letter out of the
    # middle of a long rationale
    tail = str(generated_text)[-500:]
    patterns = (
        r"resposta\s+final\s*[:\-]?\s*([A-E])\b",
        r"alternativa\s+correta\s*[:\-]?\s*([A-E])\b",
        r"answer\s+(?:is|should\s+be)\s+([A-E])\b",
        r"\bis\s+([A-E])[\),\.]",
        r"[Oo]ption\s+([A-E])\b",
        r"\b([A-E])\s*[\.|\)]\s*$",
    )
    for pattern in patterns:
        m = re.search(pattern, tail, re.I)
        if m:
            return m.group(1).upper()
    return None


def think_chars_from_generation(text):
    """Characters of reasoning emitted before the answer, whether or not the model
    closed a <think> span -- this is how reasoning that moved into the response
    body is detected."""
    s = "" if text is None else str(text)
    prefix = s.split("</think>", 1)[0] if "</think>" in s else s
    prefix = re.sub(r"(?i)<think>", "", prefix)
    return len(prefix.strip())


@torch.no_grad()
def evaluate(model, tokenizer, test_df, args, prompt_style="reasoning",
             max_new_tokens=768, batch_size=16):
    FastLanguageModel.for_inference(model)
    tokenizer.padding_side = "left"
    device = next(model.parameters()).device

    prompts, expected, subsets, item_ids, prompt_has_rationale = [], [], [], [], []
    for idx, row in test_df.iterrows():
        has_rationale = (bool(get_reasoning_body(row, max_tokens=args.token_size))
                         if prompt_style == "rationale_prompt" else False)
        prompts.append(
            tokenizer.apply_chat_template(
                [{"role": "user",
                  "content": build_user_content(row, prompt_style,
                                                max_think_tokens=args.token_size)}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        )
        expected.append(str(row["resposta"]).strip().upper())
        subsets.append(str(row.get("subset_code", "")))
        prompt_has_rationale.append(int(has_rationale))

        # NOTE: with the released split files this is the dataset `id`; with the
        # archived 2026 runs it fell through to the dataframe index, so those
        # item_ids are positional and NOT joinable to data/questions_2024_2025.jsonl.
        if "item_id" in row:
            item_ids.append(str(row["item_id"]))
        elif "id_original" in row:
            item_ids.append(str(row["id_original"]))
        else:
            item_ids.append(str(idx))

    generations = []
    from tqdm import tqdm

    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id

    for i in tqdm(range(0, len(prompts), batch_size),
                  desc=f"inference/{prompt_style}", unit="batch"):
        batch = prompts[i: i + batch_size]
        encoded = tokenizer(batch, return_tensors="pt", padding=True,
                            add_special_tokens=False).to(device)

        out = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
        )

        new_tokens = out[:, encoded["input_ids"].shape[1]:]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

        for j, (text, gold, subset_code, item_id, has_rationale) in enumerate(
            zip(decoded,
                expected[i: i + batch_size],
                subsets[i: i + batch_size],
                item_ids[i: i + batch_size],
                prompt_has_rationale[i: i + batch_size])
        ):
            sequence = new_tokens[j]
            n_new = (int((sequence != pad_id).sum().item()) if pad_id is not None
                     else int(sequence.numel()))
            predicted = extract_predicted(text)
            generations.append({
                "item_id": item_id,
                "expected": gold,
                "predicted": predicted,
                # a generation that emits no letter counts as WRONG, never dropped
                "correct": int(predicted == gold),
                "subset_code": subset_code,
                "prompt_style": prompt_style,
                "prompt_has_rationale": has_rationale,
                "new_tokens": n_new,
                "hit_max_new_tokens": int(n_new >= max_new_tokens),
                "closed_think": int("</think>" in str(text)),
                "has_answer_pattern": int(
                    re.search(r"Resposta\s*[:\-]", str(text), re.I) is not None),
                "think_chars": think_chars_from_generation(text),
                "gen_chars": len(str(text)),
                "gen": text,
            })

    return generations


def bootstrap(generations, n_boot=10000, seed=0):
    """Item-level bootstrap WITHIN a single run. The paper reports Stage-1
    accuracies as ten-seed means with paired tests across seeds; this interval
    describes one run and is not the paper's inferential procedure (for that see
    code/analysis/stage2_analyze_full.py, two_level_bootstrap)."""
    correct = np.array([g["correct"] for g in generations], dtype=float)
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(correct, len(correct), replace=True).mean()
                      for _ in range(n_boot)])
    return {
        "mean": float(correct.mean()),
        "ci_025": float(np.percentile(boots, 2.5)),
        "ci_975": float(np.percentile(boots, 97.5)),
        "n": int(len(correct)),
        "n_emitted": int(sum(g["predicted"] is not None for g in generations)),
    }


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_one(seed, think_mode, args):
    assert think_mode in set(ALL_MODES)
    args.current_split_seed = int(seed)

    # Separating the split seed from the initialisation seed keeps split effects
    # from being confounded with LoRA initialisation effects.
    runtime_seed = args.lora_random_state if args.lora_random_state >= 0 else seed
    set_all_seeds(runtime_seed)

    epoch_tag = 0 if think_mode in INFERENCE_ONLY_MODES else args.epochs
    train_flag = "ratonly" if args.train_only_rationale_examples else "mixed"
    # WARNING: the run name encodes the FULL configuration. Two runs differing
    # only in max_new_tokens or subset are DIFFERENT CONDITIONS -- see the
    # "aggregation trap" section of the README before pooling anything.
    run_name = (
        f"stage1_train{args.subset}_eval{args.eval_subset}_"
        f"{think_mode}_ep{epoch_tag}_tok{args.token_size}_mnt{args.max_new_tokens}_"
        f"prompt{args.inference_prompt_style}_{train_flag}_lora{runtime_seed}_seed{seed}"
    )

    out_dir = Path(args.output_dir) / run_name
    save_dir = Path(args.model_save_root) / run_name
    ckpt_dir = save_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)

    boot_path = out_dir / "bootstrap_results.json"
    if boot_path.exists() and not args.force:
        log(f"[SKIP] {run_name} -- use --force to overwrite")
        return

    log("=" * 100)
    log(f"===== {run_name} =====")
    log("=" * 100)

    run_config = vars(args).copy()
    run_config["seed"] = seed
    run_config["think_mode"] = think_mode
    run_config["run_name"] = run_name
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False)

    test_path = Path(args.data_dir) / f"test_seed_{seed}_{args.token_size}tokens.parquet"
    test_df = pd.read_parquet(test_path)
    test_df = filter_by_subset(test_df, args.eval_subset, label="Eval")
    log(f"Final test set: {len(test_df)} rows")

    if think_mode in INFERENCE_ONLY_MODES:
        log(f"Mode {think_mode.upper()}: untrained baseline | "
            f"prompt_style={args.inference_prompt_style}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_name,
            max_seq_length=args.max_seq_length,
            load_in_8bit=False,
            full_finetuning=False,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        generations = evaluate(model, tokenizer, test_df, args,
                               prompt_style=args.inference_prompt_style,
                               max_new_tokens=args.max_new_tokens,
                               batch_size=args.infer_batch_size)
        pd.DataFrame(generations).to_csv(out_dir / "infer_results.csv", index=False)
        boot = bootstrap(generations, seed=seed)
        with open(boot_path, "w") as f:
            json.dump(boot, f, indent=2, ensure_ascii=False)
        log(f"{think_mode} acc={boot['mean']:.3%} | emitted={boot['n_emitted']}/{boot['n']}")
        del model, tokenizer
        clear_memory()
        return

    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_path = Path(args.data_dir) / f"train_seed_{seed}_{args.token_size}tokens.parquet"
    train_df = pd.read_parquet(train_path)
    log(f"Train (raw): {len(train_df)} | Test: {len(test_df)}")

    model_base, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_8bit=False,
        full_finetuning=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model_base,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=runtime_seed,
        use_rslora=False,
        loftq_config=None,
    )

    train_ds, train_audit, example_audit = build_dataset(train_df, tokenizer, think_mode, args)
    with open(out_dir / "train_audit.json", "w") as f:
        json.dump(train_audit, f, indent=2, ensure_ascii=False)
    example_audit.to_csv(out_dir / "train_example_audit.csv", index=False)

    cfg = SFTConfig(
        output_dir=str(ckpt_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        lr_scheduler_type="linear",
        weight_decay=0.01,
        logging_steps=10,
        report_to="none",
        dataset_text_field=None,
        max_length=args.max_seq_length,
        fp16=False,
        bf16=True,
        packing=False,
        completion_only_loss=False,   # the mask is built by tokenize_example
        remove_unused_columns=False,
        save_strategy="no",
    )

    trainer = Trainer(model=model, processing_class=tokenizer, args=cfg,
                      train_dataset=train_ds, data_collator=PadCollator(tokenizer))
    log(f"Training mode={think_mode}...")
    trainer.train()

    model.save_pretrained(save_dir / "model_lora")
    tokenizer.save_pretrained(save_dir / "model_lora")

    log(f"Inference mode={think_mode} | prompt_style={args.inference_prompt_style}...")
    generations = evaluate(model, tokenizer, test_df, args,
                           prompt_style=args.inference_prompt_style,
                           max_new_tokens=args.max_new_tokens,
                           batch_size=args.infer_batch_size)
    pd.DataFrame(generations).to_csv(out_dir / "infer_results.csv", index=False)

    boot = bootstrap(generations, seed=seed)
    with open(boot_path, "w") as f:
        json.dump(boot, f, indent=2, ensure_ascii=False)
    log(f"{think_mode} acc={boot['mean']:.3%} | emitted={boot['n_emitted']}/{boot['n']}")

    del model, model_base, tokenizer, trainer, train_ds
    clear_memory()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--think-modes", nargs="+", choices=ALL_MODES,
                    default=["base", "correct", "shuffle"])
    ap.add_argument("--subset", default="AB", help="training subset (letters, e.g. AB)")
    ap.add_argument("--eval-subset", default="B", help="evaluation subset")
    ap.add_argument("--seeds", type=int, nargs="+", default=[8])
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--token-size", type=int, default=250,
                    help="rationale truncation, in whitespace words")
    ap.add_argument("--data-dir", default=RATIONALE_DATA_DIR)
    ap.add_argument("--model-name", default=MODEL_NAME)
    ap.add_argument("--model-save-root", default=MODEL_SAVE_ROOT)
    ap.add_argument("--output-dir", default=BASE_OUTPUT_DIR)
    ap.add_argument("--shuffle-seed", type=int, default=123)
    ap.add_argument("--max-seq-length", type=int, default=MAX_SEQ_LENGTH)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--infer-batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-random-state", type=int, default=3407,
                    help="use -1 for the legacy behaviour random_state=seed")
    ap.add_argument("--inference-prompt-style",
                    choices=["answer_only", "reasoning", "rationale_prompt"],
                    default="reasoning")
    ap.add_argument("--train-only-rationale-examples", action="store_true")
    ap.add_argument("--force", action="store_true")

    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_save_root, exist_ok=True)

    log(f"Seeds/splits: {args.seeds}")
    log(f"Think modes: {args.think_modes}")
    log(f"Train subset: {args.subset} | Eval subset: {args.eval_subset}")
    log(f"Token size: {args.token_size} | Max seq length: {args.max_seq_length} "
        f"| Max new tokens: {args.max_new_tokens}")
    log(f"Inference prompt style: {args.inference_prompt_style}")
    log(f"LoRA random state: {args.lora_random_state}")
    log(f"Train only rationale examples: {args.train_only_rationale_examples}")

    for seed in args.seeds:
        for think_mode in args.think_modes:
            run_one(seed, think_mode, args)


if __name__ == "__main__":
    main()

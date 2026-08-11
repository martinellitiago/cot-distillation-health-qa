#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 2 -- three distillation techniques x two inference regimes x three alphas.

Techniques
  pure_sft      every target is answer-only ("Resposta: X"), rationales ignored
                even where they exist. Inference: direct answering ONLY, since
                the model was never trained to reason.
  distill_sft   A/B -> "<think>rationale</think>\\n\\nResposta: X"
                C/D -> "Resposta: X" (no rationale available)
                Inference: direct answering AND reasoning.
  step_by_step  multi-task (Hsieh et al., 2023)
                task [label]     -> every example
                task [rationale] -> only items that have a rationale
                Inference: direct answering AND reasoning.

THE TWO ALPHAS ARE DIFFERENT QUANTITIES and must not be read as one knob:
  distill_sft   alpha weights the RATIONALE TOKENS inside a single target;
                answer tokens always weigh 1.0            (token-level)
  step_by_step  alpha weights the whole rationale TASK relative to the label
                task -- this is Hsieh's lambda            (example-level)

Data: data/splits_stage2/
  df_train_seed{S}.pkl     -> ABC(85%) + hard_train (answer-only)
  df_test_abc_seed{S}.pkl  -> ABC(15%) held out
  df_test_hard_seed{S}.pkl -> hard held out

INFERENCE REGIMES. Both build the prompt with enable_thinking=True, identical to
training, so the prompt is an exact prefix. They differ in three ways, applied
identically to every technique, so between-technique comparisons within a regime
remain clean:
  1. the instruction ("reason step by step and finish with 'Resposta: X'" vs
     "answer only with 'Resposta: X'; do not explain");
  2. reasoning PRIMES the span itself by appending "\\n<think>\\nVamos analisar a
     questão:" -- without this Qwen3 falls back to its NATIVE reasoning mode,
     which comes out in English instead of the distilled Portuguese;
  3. the generation ceiling (--mnt-answer 64 vs --mnt-reasoning 1536).

HISTORY (kept for transparency):
  13 Jul -- inference used to force enable_thinking=False, which made Qwen3 inject
    an EMPTY <think></think> and answer immediately; the 'reasoning' regime then
    generated ~5 tokens and scored IDENTICALLY to direct answering. Replaced by
    the explicit Portuguese priming above.
  13 Jul -- separate ceilings per regime. With a single ceiling of 768 the
    reasoning regime truncated before concluding (only 74% of answers emitted),
    depressing acc_all even though acc_emitted was HIGHER than direct answering.
    The script now logs the share that hit the ceiling and warns above 10%.
  14 Jul -- training masking fix; see tokenize_example.

Runtime guarantees (do not remove): the MASKING BROKEN / MASKING EMPTY asserts,
the automatic masking self-test, and the ceiling alert.

Usage
  python stage2_run.py --techniques pure_sft distill_sft step_by_step \\
      --seeds 8 12 17 23 25 31 37 44 52 61
"""

import argparse, gc, json, os, re, random, time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from tqdm import tqdm
import transformers
from transformers import Trainer, TrainerCallback, set_seed
from trl import SFTConfig
from unsloth import FastLanguageModel

# silence the repeated max_new_tokens vs max_length warning (it floods the log)
transformers.logging.set_verbosity_error()
import warnings
warnings.filterwarnings("ignore", message=".*max_new_tokens.*")

MODEL_NAME = "unsloth/Qwen3-4B"
SPLITS_DIR = Path("data/splits_stage2")
OUT_ROOT = Path("outputs_stage2")
MAX_SEQ_LEN = 2048

TECHNIQUES = ["pure_sft", "distill_sft", "step_by_step"]

# Inference regimes per technique. pure_sft never reasons: it was not trained to,
# so scoring it in a mode it never learned would be an unfair comparison.
INFER_REGIMES = {
    "pure_sft":     ["answer_only"],
    "distill_sft":  ["answer_only", "reasoning"],
    "step_by_step": ["answer_only", "reasoning"],
}


def log(m):
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)


def set_all_seeds(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    set_seed(s)


# ---------------------------------------------------------
# PROMPTS  (Portuguese by design: the examination and the distilled
# rationales are Portuguese; these strings are data, not UI text)
# ---------------------------------------------------------
def format_options(options):
    import ast
    if isinstance(options, str):
        options = ast.literal_eval(options)
    return "\n".join(f"{k.strip().upper()}) {v}" for k, v in sorted(options.items()))


def user_content(stem, options_fmt, style, task=None):
    """style: 'answer_only' | 'reasoning'.  task: step_by_step only ('label'|'rationale')."""
    base = f"Questão:\n{stem}\n\nAlternativas:\n{options_fmt}\n\n"
    if task == "rationale":
        return base + "Explique o raciocínio clínico que leva à alternativa correta."
    if style == "reasoning":
        return base + "Raciocine passo a passo e termine com 'Resposta: X' (X = A-E)."
    return base + "Responda apenas com 'Resposta: X' (X = A-E). Não explique."


def assistant_content(row, technique, task=None):
    """The training target, per technique."""
    answer = f"Resposta: {str(row['resposta']).strip().upper()}"
    rationale = row.get("target_thinking")
    has_rationale = isinstance(rationale, str) and len(rationale.strip()) > 10

    if technique == "pure_sft":
        return answer                                  # always answer-only

    if technique == "distill_sft":
        if has_rationale:
            return f"<think>\n{rationale.strip()}\n</think>\n\n{answer}"
        return answer                                  # C/D: answer-only

    if technique == "step_by_step":
        if task == "rationale":
            return rationale.strip() if has_rationale else None
        return answer                                  # label task

    raise ValueError(technique)


# ---------------------------------------------------------
# TOKENIZATION + MASKING  (same contract as Stage 1)
# ---------------------------------------------------------
def tokenize_example(tokenizer, user_c, assistant_c, max_len,
                     _retry=0, alpha=1.0, rationale_text=None):
    """Loss ONLY on the answer/rationale; the prompt is masked with -100.
    If the sequence exceeds max_len the RATIONALE is shortened -- never the tail,
    because the tail is where "Resposta: X" lives.

    alpha           weight applied to the RATIONALE tokens (1.0 = same as answer)
    rationale_text  the rationale, used to locate where it ends and the answer begins

    MASKING FIX (14 Jul): with enable_thinking=False the Qwen3 template injects an
    empty "<think>\\n\\n</think>" block at the end of the prompt but does NOT inject
    it into the full sequence when the assistant content already has <think>. The
    two diverged, so masking the first len(prompt) tokens ate the START OF THE
    TARGET -- for distill_sft, exactly the tokens that OPEN the <think>. Now the
    prompt uses enable_thinking=True and full = prompt + target, asserted below."""
    prompt_txt = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_c}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True,
    )
    full_txt = prompt_txt + assistant_c + "<|im_end|>\n"

    prompt_ids = tokenizer(prompt_txt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_txt, add_special_tokens=False)["input_ids"]

    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("MASKING BROKEN: the prompt is not a prefix of the full sequence.")

    if len(full_ids) > max_len:
        if _retry >= 3:
            raise ValueError(
                f"Example exceeds max_len={max_len} even after shortening the rationale.")
        excess = len(full_ids) - max_len
        m = re.search(r"<think>\n?(.*?)\n?</think>", assistant_c, re.S)
        if m:
            current = m.group(1)
            # cut from the START of the rationale, keeping the conclusion at the end
            keep_chars = max(50, len(current) - excess * 6)   # ~6 chars/token of slack
            shortened = current[-keep_chars:]
            new_assistant = assistant_c.replace(m.group(0), f"<think>\n{shortened}\n</think>")
            # alpha/rationale_text MUST be threaded through the recursion: without
            # them a truncated example silently loses its loss weighting. Harmless
            # at max_len 2048 (nothing truncates) but critical if it is lowered.
            return tokenize_example(tokenizer, user_c, new_assistant, max_len, _retry + 1,
                                    alpha=alpha, rationale_text=shortened)
        raise ValueError(f"Answer-only example exceeds max_len={max_len} (the prompt is enormous).")

    labels = list(full_ids)
    n_prompt = min(len(prompt_ids), len(labels))
    for i in range(n_prompt):
        labels[i] = -100

    n_loss = sum(x != -100 for x in labels)
    if n_loss == 0:
        raise ValueError("MASKING EMPTY: no loss tokens (the prompt swallowed the target).")

    # ---- PER-TOKEN WEIGHTS (alpha on the rationale, 1.0 on the answer) ----
    weights = [0.0] * len(full_ids)
    if alpha == 1.0 or not rationale_text:
        for i in range(n_prompt, len(full_ids)):
            weights[i] = 1.0
    else:
        # locate where the rationale ends by tokenizing prompt + "<think>...</think>"
        # NOTE: tokenizing the head separately can differ from the full sequence by
        # one token at the boundary if the tokenizer merges across it, so the cut may
        # be off by ~1 token out of several hundred. Immaterial to the weighting.
        m = re.search(r"(<think>.*?</think>)", assistant_c, re.S)
        if m:
            head_txt = prompt_txt + m.group(1)        # same construction as full_txt
            head_ids = tokenizer(head_txt, add_special_tokens=False)["input_ids"]
            cut = min(len(head_ids), len(full_ids))
        else:
            cut = n_prompt                            # no rationale -> all answer
        for i in range(n_prompt, len(full_ids)):
            weights[i] = alpha if i < cut else 1.0

    return {"input_ids": full_ids, "labels": labels, "weights": weights,
            "n_prompt": n_prompt, "n_loss": n_loss}


def masking_self_test(tokenizer, ds_rows, technique, n_check=3):
    """Fails loudly if the masking is wrong, and prints one decoded example so the
    target is visible in the log rather than merely asserted."""
    log(f"  >>> MASKING SELF-TEST ({technique})")
    for i, ex in enumerate(ds_rows[:n_check]):
        ids, lab = ex["input_ids"], ex["labels"]
        assert len(ids) == len(lab), "input_ids and labels have different lengths"
        assert any(x != -100 for x in lab), "no loss tokens"
        assert lab[0] == -100, "the first token should be masked (it is prompt)"
        # every unmasked token must sit at the END of the sequence (it is the target)
        first_loss = next(j for j, x in enumerate(lab) if x != -100)
        assert all(x != -100 for x in lab[first_loss:]), \
            "masked tokens found AFTER the target starts (inconsistent masking)"
        for j in range(first_loss, len(lab)):
            assert lab[j] == ids[j], f"label != input_id at position {j}"
        if i == 0:
            masked = tokenizer.decode(ids[:first_loss], skip_special_tokens=False)
            trained = tokenizer.decode(ids[first_loss:], skip_special_tokens=False)
            print("  " + "-" * 60)
            print("  [MASKED - no loss] (prompt):")
            print("  " + masked[:300].replace("\n", "\n  "))
            print("  [WITH LOSS - what the model learns to generate]:")
            print("  " + trained[:300].replace("\n", "\n  "))
            print(f"  prompt tokens={first_loss} | loss tokens={len(lab)-first_loss}")
            w = ex.get("weights")
            if w:
                target_w = w[first_loss:]
                print(f"  loss weights on the target: "
                      f"{sorted(set(round(x, 3) for x in target_w))} "
                      f"(rationale={sum(1 for x in target_w if x < 1.0)} tokens, "
                      f"answer={sum(1 for x in target_w if x == 1.0)} tokens)")
            print("  " + "-" * 60)
    log(f"  >>> MASKING OK ({technique})")


# ---------------------------------------------------------
# DATASET
# ---------------------------------------------------------
def build_dataset(df, tokenizer, technique, max_len, alpha=1.0):
    rows = []
    audit = {"with_rationale": 0, "answer_only": 0, "rationale_task": 0}
    for _, r in df.iterrows():
        options = format_options(r["alternativas"])

        if technique == "step_by_step":
            # task 1: label (EVERY example)
            uc = user_content(r["enunciado"], options, "answer_only", task="label")
            ac = assistant_content(r, technique, task="label")
            rows.append(tokenize_example(tokenizer, uc, ac, max_len))
            audit["answer_only"] += 1
            # task 2: rationale (only where one exists) -- alpha weights the WHOLE task
            ac_rationale = assistant_content(r, technique, task="rationale")
            if ac_rationale:
                uc_rationale = user_content(r["enunciado"], options, "reasoning", task="rationale")
                ex = tokenize_example(tokenizer, uc_rationale, ac_rationale, max_len)
                if alpha != 1.0:
                    ex["weights"] = [w * alpha for w in ex["weights"]]
                rows.append(ex)
                audit["rationale_task"] += 1
        else:
            rationale = r.get("target_thinking")
            has_rationale = isinstance(rationale, str) and len(str(rationale).strip()) > 10
            style = "reasoning" if (technique == "distill_sft" and has_rationale) else "answer_only"
            uc = user_content(r["enunciado"], options, style)
            ac = assistant_content(r, technique)
            rows.append(tokenize_example(
                tokenizer, uc, ac, max_len,
                alpha=alpha if (technique == "distill_sft" and has_rationale) else 1.0,
                rationale_text=str(rationale) if has_rationale else None))
            if technique == "distill_sft" and has_rationale:
                audit["with_rationale"] += 1
            else:
                audit["answer_only"] += 1

    log(f"  dataset [{technique}]: {len(rows)} examples | audit={audit}")
    return rows, audit


class Collator:
    def __init__(self, tokenizer):
        self.pad = tokenizer.pad_token_id

    def __call__(self, features):
        maxlen = max(len(f["input_ids"]) for f in features)
        ids, labels, attention, weights = [], [], [], []
        for f in features:
            pad = maxlen - len(f["input_ids"])
            ids.append(f["input_ids"] + [self.pad] * pad)
            labels.append(f["labels"] + [-100] * pad)     # padding never enters the loss
            attention.append([1] * len(f["input_ids"]) + [0] * pad)
            weights.append(f.get("weights", [1.0] * len(f["input_ids"])) + [0.0] * pad)
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(attention),
                "labels": torch.tensor(labels),
                "token_weights": torch.tensor(weights, dtype=torch.float)}


class WeightedTrainer(Trainer):
    """Per-token weighted loss: alpha on the rationale, 1.0 on the answer.

    AUDIT FIX: this trainer is now used for EVERY arm, including alpha=1.0 where
    all weights are 1.0 and the arithmetic reduces exactly to the unweighted mean.
    Previously alpha=1.0 fell back to the stock HF Trainer, which -- from
    transformers 4.46 -- consumes `num_items_in_batch` to renormalise the loss
    under gradient accumulation, while this custom compute_loss does not. With
    --grad-accum 8 that meant the alpha=1.0 arm and the alpha!=1.0 arms used
    subtly different effective normalisation, precisely in the comparison the
    alpha claim rests on. Routing every arm through one code path removes the
    asymmetry by construction. NOTE: the published alpha=1.0 runs were trained
    with the stock Trainer; see README, "Known deviation: loss normalisation".

    The loss is a weighted MEAN over the micro-batch, and the denominator uses the
    same weight*mask as the numerator, so alpha genuinely changes the gradient
    rather than applying a constant scale. Evidence that the objective is live:
    the training loss shifts with alpha (0.565 -> 3.62 at alpha=0.1)."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        weights = inputs.pop("token_weights", None)
        labels = inputs["labels"]
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.logits

        # shift for causal LM: position t predicts token t+1
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_weights = weights[..., 1:].contiguous() if weights is not None else None

        loss_fn = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        losses = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        mask = (shift_labels.view(-1) != -100).float()
        if shift_weights is not None:
            w = shift_weights.view(-1)
            denom = (w * mask).sum().clamp(min=1e-8)
            loss = (losses * w * mask).sum() / denom
        else:
            loss = (losses * mask).sum() / mask.sum().clamp(min=1e-8)

        return (loss, outputs) if return_outputs else loss


class LossLogger(TrainerCallback):
    """Records the loss curve so training robustness can be inspected afterwards."""
    def __init__(self):
        self.history = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.history.append({"step": state.global_step, "loss": logs["loss"],
                                 "epoch": logs.get("epoch"), "grad_norm": logs.get("grad_norm"),
                                 "lr": logs.get("learning_rate")})


# ---------------------------------------------------------
# INFERENCE
# ---------------------------------------------------------
ANSWER_RE = re.compile(r"Resposta\s*[:\-]?\s*\**\s*([A-E])", re.I)


def extract_answer(text):
    """Read the predicted letter, taking the LAST match.

    AUDIT FIX: this used to take the FIRST match. Under the reasoning regime the
    rationale precedes the answer, so a letter mentioned mid-reasoning could beat
    the model's actual conclusion. Under answer_only (~5 tokens, one occurrence)
    the two rules coincide.

    The Stage-2 numbers in the paper were produced with the old first-match rule.
    Its magnitude could not be re-measured because the generated text was not
    retained in the archived inference files; the equivalent effect measured in
    Stage 1 was about +1 pp on reasoning conditions and ~0 on direct answering.
    See README, "Known deviation: answer extraction"."""
    matches = ANSWER_RE.findall(str(text))
    return matches[-1].upper() if matches else None


@torch.no_grad()
def run_inference(model, tokenizer, df, regime, max_new_tokens, batch_size=8, warmup=True):
    FastLanguageModel.for_inference(model)
    prompts, golds = [], []
    for _, r in df.iterrows():
        options = format_options(r["alternativas"])
        uc = user_content(r["enunciado"], options, regime)
        # clean prompt (enable_thinking=True injects no empty <think></think>)
        # -> identical to training. Train/inference consistency.
        p = tokenizer.apply_chat_template(
            [{"role": "user", "content": uc}],
            tokenize=False, add_generation_prompt=True, enable_thinking=True)
        if regime == "reasoning":
            # we open the <think> OURSELVES, in PORTUGUESE, to force the DISTILLED
            # reasoning; without this Qwen3 falls back to its native English mode
            p = p.rstrip() + "\n<think>\nVamos analisar a questão:"
        prompts.append(p)
        golds.append(str(r["resposta"]).strip().upper())

    generations, n_tokens = [], []
    n_batches = (len(prompts) + batch_size - 1) // batch_size

    # THROUGHPUT: time ONLY model.generate, excluding prompt construction, with
    # torch.cuda.synchronize on both sides so the measurement reflects completed
    # GPU work rather than asynchronous kernel launch. Robust to batching: this
    # measures real throughput (tokens/s), never wall time divided by batch size.
    #
    # The first generate() call after loading pays a one-off Triton compilation
    # cost. With `warmup` enabled it runs once on a tiny slice and is NOT counted.
    # AUDIT NOTE: the published timings were measured WITHOUT this warm-up, which
    # inflates the direct-answer regime (total ~8 s) far more than the reasoning
    # regime (~600 s) and therefore makes the reported latency ratio CONSERVATIVE.
    # Use --no-warmup to reproduce the original timing procedure exactly.
    if warmup and prompts:
        warm = tokenizer(prompts[:min(2, len(prompts))], return_tensors="pt",
                         padding=True, truncation=True,
                         max_length=MAX_SEQ_LEN).to(model.device)
        model.generate(**warm, max_new_tokens=8, do_sample=False,
                       pad_token_id=tokenizer.pad_token_id)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    gen_secs_total = 0.0
    for i in tqdm(range(0, len(prompts), batch_size), total=n_batches,
                  desc=f"    infer[{regime}]", ncols=90, leave=False):
        batch = prompts[i:i + batch_size]
        encoded = tokenizer(batch, return_tensors="pt", padding=True,
                            truncation=True, max_length=MAX_SEQ_LEN).to(model.device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model.generate(**encoded, max_new_tokens=max_new_tokens,
                             do_sample=False, temperature=None, top_p=None,
                             pad_token_id=tokenizer.pad_token_id)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gen_secs_total += time.perf_counter() - t0

        for j in range(len(batch)):
            new = out[j][encoded["input_ids"].shape[1]:]
            # count generated tokens up to the first EOS
            new_list = new.tolist()
            if tokenizer.eos_token_id in new_list:
                n_gen = new_list.index(tokenizer.eos_token_id) + 1
            else:
                n_gen = len(new_list)
            n_tokens.append(n_gen)
            generations.append(tokenizer.decode(new, skip_special_tokens=True))

    # sanity: print the first generation (catches a duplicated <think> or a model
    # that silently stopped reasoning)
    if generations:
        preview = generations[0][:200].replace("\n", " | ")
        log(f"      [first generation, regime={regime}] {preview!r}")

    predictions = [extract_answer(g) for g in generations]
    res = pd.DataFrame({
        "gold": golds, "pred": predictions, "gen": generations, "n_new_tokens": n_tokens,
        "max_new_tokens": max_new_tokens,            # ceiling used (traceability)
        "truncated": [n >= max_new_tokens - 2 for n in n_tokens],
        "subset_code": df["subset_code"].values if "subset_code" in df else "?",
    })
    res["correct"] = (res["pred"] == res["gold"]).astype(int)
    # where the reasoning actually IS -- the lesson from Stage 1
    res["reasoning_chars"] = (res["gen"].str.replace(r"<think>\s*</think>", "", regex=True)
                                        .str.strip().str.len())
    # language of the reasoning (EN = fell back to Qwen's native mode; PT = distilled)
    res["reasoning_en"] = res["gen"].astype(str).str.contains(
        r"Okay,|let's|the user|the question|Let me|First,|We need", case=False, regex=True)

    total_tokens = int(sum(n_tokens))
    timing = {
        "gen_secs_total": round(gen_secs_total, 3),
        "n_questions": len(prompts),
        "total_new_tokens": total_tokens,
        "tokens_per_sec": round(total_tokens / gen_secs_total, 2) if gen_secs_total > 0 else 0.0,
        "secs_per_question": round(gen_secs_total / len(prompts), 4) if prompts else 0.0,
        "infer_batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "warmup_discarded": bool(warmup),
    }
    return res, timing


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--techniques", nargs="+", default=TECHNIQUES, choices=TECHNIQUES)
    ap.add_argument("--seeds", nargs="+", type=int,
                    default=[8, 12, 17, 23, 25, 31, 37, 44, 52, 61])
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-random-state", type=int, default=3407)
    ap.add_argument("--mnt-answer", type=int, default=64,
                    help="ceiling for the answer_only regime (~5 tokens are generated)")
    ap.add_argument("--mnt-reasoning", type=int, default=1536,
                    help="ceiling for the reasoning regime (avoids truncating mid-conclusion)")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--infer-batch-size", type=int, default=32,
                    help="inference batch size (an 80GB H100 handles 32)")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="loss weight on the RATIONALE. distill_sft: per token. "
                         "step_by_step: on the whole rationale task (Hsieh's lambda). "
                         "No effect on pure_sft.")
    ap.add_argument("--data-fraction", type=float, default=1.0,
                    help="fraction of the training pool (efficiency curve: 0.25, 0.5, 1.0)")
    ap.add_argument("--no-warmup", action="store_true",
                    help="do not discard a warm-up generate() before timing "
                         "(reproduces the timing procedure used for the paper)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    OUT_ROOT.mkdir(exist_ok=True)

    for seed in args.seeds:
        for tech in args.techniques:
            frac_tag = "" if args.data_fraction == 1.0 else f"_frac{int(args.data_fraction*100)}"
            # alpha tag: 0.1 -> "_a01". Kept for compatibility with the archived run
            # directories; the authoritative value always lives in run_config.json.
            alpha_tag = "" if args.alpha == 1.0 else f"_a{args.alpha:g}".replace(".", "")
            run_dir = OUT_ROOT / f"{tech}_seed{seed}_ep{args.epochs}{frac_tag}{alpha_tag}"
            if (run_dir / "DONE").exists() and not args.force:
                log(f"SKIP {run_dir.name} (DONE)")
                continue
            run_dir.mkdir(parents=True, exist_ok=True)

            log("=" * 70)
            log(f"TECHNIQUE={tech} SEED={seed} FRACTION={args.data_fraction} ALPHA={args.alpha}")
            log("=" * 70)
            set_all_seeds(seed)

            # ---- data ----
            df_train = pd.read_pickle(SPLITS_DIR / f"df_train_seed{seed}.pkl")
            df_abc = pd.read_pickle(SPLITS_DIR / f"df_test_abc_seed{seed}.pkl")
            df_hard = pd.read_pickle(SPLITS_DIR / f"df_test_hard_seed{seed}.pkl")

            if args.data_fraction < 1.0:   # efficiency curve: STRATIFIED sample
                parts = []
                for subset_code, g in df_train.groupby("subset_code"):
                    n = max(1, int(round(len(g) * args.data_fraction)))
                    parts.append(g.sample(n=n, random_state=seed))
                df_train = (pd.concat(parts).sample(frac=1.0, random_state=seed)
                              .reset_index(drop=True))
                log(f"  fraction {args.data_fraction}: train={len(df_train)}")

            log(f"  train={len(df_train)} | test_abc={len(df_abc)} | test_hard={len(df_hard)}")

            # ---- model ----
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=MODEL_NAME, max_seq_length=MAX_SEQ_LEN,
                dtype=None, load_in_4bit=False,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"

            model = FastLanguageModel.get_peft_model(
                model, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
                bias="none", use_gradient_checkpointing="unsloth",
                random_state=args.lora_random_state,
            )

            # ---- dataset + MASKING SELF-TEST ----
            rows, audit = build_dataset(df_train, tokenizer, tech, MAX_SEQ_LEN, alpha=args.alpha)
            if args.alpha != 1.0:
                log(f"  ALPHA={args.alpha} -> the rationale weighs {args.alpha}x in the loss "
                    f"(answer = 1.0)")
            masking_self_test(tokenizer, rows, tech)          # <<< fails if wrong
            json.dump(audit, open(run_dir / "train_audit.json", "w"), indent=2)

            ds = Dataset.from_list([{"input_ids": r["input_ids"], "labels": r["labels"],
                                     "weights": r["weights"]} for r in rows])

            # ---- training ----
            tokenizer.padding_side = "right"    # training pads on the right
            cfg_kwargs = dict(
                output_dir=str(run_dir / "_trainer"),
                per_device_train_batch_size=args.batch_size,
                gradient_accumulation_steps=args.grad_accum,
                num_train_epochs=args.epochs, learning_rate=args.lr,
                lr_scheduler_type="linear", warmup_ratio=0.03,
                logging_steps=10, save_strategy="no", report_to="none",
                bf16=True, seed=seed,
                remove_unused_columns=False,
            )
            try:
                cfg = SFTConfig(**cfg_kwargs, max_seq_length=MAX_SEQ_LEN,
                                dataset_kwargs={"skip_prepare_dataset": True})
            except TypeError:
                # newer TRL renamed max_seq_length -> max_length
                cfg = SFTConfig(**cfg_kwargs, max_length=MAX_SEQ_LEN,
                                dataset_kwargs={"skip_prepare_dataset": True})
            loss_cb = LossLogger()
            # every arm goes through WeightedTrainer -- see the class docstring
            trainer = WeightedTrainer(model=model, processing_class=tokenizer, args=cfg,
                                      train_dataset=ds, data_collator=Collator(tokenizer),
                                      callbacks=[loss_cb])
            log("  training...")
            train_out = trainer.train()

            pd.DataFrame(loss_cb.history).to_csv(run_dir / "train_loss_curve.csv", index=False)
            json.dump({"train_loss": train_out.training_loss, "steps": train_out.global_step},
                      open(run_dir / "train_summary.json", "w"), indent=2)
            log(f"  final train_loss: {train_out.training_loss:.4f}")

            # save the LoRA adapter (needed for the ENEM forgetting probe)
            adapter_dir = run_dir / "adapter"
            model.save_pretrained(str(adapter_dir))
            tokenizer.save_pretrained(str(adapter_dir))
            log(f"  adapter saved to {adapter_dir}")

            # ---- inference: regimes per technique ----
            tokenizer.padding_side = "left"
            infer_timings = {}
            for regime in INFER_REGIMES[tech]:
                for split_name, df_test in [("abc", df_abc), ("hard", df_hard)]:
                    log(f"  inferring [{regime}] on {split_name} ({len(df_test)})...")
                    mnt = args.mnt_reasoning if regime == "reasoning" else args.mnt_answer
                    res, timing = run_inference(model, tokenizer, df_test, regime, mnt,
                                                batch_size=args.infer_batch_size,
                                                warmup=not args.no_warmup)
                    infer_timings[f"{split_name}_{regime}"] = timing
                    res.to_csv(run_dir / f"infer_{split_name}_{regime}.csv", index=False)

                    acc_all = res["correct"].mean()
                    emitted = res["pred"].notna()
                    acc_emitted = res[emitted]["correct"].mean() if emitted.any() else float("nan")
                    tokens_median = res["n_new_tokens"].median()
                    hit_ceiling = (res["n_new_tokens"] >= mnt - 2).mean()
                    log(f"    {split_name}/{regime}: acc_all={acc_all:.3f} "
                        f"acc_emitted={acc_emitted:.3f} emitted={emitted.sum()}/{len(res)} "
                        f"tokens_med={tokens_median:.0f}/{mnt} hit_ceiling={hit_ceiling:.1%}")
                    log(f"    timing[{split_name}/{regime}]: {timing['gen_secs_total']:.1f}s "
                        f"| {timing['tokens_per_sec']:.1f} tok/s "
                        f"| {timing['secs_per_question']:.3f}s/question "
                        f"(batch={timing['infer_batch_size']})")
                    if "reasoning_en" in res and regime == "reasoning":
                        en = res["reasoning_en"].mean()
                        log(f"    language: {en:.0%} reasoned in ENGLISH (native) | "
                            f"{1-en:.0%} in PT (distilled)")
                    if hit_ceiling > 0.10:
                        log(f"    !! WARNING: {hit_ceiling:.0%} hit the {mnt}-token ceiling "
                            f"-> acc_all is underestimated. Consider raising --mnt-reasoning.")

                    if split_name == "abc" and "subset_code" in res:
                        for subset_code, g in res.groupby("subset_code"):
                            log(f"      subset {subset_code}: acc={g['correct'].mean():.3f} "
                                f"(n={len(g)})")

            json.dump(infer_timings, open(run_dir / "infer_timing.json", "w"), indent=2)

            # ---- run configuration ----
            json.dump({
                "technique": tech, "seed": seed, "epochs": args.epochs,
                "lr": args.lr, "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
                "lora_random_state": args.lora_random_state,
                "mnt_answer": args.mnt_answer,
                "mnt_reasoning": args.mnt_reasoning,
                "data_fraction": args.data_fraction,
                "alpha": args.alpha,
                "n_train": len(df_train), "n_test_abc": len(df_abc), "n_test_hard": len(df_hard),
                "infer_regimes": INFER_REGIMES[tech],
                "train_audit": audit,
                "warmup_discarded": not args.no_warmup,
            }, open(run_dir / "run_config.json", "w"), indent=2)

            del model, trainer
            gc.collect()
            torch.cuda.empty_cache()
            (run_dir / "DONE").write_text(datetime.now().isoformat())
            log(f"  OK -> {run_dir}")

    log("ALL RUNS COMPLETE.")


if __name__ == "__main__":
    main()

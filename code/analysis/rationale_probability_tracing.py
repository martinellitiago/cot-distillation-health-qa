#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rationale-prefix restricted-choice probability diagnostic.

This is the executed analysis path behind the paper's probability trajectories.
It saves per-evaluation P(gold) and P(teacher answer) while increasing fractions
of a coherent or cross-question-shuffled rationale are revealed. The paper uses
the frozen Qwen3-4B student as reader and the behavioural A--E logit score.

Public-release invocation (writes arrays to the current directory):

    python code/analysis/rationale_probability_tracing.py \
      --data data/teacher_rationales.jsonl \
      --gold data/questions_2024_2025.jsonl \
      --n 4260 --readers student --methods behavioural

The file retains the optional hidden-state and teacher-reader routines used in
development, but they are not part of the reported manuscript analysis and are
not run by default.
"""
import unsloth  # must precede transformers/torch so Unsloth can patch
from unsloth import FastLanguageModel
import argparse
import ast
import gc
import glob
import os
import re
import numpy as np
import pandas as pd
import torch
import transformers
transformers.logging.set_verbosity_error()

READERS = {"student": "unsloth/Qwen3-4B", "teacher": "unsloth/Qwen3-32B"}
MAX_SEQ = 2048
LETTERS = ["A", "B", "C", "D", "E"]
FRACS = [0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.0]
NBINS = 20
THINK_END_ID = 151668                                  # Qwen3 </think> token id
ANCHOR_OFFSETS = [-16, -12, -8, -4, -2, -1, 0, 1, 2]   # token offsets around the anchor (0 = anchor)

ANS_PATTERNS = [
    re.compile(r'(?:resposta|alternativa|op[çc][ãa]o)\s*(?:correta|final)?[\s:é\-\.]*\(?\s*[A-E]\b.*$',
               re.IGNORECASE | re.DOTALL),
    re.compile(r'\*\*\s*\(?\s*[A-E]\s*\)?\s*\*{0,2}\s*$'),
]


# ---------- data loading (pkl, single parquet, or a directory/glob of parquet shards) ----------
def load_frame(path):
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.parquet")))
        assert files, f"no .parquet in {path}"
        print(f"loading {len(files)} parquet shards from {path}")
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if any(ch in path for ch in "*?[") or path.endswith(".parquet"):
        files = sorted(glob.glob(path))
        if len(files) > 1:
            print(f"loading {len(files)} parquet shards from glob")
            return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        return pd.read_parquet(files[0] if files else path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    if path.endswith(".jsonl"):
        return pd.read_json(path, lines=True)
    return pd.read_pickle(path)


def normalize_frame(df, gold_path=None):
    """Map the parquet schema (output_raw, resposta_professor, no gold) onto what the pipeline
    expects (output, letra, thinking, resposta), attaching the gold answer by enunciado join."""
    df = df.copy()
    # Public-release schema: teacher_rationales.jsonl plus questions JSONL.
    if "rationale" in df.columns and "id" in df.columns:
        assert gold_path, (
            "released teacher_rationales.jsonl requires --gold "
            "data/questions_2024_2025.jsonl"
        )
        questions = load_frame(gold_path).copy()
        if questions["id"].duplicated().any() or df["id"].duplicated().any():
            raise RuntimeError(
                "release ids are not one-to-one; regenerate data with "
                "code/data_prep/build_dataset.py before running this analysis"
            )
        df = df.merge(
            questions[["id", "question_cluster_id", "stem", "options", "gold_answer"]],
            on="id", how="left", validate="one_to_one",
            suffixes=("", "_question"),
        )
        if df["stem"].isna().any():
            raise RuntimeError("one or more rationales did not match a released question id")
        df = df.rename(columns={
            "rationale": "output",
            "teacher_answer": "letra",
            "stem": "enunciado",
            "options": "alternativas",
            "gold_answer": "resposta",
        })
    if "output" not in df.columns and "output_raw" in df.columns:
        df["output"] = df["output_raw"]
    if "letra" not in df.columns and "resposta_professor" in df.columns:
        df["letra"] = df["resposta_professor"]
    if "thinking" not in df.columns:                       # extract CoT if the trace has <think>..</think>
        df["thinking"] = df["output"].astype(str).str.extract(r"(?is)<think>(.*?)</think>", expand=False)
    if "resposta" not in df.columns:                       # gold answer not in the dump -> join it in
        assert gold_path, "no gold 'resposta' column; pass --gold gold_map.csv"
        g = load_frame(gold_path)[["enunciado", "resposta"]].copy()
        g["_k"] = g["enunciado"].astype(str).str.strip()
        g = g.drop_duplicates("_k")
        df["_k"] = df["enunciado"].astype(str).str.strip()
        df = df.merge(g[["_k", "resposta"]], on="_k", how="left").drop(columns="_k")
        miss = df["resposta"].isna().mean()
        print(f"gold join: {(1 - miss) * 100:.1f}% matched" + (f"  ({miss * 100:.1f}% UNMATCHED)" if miss else ""))
    if "letra" in df.columns and "resposta" in df.columns:      # teacher-correct = teacher answer == gold
        df["teacher_correct"] = (df["letra"].astype(str).str.strip().str.upper()
                                 == df["resposta"].astype(str).str.strip().str.upper())
    return df


# ---------- shared text helpers ----------
def strip_answer(text):
    t = re.sub(r"(?i)</?think>", "", str(text)).strip()
    t = re.sub(r"(?is)^\s*racioc[ií]nio\s*:\s*", "", t)     # drop the "Raciocínio:" label prefix
    for rx in ANS_PATTERNS:
        t = rx.sub("", t).strip()
    return re.sub(r"[\s*\-:]+$", "", t).strip()


def parse_alts(alts):
    if isinstance(alts, str):
        alts = ast.literal_eval(alts)
    return {str(k).strip().upper(): v for k, v in alts.items()}


def alts_fmt(alts):
    return "\n".join(f"{k}) {v}" for k, v in sorted(alts.items()))


def gold_answer_text(row):
    alts = parse_alts(row["alternativas"])
    g = str(row["resposta"]).strip().upper()
    return f"{g}) {alts.get(g, g)}"


def prefix_by_frac(text, frac):
    toks = text.split()
    if frac <= 0 or not toks:
        return ""
    return " ".join(toks[:max(1, int(round(len(toks) * frac)))])


# ---------- behavioural (output logits) ----------
def build_answer_prompt(tok, enunciado, af, rationale_prefix):
    # Reader task = "given question + this reasoning, what is the answer?" -> non-thinking direct read.
    body = f"Enunciado:\n{enunciado}\n\nAlternativas:\n{af}\n\n"
    if rationale_prefix.strip():
        body += f"Raciocínio de apoio:\n{rationale_prefix}\n\n"
    body += "Responda apenas com 'Resposta: X' (X = A-E). Não explique."
    p = tok.apply_chat_template([{"role": "user", "content": body}],
                                tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return p + "Resposta:"


@torch.no_grad()
def answer_logprobs(model, tok, prompts, letter_ids, batch=16):
    tok.padding_side = "left"
    out, lid = [], torch.tensor(letter_ids, device=model.device)
    for i in range(0, len(prompts), batch):
        enc = tok(prompts[i:i + batch], return_tensors="pt", padding=True,
                  truncation=True, max_length=MAX_SEQ).to(model.device)
        logits = model(**enc).logits[:, -1, :]
        out.append(torch.log_softmax(logits[:, lid].double(), dim=-1).cpu())   # float64: no exp round-trip
    return torch.cat(out).numpy()


def run_behavioural(model, tok, df, reader, tag):
    af = [alts_fmt(parse_alts(r["alternativas"])) for _, r in df.iterrows()]
    letter_ids = [tok(" " + L, add_special_tokens=False)["input_ids"][-1] for L in LETTERS]
    gold_idx = df["gold"].map({L: i for i, L in enumerate(LETTERS)}).to_numpy()
    L2I = {L: i for i, L in enumerate(LETTERS)}
    n = len(df)
    ridx = np.arange(n)
    teacher_idx = df["letra"].astype(str).str.strip().str.upper().map(lambda x: L2I.get(x, -1)).to_numpy()
    has_t = teacher_idx >= 0
    tc = df.get("teacher_correct", pd.Series([False] * n)).fillna(False).to_numpy().astype(bool)
    groups = [("all", np.ones(n, bool)), ("tc_correct", tc), ("tc_wrong", ~tc)]
    np.save(f"tc_{reader}_{tag}.npy", tc)
    # Unique-question cluster per row. All 4,260 evaluations are retained; 468
    # repeated stems yield 3,792 resampling clusters.
    if "question_cluster_id" in df.columns:
        qid = pd.factorize(df["question_cluster_id"].astype(str))[0]
    else:
        qid = pd.factorize(df["enunciado"].astype(str).str.replace(
            r"\s+", " ", regex=True).str.strip().str.lower())[0]
    np.save(f"qid_{tag}.npy", qid)

    rows, base_pg, base_pt = [], {}, {}
    traj_g = {"coherent": [], "shuffled": []}     # per-item LOG P(gold) trajectory [n, nfracs] (float64)
    traj_t = {"coherent": [], "shuffled": []}     # per-item LOG P(teacher) trajectory [n, nfracs] (float64)
    for cond, col in [("coherent", "rat"), ("shuffled", "rat_shuf")]:
        for frac in FRACS:
            print(f"    [{reader}/{cond}] frac={frac:.2f} ({n} items)...", flush=True)
            prompts = [build_answer_prompt(tok, df["enunciado"].iloc[i], af[i],
                                           prefix_by_frac(df[col].iloc[i], frac)) for i in range(n)]
            lp_all = answer_logprobs(model, tok, prompts, letter_ids)            # [n,5] log-probs (float64)
            p_all = np.exp(lp_all)
            p_gold = p_all[ridx, gold_idx]
            p_teach = np.where(has_t, p_all[ridx, np.clip(teacher_idx, 0, 4)], np.nan)
            traj_g[cond].append(lp_all[ridx, gold_idx])                          # store LOG-prob (rigorous G)
            traj_t[cond].append(np.where(has_t, lp_all[ridx, np.clip(teacher_idx, 0, 4)], np.nan))
            for gname, gm in groups:
                rows.append({"cond": cond, "frac": frac, "group": gname,
                             "p_gold_mean": float(np.nanmean(p_gold[gm])),
                             "p_teacher_mean": float(np.nanmean(p_teach[gm]))})
            if frac == 0.0:
                base_pg[cond], base_pt[cond] = p_gold.copy(), p_teach.copy()
            if frac == 1.0:
                dpg = p_gold - base_pg[cond]       # election score toward the GOLD answer
                dpt = p_teach - base_pt[cond]      # election score toward the TEACHER's answer (faithfulness)
                np.save(f"deltap_gold_{cond}_{reader}_{tag}.npy", dpg)
                np.save(f"deltap_teacher_{cond}_{reader}_{tag}.npy", dpt)
                for gname, gm in groups:
                    print(f"  [behav/{cond}/{gname}] n={int(gm.sum())}  "
                          f"dP(gold)={np.nanmean(dpg[gm]):+.3f}  dP(teacher)={np.nanmean(dpt[gm]):+.3f}")
    # per-item LOG-prob trajectories (float64): rigorous G = (logp_final - logp_base)/ln2; belief = exp(logp)
    for cond in ("coherent", "shuffled"):
        np.save(f"traj_logp_gold_{cond}_{reader}_{tag}.npy", np.stack(traj_g[cond], axis=1))     # [n, nfracs]
        np.save(f"traj_logp_teacher_{cond}_{reader}_{tag}.npy", np.stack(traj_t[cond], axis=1))
    np.save(f"fracs_{tag}.npy", np.array(FRACS))
    res = pd.DataFrame(rows)
    res.to_csv(f"mi_convergence_{reader}_{tag}.csv", index=False)
    piv = res[res.group == "all"].pivot(index="frac", columns="cond", values="p_gold_mean")
    print("  P(gold) trajectory [all]:\n  " + piv.round(3).to_string().replace("\n", "\n  "))


# ---------- geometric (HSIC on hidden states) ----------
def build_context(tok, enunciado, alts, thinking):
    # Faithful to how each dataset was generated (the two came from different pipelines):
    #  - thinking=True  : local batch5 English <think> CoT (EinsteinResearch "quinhentos tokens" prompt)
    #  - thinking=False : pod Portuguese external "Raciocínio:" reasoning (canonical project prompt,
    #    build_user_content 'reasoning' in train_reasoning_models.py). The "Raciocínio:" prime already
    #    lives inside output_raw, so teacher-forcing the stored output reproduces the trace.
    if thinking:
        body = (f"Enunciado:\n{enunciado}\n\nAlternativas:\n{alts_fmt(alts)}\n\n"
                "Você DEVE OBRIGATORIAMENTE terminar seu reasoning com menos de quinhentos tokens "
                "e responder a alternativa correta (APENAS a letra):")
    else:
        body = (f"Questão:\n{enunciado}\n\nAlternativas:\n{alts_fmt(alts)}\n\n"
                "Raciocine de forma breve e objetiva. Finalize obrigatoriamente com 'Resposta: X'.")
    return tok.apply_chat_template([{"role": "user", "content": body}],
                                   tokenize=False, add_generation_prompt=True, enable_thinking=thinking)


@torch.no_grad()
def rationale_hiddens(model, tok, context, rationale, layer, nbins, offsets):
    """One forward over context+rationale; returns hidden states sampled two ways:
       binned   = nbins evenly-spaced positions (0..1)          -> position curve
       anchored = positions at `offsets` around the anchor      -> anchor curve
    Anchor = the </think> token if present (thinking mode), else the last token (no-think mode:
    the conclusion/answer end). This makes the paper's </think>-aligned peak analysis work for the
    no-think data too, which is what Paper 1 was based on."""
    ctx_ids = tok(context, add_special_tokens=False)["input_ids"]
    rat_ids = tok(rationale, add_special_tokens=False)["input_ids"]
    ids = torch.tensor([ctx_ids + rat_ids], device=model.device)[:, :MAX_SEQ]
    hs = model(ids, output_hidden_states=True).hidden_states[layer][0]
    start = min(len(ctx_ids), ids.shape[1] - 1)
    rat_h = hs[start:]
    tr = rat_h.shape[0]
    if tr == 0:
        rat_h, tr = hs[-1:], 1
    bidx = np.clip(np.round((np.arange(nbins) + 0.5) / nbins * (tr - 1)).astype(int), 0, tr - 1)
    binned = rat_h[bidx].float().cpu().numpy()
    rids = rat_ids[:tr]
    anchor = rids.index(THINK_END_ID) if THINK_END_ID in rids else tr - 1
    aidx = [min(max(anchor + o, 0), tr - 1) for o in offsets]
    anchored = rat_h[aidx].float().cpu().numpy()
    return binned, anchored


@torch.no_grad()
def answer_embed(model, tok, text, layer):
    enc = tok(text, add_special_tokens=True, return_tensors="pt").to(model.device)
    return model(**enc, output_hidden_states=True).hidden_states[layer][0].float().mean(0).cpu().numpy()


def _rbf(X):
    sq = (X ** 2).sum(1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0.0)
    med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
    return np.exp(-d2 / (med + 1e-12))


def nhsic(X, Y):
    n = X.shape[0]
    H = np.eye(n) - 1.0 / n
    K, L = H @ _rbf(X) @ H, H @ _rbf(Y) @ H
    return float((K * L).sum() / (np.sqrt((K * K).sum() * (L * L).sum()) + 1e-12))


def run_geometric(model, tok, df, reader, layer, tag):
    Y = np.stack([answer_embed(model, tok, gold_answer_text(r), layer) for _, r in df.iterrows()])
    thinking = (tag == "think")
    ctx = [build_context(tok, r["enunciado"], parse_alts(r["alternativas"]), thinking) for _, r in df.iterrows()]
    rows, arows = [], []
    # geometric method teacher-forces the FULL trace (output, with </think>), not the stripped rationale
    for cond, col in [("coherent", "full"), ("shuffled", "full_shuf")]:
        binned, anchored = [], []
        for i in range(len(df)):
            if i % 100 == 0:
                print(f"    [{reader}/geom/{cond}] {i}/{len(df)} hidden states...", flush=True)
            b, a = rationale_hiddens(model, tok, ctx[i], df[col].iloc[i], layer, NBINS, ANCHOR_OFFSETS)
            binned.append(b); anchored.append(a)
        Hb, Ha = np.stack(binned), np.stack(anchored)
        curve = [nhsic(Hb[:, b, :], Y) for b in range(NBINS)]
        for b, v in enumerate(curve):
            rows.append({"cond": cond, "frac": round((b + 0.5) / NBINS, 3), "nhsic": v})
        peak = (int(np.argmax(curve)) + 0.5) / NBINS
        print(f"  [geom/{cond}] nHSIC mean={np.mean(curve):.4f} peak={max(curve):.4f} at frac~{peak:.2f}")
        acurve = [nhsic(Ha[:, j, :], Y) for j in range(len(ANCHOR_OFFSETS))]
        for off, v in zip(ANCHOR_OFFSETS, acurve):
            arows.append({"cond": cond, "offset": off, "nhsic": v})
        pk = ANCHOR_OFFSETS[int(np.argmax(acurve))]
        print(f"  [geom-anchor/{cond}] peak nHSIC={max(acurve):.4f} at offset {pk:+d}  "
              f"(0 = </think> in think, or conclusion/end in no-think)")
    pd.DataFrame(rows).to_csv(f"mi_hsic_{reader}_{tag}.csv", index=False)
    pd.DataFrame(arows).to_csv(f"mi_hsic_anchor_{reader}_{tag}.csv", index=False)


# ---------- transfer gap ----------
def transfer_gap(tag):
    try:
        for cond in ("coherent", "shuffled"):
            s = np.load(f"deltap_gold_{cond}_student_{tag}.npy")
            t = np.load(f"deltap_gold_{cond}_teacher_{tag}.npy")
            g = t - s
            print(f"  [gap/{cond}] dP_student={np.nanmean(s):+.3f} dP_teacher={np.nanmean(t):+.3f} "
                  f"gap={np.nanmean(g):+.3f} (frac gap>0={np.nanmean(g > 0):.1%})")
    except FileNotFoundError:
        print("  (transfer gap skipped: need behavioural runs for BOTH readers)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="use the THINKING pkl: qwen32_qa_thinkings_THINK.pkl")
    ap.add_argument("--rationale-col", default="thinking",
                    help="'thinking' = leak-free CoT (no answer letter); behavioural method feeds this")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=8)
    ap.add_argument("--readers", default="student", help="comma list: student,teacher")
    ap.add_argument("--methods", default="behavioural", help="comma list: behavioural,geometric")
    ap.add_argument("--layer", type=int, default=-1)
    ap.add_argument("--tag", default=None,
                    help="suffix for output files so think/no-think runs don't collide; "
                         "default 'think' if the pkl has a </think> trace, else 'nothink'")
    ap.add_argument("--gold", default=None,
                    help="file (csv/parquet/pkl) with enunciado+resposta to attach the gold answer "
                         "when the data has none (the parquet dumps only have the model's answer)")
    args = ap.parse_args()
    readers = [r.strip() for r in args.readers.split(",") if r.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    df = normalize_frame(load_frame(args.data), args.gold)
    if args.rationale_col not in df.columns:
        args.rationale_col = "output"
    if args.rationale_col == "thinking" and df["thinking"].isna().mean() > 0.5:
        print("no </think> traces -> using 'output' as rationale (no-think mode)")
        args.rationale_col = "output"
    if args.tag is None:
        args.tag = "think" if df["output"].astype(str).str.contains("</think>").mean() > 0.5 else "nothink"
    df = df.sample(n=min(args.n, len(df)), random_state=args.seed).reset_index(drop=True)
    df["gold"] = df["resposta"].astype(str).str.strip().str.upper()
    df = df[df["gold"].isin(LETTERS)].reset_index(drop=True)
    df["rat"] = df[args.rationale_col].map(strip_answer)          # behavioural: leak-free CoT
    df["full"] = df["output"].astype(str)                          # geometric: full trace with </think>
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(df))
    for k in range(len(df)):
        if perm[k] == k:
            perm[k], perm[(k + 1) % len(df)] = perm[(k + 1) % len(df)], perm[k]
    df["rat_shuf"] = [df["rat"].iloc[perm[k]] for k in range(len(df))]
    df["full_shuf"] = [df["full"].iloc[perm[k]] for k in range(len(df))]
    print(f"N={len(df)}  readers={readers}  methods={methods}  layer={args.layer}")

    for reader in readers:
        load_4bit = (reader == "teacher")
        print(f"\n=== reader={reader}  model={READERS[reader]}  4bit={load_4bit} ===")
        model, tok = FastLanguageModel.from_pretrained(model_name=READERS[reader], max_seq_length=MAX_SEQ,
                                                       dtype=None, load_in_4bit=load_4bit)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        FastLanguageModel.for_inference(model)
        if "behavioural" in methods:
            run_behavioural(model, tok, df, reader, args.tag)
        if "geometric" in methods:
            run_geometric(model, tok, df, reader, args.layer, args.tag)
        del model, tok
        gc.collect()
        torch.cuda.empty_cache()

    if "behavioural" in methods and {"student", "teacher"} <= set(readers):
        print(f"\n=== transfer gap (I_teacher - I_student), tag={args.tag} ===")
        transfer_gap(args.tag)
    saved = [f"mi_convergence_*_{args.tag}.csv", f"deltap_*_{args.tag}.npy", f"tc_*_{args.tag}.npy",
             f"traj_logp_gold/teacher_*_{args.tag}.npy (per-item log-probs, float64)", f"fracs_{args.tag}.npy"]
    if "geometric" in methods:
        saved += [f"mi_hsic_*_{args.tag}.csv", f"mi_hsic_anchor_*_{args.tag}.csv"]
    print(f"\nDONE (tag={args.tag}, methods={methods}). saved: " + ", ".join(saved))


if __name__ == "__main__":
    main()

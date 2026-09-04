# Fast lexical-leakage audit

Run this on the Linux environment that contains the 4,260 aligned item rows and
the stored probability trajectories. It is CPU-only and normally takes seconds
to a few minutes; bootstrap time depends on `--bootstrap`.

First verify the table before running the audit:

```bash
cd ~/hiae-ellme-distill
python - <<'PY'
import pandas as pd
p = "data/qwen32_qa_thinkings.pkl"
d = pd.read_pickle(p)
print("type:", type(d))
print("rows:", len(d))
print("columns:", list(d.columns))
PY
```

Proceed with the example below only if the table has 4,260 aligned rows and
contains the question, alternatives, gold answer, and rationale. Otherwise,
send the preflight output back so the correct source can be joined by
normalised question stem; do not assume positional alignment.

Example:

```bash
python analysis/label_leakage_audit.py \
  --items data/qwen32_qa_thinkings.pkl \
  --teacher-correct tc_student_nothink.npy \
  --coherent traj_gold_coherent_student_nothink.npy \
  --shuffled traj_gold_shuffled_student_nothink.npy \
  --cluster-ids question_cluster_ids_nothink.npy \
  --bootstrap 2000 \
  --out-dir label_leakage_audit
```

If the rationale or item columns have different names, pass them explicitly:

```bash
  --question-col enunciado \
  --options-col alternativas \
  --gold-col resposta \
  --rationale-col target_thinking
```

The item table must have the same row order as the trajectory arrays. If
`question_cluster_ids_nothink.npy` has another name, use the cluster-ID file
written by the improved `run_mi_pid.py`. If it is omitted, the script derives
clusters from normalised question stems.

Outputs:

- `label_leakage_items.csv`: one row per evaluation;
- `label_leakage_effect_sensitivity.csv`: main coherent-minus-shuffled effects
  before and after exclusions;
- `label_leakage_summary.json`: manuscript-ready summary values.

The defensible manuscript claim is that the effect did or did not persist after
excluding overt lexical cues. Do not claim that this audit rules out semantic
paraphrase or validates rationale factuality.

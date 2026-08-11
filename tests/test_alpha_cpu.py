"""The alpha weighting is the mechanism behind one of the paper's claims, so it
gets a test that runs on CPU, with a tokenizer only -- no GPU, no training.

What it guards:

  * `distill_sft` weights the rationale TOKENS and leaves the answer at 1.0;
  * `step_by_step` weights the whole rationale TASK (Hsieh's lambda) -- a
    different quantity that must not silently become the token-level one;
  * at alpha = 1.0 every target token weighs exactly 1.0, so the weighted loss
    reduces to the unweighted one and the two arms remain comparable;
  * prompt tokens never carry loss weight, whatever alpha is.

The historical failure this catches: the truncation recursion in tokenize_example
used to call itself WITHOUT passing alpha, so any example long enough to be
shortened silently lost its weighting. If the weighted-token counts come back far
below the rationale length, that bug is back.
"""
import numpy as np
import pytest

from conftest import load

TECHNIQUES = ["distill_sft", "step_by_step"]


@pytest.fixture(scope="module")
def stage2():
    try:
        return load("stage2")
    except ImportError as e:
        pytest.skip(f"pipeline dependencies unavailable: {e}")


@pytest.fixture(scope="module")
def tokenizer(stage2):
    transformers = pytest.importorskip("transformers")
    tok = transformers.AutoTokenizer.from_pretrained(stage2.MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


@pytest.fixture(scope="module")
def sample_rows():
    """A tiny synthetic split, so the test needs no data files."""
    import pandas as pd
    return pd.DataFrame([
        {"enunciado": "Paciente com febre há três dias. Qual a conduta?",
         "alternativas": {"A": "Observar", "B": "Antibiótico", "C": "Internar",
                          "D": "Exames", "E": "Alta"},
         "resposta": "B",
         "target_thinking": "A febre persistente sugere infecção bacteriana. "
                            "Portanto o antibiótico é indicado nesta situação clínica."},
        {"enunciado": "Qual exame confirma o diagnóstico?",
         "alternativas": {"A": "Hemograma", "B": "Raio-X", "C": "Cultura",
                          "D": "Ultrassom", "E": "Nenhum"},
         "resposta": "C",
         "target_thinking": None},          # no rationale -> answer-only target
    ])


def target_weights(example):
    """The loss weights on the target tokens only (prompt positions excluded)."""
    weights = np.array(example["weights"])
    labels = np.array(example["labels"])
    return weights[labels != -100]


@pytest.mark.parametrize("technique", TECHNIQUES)
def test_alpha_one_leaves_every_target_token_at_weight_one(stage2, tokenizer,
                                                          sample_rows, technique):
    rows, _ = stage2.build_dataset(sample_rows.copy(), tokenizer, technique,
                                   stage2.MAX_SEQ_LEN, alpha=1.0)
    for ex in rows:
        w = target_weights(ex)
        assert len(w) > 0, "an example ended up with no loss tokens"
        assert np.allclose(w, 1.0), \
            f"{technique} at alpha=1.0 produced weights {sorted(set(w))}"


def test_distill_sft_weights_the_rationale_tokens_not_the_answer(stage2, tokenizer,
                                                                 sample_rows):
    """Token-level: inside ONE target, the rationale gets alpha and the answer 1.0."""
    alpha = 0.1
    rows, audit = stage2.build_dataset(sample_rows.copy(), tokenizer, "distill_sft",
                                       stage2.MAX_SEQ_LEN, alpha=alpha)
    assert audit["with_rationale"] == 1, "the fixture should give exactly one rationale"

    weighted = [ex for ex in rows if (target_weights(ex) < 1.0).any()]
    assert len(weighted) == 1, \
        "exactly the one example carrying a rationale should be down-weighted"

    w = target_weights(weighted[0])
    assert set(np.round(np.unique(w), 6)) == {alpha, 1.0}, \
        f"expected weights {{{alpha}, 1.0}}, got {sorted(set(np.round(w, 6)))}"
    assert (w < 1.0).sum() > 5, \
        "too few down-weighted tokens: the rationale span was not found (or the " \
        "truncation recursion dropped alpha)"
    assert w[-1] == 1.0, "the answer tokens at the end must keep weight 1.0"


def test_step_by_step_weights_the_whole_rationale_task(stage2, tokenizer, sample_rows):
    """Example-level: the rationale TASK is scaled, the label task is not.

    This is Hsieh's lambda, and it is a different quantity from the token weight
    above -- conflating them would misreport what the alpha sweep measured."""
    alpha = 0.1
    rows, audit = stage2.build_dataset(sample_rows.copy(), tokenizer, "step_by_step",
                                       stage2.MAX_SEQ_LEN, alpha=alpha)
    assert audit["rationale_task"] == 1
    assert audit["answer_only"] == len(sample_rows), "every item needs a label task"

    scaled = [ex for ex in rows if np.allclose(target_weights(ex), alpha)]
    unscaled = [ex for ex in rows if np.allclose(target_weights(ex), 1.0)]
    assert len(scaled) == 1, "exactly the rationale task should be scaled"
    assert len(unscaled) == len(sample_rows), "the label tasks must stay at 1.0"
    # the whole task is uniformly scaled -- no mixed weights inside one example
    assert len(set(np.round(target_weights(scaled[0]), 6))) == 1


@pytest.mark.parametrize("technique", TECHNIQUES)
@pytest.mark.parametrize("alpha", [1.0, 0.3, 0.1])
def test_prompt_tokens_never_carry_loss(stage2, tokenizer, sample_rows,
                                        technique, alpha):
    rows, _ = stage2.build_dataset(sample_rows.copy(), tokenizer, technique,
                                   stage2.MAX_SEQ_LEN, alpha=alpha)
    for ex in rows:
        weights = np.array(ex["weights"])
        labels = np.array(ex["labels"])
        assert np.allclose(weights[labels == -100], 0.0), \
            "a masked prompt token carries a non-zero loss weight"

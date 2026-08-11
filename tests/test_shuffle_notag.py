"""The derangement is the whole causal argument of Stage 1.

If a single item keeps its own rationale, the 'shuffled' arm is quietly a little
bit coherent, and the collapse it is supposed to demonstrate is understated by
exactly that much. These tests run on CPU in under a second and should be run
before any GPU job.

The historical concern: fixed points are repaired by swapping perm[k] with
perm[k+1], and the wrap-around at the last position can in principle leave one
behind. The production code asserts this at runtime; here we hunt for a seed that
triggers it.
"""
import numpy as np
import pytest

from conftest import load

pytest.importorskip("numpy")


@pytest.fixture(scope="module")
def stage1():
    try:
        return load("stage1")
    except ImportError as e:      # unsloth/trl absent on a CPU-only machine
        pytest.skip(f"pipeline dependencies unavailable: {e}")


def test_no_item_keeps_its_own_rationale_on_the_canonical_seeds(stage1):
    """The ten canonical seeds, at the shuffle seed actually used (123)."""
    eligible = list(range(400))
    for split_seed in [8, 12, 17, 23, 25, 31, 37, 44, 52, 61]:
        rng = np.random.default_rng(123 + split_seed)
        perm = stage1.derange(eligible, rng)
        assert not any(perm[k] == eligible[k] for k in range(len(eligible))), \
            f"self-assignment at split seed {split_seed}"


@pytest.mark.parametrize("n", [2, 3, 5, 17, 100, 913])
def test_derangement_holds_over_many_seeds_and_sizes(stage1, n):
    """Sweep seeds broadly: a true derangement must survive all of them, and any
    seed that raises DERANGEMENT BROKEN is a seed that must not be used."""
    eligible = list(range(n))
    for seed in range(300):
        rng = np.random.default_rng(seed)
        perm = stage1.derange(eligible, rng)
        assert not any(perm[k] == eligible[k] for k in range(n)), \
            f"self-assignment with n={n}, seed={seed}"


def test_derangement_is_a_permutation(stage1):
    """Every rationale must be used exactly once -- otherwise some item's
    rationale is duplicated and another's is dropped, which changes the training
    distribution rather than only its relevance."""
    eligible = list(range(250))
    for seed in range(50):
        rng = np.random.default_rng(seed)
        perm = stage1.derange(eligible, rng)
        assert sorted(perm) == sorted(eligible)


def test_derangement_is_reproducible(stage1):
    """Same seed, same assignment -- otherwise a rerun is not the same experiment."""
    eligible = list(range(120))
    a = stage1.derange(eligible, np.random.default_rng(123 + 8))
    b = stage1.derange(eligible, np.random.default_rng(123 + 8))
    assert list(a) == list(b)


def test_different_splits_get_different_derangements(stage1):
    """The shuffle is seeded with shuffle_seed + split_seed so that each split
    gets its own assignment; if two splits shared one, the control would be
    correlated across splits."""
    eligible = list(range(120))
    a = stage1.derange(eligible, np.random.default_rng(123 + 8))
    b = stage1.derange(eligible, np.random.default_rng(123 + 12))
    assert list(a) != list(b)


def test_single_eligible_item_is_not_shuffled(stage1):
    """With one eligible item a derangement does not exist; the caller guards
    this with `len(eligible) > 1`. Document the boundary so nobody removes it."""
    eligible = [0]
    with pytest.raises(RuntimeError, match="DERANGEMENT BROKEN"):
        stage1.derange(eligible, np.random.default_rng(0))

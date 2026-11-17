"""Statistics students implement (M2 judge calibration, M4 paired bootstrap).

Rules: a judge is a classifier -- measure it like one. A higher mean is not
"better" until the paired CI of the difference excludes 0.
Reference values to self-check are in tests/test_lab9_stats.py.
"""

from __future__ import annotations

from collections.abc import Sequence

POS = "FAIL"   # the judge's job is to FIND failures, so FAIL is the positive class


def confusion(human: Sequence[str], judge: Sequence[str]) -> dict[str, int]:
    """tp/fp/fn/tn with FAIL as positive."""
    tp = fp = fn = tn = 0
    for h, j in zip(human, judge):
        h_pos = (h == POS)
        j_pos = (j == POS)
        if h_pos and j_pos:
            tp += 1
        elif not h_pos and j_pos:
            fp += 1
        elif h_pos and not j_pos:
            fn += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def precision_recall(human: Sequence[str], judge: Sequence[str]) -> tuple[float, float]:
    # TODO M2: precision = tp/(tp+fp), recall = tp/(tp+fn); return 0.0 when the denominator is 0
    conf = confusion(human, judge)
    tp, fp, fn = conf["tp"], conf["fp"], conf["fn"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall


def cohen_kappa(human: Sequence[str], judge: Sequence[str]) -> float:
    """kappa = (p_o - p_e) / (1 - p_e). Raise ValueError when undefined (p_e == 1):
    a constant judge has no signal and must FAIL calibration, not pass on a technicality."""
    conf = confusion(human, judge)
    tp, fp, fn, tn = conf["tp"], conf["fp"], conf["fn"], conf["tn"]
    total = tp + fp + fn + tn
    if total == 0:
        raise ValueError("kappa undefined: empty sequences")
    p_o = (tp + tn) / total
    judge_pos = tp + fp
    judge_neg = tn + fn
    human_pos = tp + fn
    human_neg = tn + fp
    p_e = (judge_pos * human_pos + judge_neg * human_neg) / (total * total)
    if abs(1.0 - p_e) < 1e-12:
        raise ValueError("kappa undefined for constant judge (p_e == 1)")
    return (p_o - p_e) / (1.0 - p_e)


def rogan_gladen(observed_rate: float, sensitivity: float, specificity: float) -> float:
    """Correct an observed failure rate for judge error; clip to [0, 1]."""
    # TODO M2: (p_obs + spec - 1) / (sens + spec - 1)
    denom = sensitivity + specificity - 1.0
    if abs(denom) < 1e-12:
        val = observed_rate
    else:
        val = (observed_rate + specificity - 1.0) / denom
    return max(0.0, min(1.0, float(val)))


def paired_bootstrap(a: Sequence[float], b: Sequence[float], *, n_resamples: int = 1000,
                     rng: int = 2027, confidence: float = 0.95) -> tuple[float, float, float]:
    """a[i], b[i] = per-item mean over seeds, SAME items, SAME order.
    Return (mean(a)-mean(b), lo, hi) with scipy.stats.bootstrap(paired=True, method="percentile",
    rng=numpy.random.default_rng(rng)). If a == b everywhere return (0.0, 0.0, 0.0)."""
    # TODO M4
    if len(a) != len(b):
        raise ValueError(f"Length mismatch: {len(a)} != {len(b)}")
    if not a:
        return (0.0, 0.0, 0.0)
    if list(a) == list(b):
        return (0.0, 0.0, 0.0)

    import numpy as np
    from scipy.stats import bootstrap

    arr_a = np.array(a, dtype=float)
    arr_b = np.array(b, dtype=float)
    diff = float(np.mean(arr_a) - np.mean(arr_b))

    def diff_mean(x, y, axis=-1):
        return np.mean(x, axis=axis) - np.mean(y, axis=axis)

    res = bootstrap(
        (arr_a, arr_b),
        diff_mean,
        paired=True,
        method="percentile",
        n_resamples=n_resamples,
        random_state=np.random.default_rng(rng),
        confidence_level=confidence,
    )
    lo = float(res.confidence_interval.low)
    hi = float(res.confidence_interval.high)
    return (diff, lo, hi)


def bootstrap_ci(data: Sequence[float], num_samples: int = 1000, ci: float = 0.95) -> tuple[float, float]:
    """Calculate (lower_bound, upper_bound) for the mean of data via bootstrap resampling."""
    if not data:
        return (0.0, 0.0)
    import numpy as np
    from scipy.stats import bootstrap

    arr = np.array(data, dtype=float)
    if np.all(arr == arr[0]):
        return (float(arr[0]), float(arr[0]))
    res = bootstrap(
        (arr,),
        np.mean,
        method="percentile",
        n_resamples=num_samples,
        confidence_level=ci,
    )
    return (float(res.confidence_interval.low), float(res.confidence_interval.high))


def pass_at_k(n: int, c: int, k: int) -> float:
    """Probability that AT LEAST ONE of k trials drawn from n (c of them passing) passes.
    Unbiased estimator (Chen et al. 2021): 1 - C(n-c, k) / C(n, k). Raise ValueError if k > n."""
    import math

    if not (0 <= c <= n and 1 <= k <= n):
        raise ValueError(f"Invalid args: n={n}, c={c}, k={k}")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def pass_hat_k(n: int, c: int, k: int) -> float:
    """pass^k (tau-bench, Yao et al. 2024): probability that ALL k trials drawn from n pass.
    Unbiased estimator: C(c, k) / C(n, k). Raise ValueError if k > n.
    pass^k <= pass@1 <= pass@k; a reliable agent keeps pass^k close to pass@1."""
    import math

    if not (0 <= c <= n and 1 <= k <= n):
        raise ValueError(f"Invalid args: n={n}, c={c}, k={k}")
    if c < k:
        return 0.0
    return math.comb(c, k) / math.comb(n, k)

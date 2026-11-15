"""Candidate-scoring policies: fixed-budget (AutoND), adaptive (confidence-guided), ablations.

Terminology
-----------
* candidate  : an input difference delta (single-key, plain_bits wide).
* sample     : one random (plaintext, key) for one round count r.
* b~_r(delta): AutoND bias score at round r, mean_j |0.5 - p^_j| over output-difference bits.
* S(delta)   : weighted score sum_r w_r * b~_r(delta). AutoND's re-evaluation step uses w_r = r.

Cost model (identical for every policy)
---------------------------------------
For every batch of ``b`` samples at round ``r`` the reference ciphertexts C0 cost ``b``
encryptions and each scored candidate costs another ``b`` encryptions (C1), exactly like
AutoND's ``evaluate_multiple_differences`` which shares (P, K, C0) across candidates.
The ledger counts block encryptions and round-function evaluations (encryptions x r).
"""

from __future__ import annotations

import math
import os
import pickle
from dataclasses import dataclass, field, asdict

import numpy as np

from .data import random_bits

# ----------------------------------------------------------------------------
# Ledger
# ----------------------------------------------------------------------------


@dataclass
class Ledger:
    encryptions: int = 0
    round_evaluations: int = 0

    def add(self, n_encryptions: int, rounds: int) -> None:
        self.encryptions += int(n_encryptions)
        self.round_evaluations += int(n_encryptions) * int(rounds)

    def as_dict(self) -> dict:
        return asdict(self)


def batch_cost(n_candidates: int, batch: int, n_rounds: int) -> int:
    """Encryptions needed to score n_candidates on `batch` samples at each of n_rounds rounds."""
    return n_rounds * batch * (n_candidates + 1)


# ----------------------------------------------------------------------------
# Samplers
# ----------------------------------------------------------------------------


class CipherSampler:
    """Draws output-difference bits for many candidates with common random numbers."""

    shares_inputs = True  # all candidates in one draw use the same (P, K, C0)

    def __init__(self, cipher, rng: np.random.Generator, ledger: Ledger | None = None):
        self.cipher = cipher
        self.n_bits = cipher.plain_bits
        self.rng = rng
        self.ledger = ledger if ledger is not None else Ledger()

    def draw(self, diffs: np.ndarray, batch: int, nr: int) -> np.ndarray:
        """Return uint8 array (m, batch, n_bits) of C(P) xor C(P xor delta)."""
        c = self.cipher
        m = len(diffs)
        keys = random_bits(self.rng, batch, c.key_bits)
        pt0 = random_bits(self.rng, batch, c.plain_bits)
        c0 = c.encrypt(pt0, keys, nr)
        self.ledger.add(batch, nr)
        if m == 0:
            return np.zeros((0, batch, self.n_bits), np.uint8)
        pt1 = (diffs[:, None, :] ^ pt0[None, :, :]).reshape(-1, c.plain_bits)
        keys1 = np.broadcast_to(keys[None], (m,) + keys.shape).reshape(-1, c.key_bits)
        c1 = c.encrypt(pt1, keys1, nr)
        self.ledger.add(m * batch, nr)
        return (c1.reshape(m, batch, -1) ^ c0[None]).astype(np.uint8)


class SyntheticSampler:
    """Toy sampler for tests: candidate i's bit j at round r is Bernoulli(p[i, r, j]).

    The exact score of candidate i at round r is mean_j |0.5 - p[i, r, j]|, so policies
    can be checked against known ground truth.
    """

    shares_inputs = False

    def __init__(self, p: np.ndarray, rounds: list[int], rng: np.random.Generator,
                 ledger: Ledger | None = None):
        self.p = np.asarray(p, dtype=float)  # (m_total, n_rounds, n_bits)
        self.rounds = list(rounds)
        self.n_bits = self.p.shape[2]
        self.rng = rng
        self.ledger = ledger if ledger is not None else Ledger()

    def draw(self, diffs: np.ndarray, batch: int, nr: int) -> np.ndarray:
        # Here `diffs` holds candidate *indices* (shape (m, 1)) into self.p.
        idx = np.asarray(diffs)[:, 0].astype(int)
        ri = self.rounds.index(nr)
        self.ledger.add(batch * (len(idx) + 1), nr)
        u = self.rng.random((len(idx), batch, self.n_bits))
        return (u < self.p[idx, ri][:, None, :]).astype(np.uint8)

    def exact_scores(self, weights) -> np.ndarray:
        per_round = np.mean(np.abs(0.5 - self.p), axis=2)  # (m, R)
        return per_round @ np.asarray(weights, float)


# ----------------------------------------------------------------------------
# Scores and intervals
# ----------------------------------------------------------------------------


def bias_from_bits(bits: np.ndarray) -> np.ndarray:
    """AutoND bias score from bits (..., t, n) -> (...,)."""
    return np.mean(np.abs(0.5 - bits.mean(axis=-2)), axis=-1)


def bias_from_counts(counts: np.ndarray, t) -> np.ndarray:
    """counts (..., n) of ones over t samples -> AutoND bias score."""
    t = np.asarray(t, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = counts / t[..., None]
    return np.mean(np.abs(0.5 - p), axis=-1)


def autond_weights(rounds) -> np.ndarray:
    """AutoND's weighted score uses weight r for round r (optimizer.optimize)."""
    return np.asarray(rounds, dtype=float)


def bootstrap_weighted(bits_by_round: list[np.ndarray], weights, B: int,
                       rng: np.random.Generator, chunk: int = 32) -> np.ndarray:
    """Poisson bootstrap of the weighted score.

    bits_by_round[r]: uint8 (c, t_r, n). Returns (c, B) bootstrap replicates of S.
    The same resampling weights are shared by all candidates (paired resampling,
    consistent with common random numbers).
    """
    c = bits_by_round[0].shape[0]
    out = np.zeros((c, B), dtype=np.float64)
    for w_r, bits in zip(weights, bits_by_round):
        t, n = bits.shape[1], bits.shape[2]
        W = rng.poisson(1.0, size=(B, t)).astype(np.float32)
        denom = np.maximum(W.sum(axis=1), 1.0)[:, None]  # (B, 1)
        for s in range(0, c, chunk):
            X = bits[s:s + chunk].astype(np.float32)           # (cc, t, n)
            cc = X.shape[0]
            P = (W @ X.transpose(1, 0, 2).reshape(t, cc * n)) / denom  # (B, cc*n)
            sc = np.abs(0.5 - P.reshape(B, cc, n)).mean(axis=2)        # (B, cc)
            out[s:s + cc] += w_r * sc.T
    return out


def intervals(point: np.ndarray, boot: np.ndarray | None, t_by_round, weights,
              method: str, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (lower, upper) bounds on the *exact* weighted score.

    bias_bounded (default):
                 upper = S + z * SE,   lower = S - z * SE - sum_r w_r * 0.5 / sqrt(t_r).
                 SE is the bootstrap standard deviation of S. Justification: by Jensen,
                 E b~_r >= b_r, so only sampling noise can push S below b (upper side);
                 and E b~_r - b_r <= mean_j E|p^_j - p_j| <= 0.5/sqrt(t_r) (lower side).
                 Only the fluctuation term relies on a normal approximation. In repeated
                 trials (tests/test_scoring.py, experiments/phase4_validate_scorer.py) it
                 reaches at least nominal coverage, while basic/percentile do not.
    basic      : basic (reverse-percentile) bootstrap; corrects the upward bias of the
                 |.| statistic. Empirical -- no coverage guarantee under adaptive stopping.
    percentile : percentile bootstrap (kept for comparison; biased upwards for noise bits).
    mcdiarmid  : distribution-free bound. b~_r changes by <= 1/t_r when one sample changes,
                 so P(|S - E S| >= eps) <= 2 exp(-2 eps^2 / sum_r w_r^2 / t_r); and
                 0 <= E b~_r - b_r <= 0.5/sqrt(t_r). Valid per look; the caller divides alpha
                 by (#candidates x #looks) for a union bound.
    """
    w = np.asarray(weights, float)
    t = np.asarray(t_by_round, float)  # (c, R)
    if method == "bias_bounded":
        from scipy.stats import norm
        z = float(norm.ppf(1 - alpha / 2))
        se = boot.std(axis=1, ddof=1)
        floor = np.sum(w[None] * 0.5 / np.sqrt(t), axis=1)
        return point - z * se - floor, point + z * se
    if method in ("basic", "percentile"):
        lo_q, hi_q = np.quantile(boot, [alpha / 2, 1 - alpha / 2], axis=1)
        if method == "percentile":
            return lo_q, hi_q
        return 2 * point - hi_q, 2 * point - lo_q
    if method == "mcdiarmid":
        var_term = np.sum(w[None] ** 2 / t, axis=1)
        eps = np.sqrt(var_term * math.log(2 / alpha) / 2)
        bias = np.sum(w[None] * 0.5 / np.sqrt(t), axis=1)
        return point - eps - bias, point + eps
    raise ValueError(f"unknown interval method {method!r}")


# ----------------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------------


@dataclass
class ScoringResult:
    policy: str
    diffs: list[int]
    rounds: list[int]
    weights: list[float]
    scores: np.ndarray            # weighted point estimate per candidate
    per_round: np.ndarray         # (m, R) bias score per round
    samples: np.ndarray           # (m, R) samples used per candidate and round
    shortlist: list[int]          # top-K diffs (ints), best first
    ledger: dict
    status: list[str] = field(default_factory=list)
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None
    history: list[dict] = field(default_factory=list)
    config: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        d = {k: v for k, v in asdict(self).items()}
        for k in ("scores", "per_round", "samples", "lower", "upper"):
            if d[k] is not None:
                d[k] = np.asarray(d[k]).tolist()
        d["diffs_hex"] = [hex(x) for x in self.diffs]
        d["shortlist_hex"] = [hex(x) for x in self.shortlist]
        return d


def top_k(diffs: list[int], scores: np.ndarray, k: int, mask: np.ndarray | None = None) -> list[int]:
    s = np.where(mask, scores, -np.inf) if mask is not None else scores
    # stable tie-break: higher score first, then smaller difference value
    order = sorted(range(len(diffs)), key=lambda i: (-s[i], diffs[i]))
    return [diffs[i] for i in order[:k] if np.isfinite(s[i])]


def _as_bits(diffs: list[int], n_bits: int) -> np.ndarray:
    from .data import ints_to_bit_matrix
    return ints_to_bit_matrix(diffs, n_bits)


# ----------------------------------------------------------------------------
# Policy A: fixed budget per candidate (AutoND re-evaluation step)
# ----------------------------------------------------------------------------


def score_fixed(sampler, diffs: list[int], rounds: list[int], t: int, k: int,
                weights=None, chunk_candidates: int = 64, diff_bits: np.ndarray | None = None,
                policy_name: str = "A_fixed", statistic: str = "bias") -> ScoringResult:
    """Every candidate gets the same t samples per round (shared P, K, C0 per round).

    Note: AutoND draws one (P, K) set per round and scores *all* candidates on it; we
    do the same, but chunk the candidates only for memory -- the (P, K) set is shared
    across chunks, so this is numerically identical to scoring them all at once.
    """
    weights = autond_weights(rounds) if weights is None else np.asarray(weights, float)
    D = _as_bits(diffs, sampler.n_bits) if diff_bits is None else diff_bits
    m, R = len(diffs), len(rounds)
    per_round = np.zeros((m, R))
    start = dict(sampler.ledger.as_dict())
    for ri, nr in enumerate(rounds):
        # Replay the RNG state per chunk so every chunk sees the same (P, K, C0).
        state = sampler.rng.bit_generator.state
        for s in range(0, m, chunk_candidates):
            if getattr(sampler, "shares_inputs", False):
                sampler.rng.bit_generator.state = state
            bits = sampler.draw(D[s:s + chunk_candidates], t, nr)
            if statistic == "spectral":
                from .spectral import cap12_and_se
                per_round[s:s + chunk_candidates, ri] = cap12_and_se(bits)[0]
            else:
                per_round[s:s + chunk_candidates, ri] = bias_from_bits(bits)
        # count C0 only once per round: undo the duplicate C0 charges from extra chunks
        n_chunks = math.ceil(m / chunk_candidates)
        if n_chunks > 1:
            sampler.ledger.encryptions -= (n_chunks - 1) * t
            sampler.ledger.round_evaluations -= (n_chunks - 1) * t * nr
    per_round[np.asarray(diffs) == 0] = 0.0  # AutoND sets the zero difference to score 0
    scores = per_round @ weights
    used = {k_: sampler.ledger.as_dict()[k_] - start[k_] for k_ in start}
    return ScoringResult(policy=policy_name, diffs=list(diffs), rounds=list(rounds),
                         weights=weights.tolist(), scores=scores, per_round=per_round,
                         samples=np.full((m, R), t), shortlist=top_k(diffs, scores, k),
                         ledger=used, status=["scored"] * m,
                         config={"t": t, "k": k, "statistic": statistic})


def max_uniform_t(budget: int, m: int, n_rounds: int) -> int:
    return budget // (n_rounds * (m + 1))


# ----------------------------------------------------------------------------
# Policy B: confidence-guided adaptive scoring
# ----------------------------------------------------------------------------


@dataclass
class AdaptiveConfig:
    k: int = 8                    # shortlist size
    t0: int = 100                 # samples per round in the first stage
    growth: float = 2.0           # cumulative samples multiply by this each stage
    t_max: int = 10_000           # per-candidate per-round cap (AutoND paper value)
    budget: int | None = None     # hard global cap on encryptions (None = no cap)
    interval: str = "bias_bounded"  # bias_bounded | mcdiarmid | basic | percentile
    alpha: float = 0.01           # per-interval miscoverage (mcdiarmid: before union bound)
    n_boot: int = 200
    tol: float = 0.0              # stop sampling once noise width (U-L minus floor) <= tol * S_K
    accept_clear: bool = True     # stop sampling candidates confidently inside the top-K
    min_batch: int = 10
    statistic: str = "bias"       # bias (AutoND score) | spectral (unbiased C_{<=2}, delta-method SE)

    def stage_targets(self) -> list[int]:
        ts, t = [], self.t0
        while t < self.t_max:
            ts.append(int(t))
            t = int(math.ceil(t * self.growth))
        ts.append(self.t_max)
        return ts


ACTIVE, ELIM, ACCEPT, PRECISE = "active", "eliminated", "accepted", "precise"


class AdaptiveScorer:
    """Successive elimination with confidence intervals at pre-declared checkpoints.

    At each checkpoint all *active* candidates receive the same new batch (common random
    numbers), intervals are recomputed, and a candidate:
      * is eliminated if at least K other non-eliminated candidates have L_o > U_a;
      * is accepted (if accept_clear) if fewer than K others have U_o > L_a;
      * is frozen as 'precise' if its noise width <= tol * S_K (S_K = K-th best point
        estimate): further samples could only reorder it among practical ties.
    Sampling stops when no candidate is active, t_max is reached, or the global budget
    would be exceeded. The shortlist is the top-K non-eliminated candidates by point
    estimate. The final test set is never touched here.
    """

    def __init__(self, sampler, diffs: list[int], rounds: list[int], cfg: AdaptiveConfig,
                 weights=None, diff_bits: np.ndarray | None = None, boot_rng=None,
                 checkpoint_path: str | None = None, policy_name: str = "B_adaptive"):
        self.sampler, self.cfg = sampler, cfg
        self.diffs, self.rounds = list(diffs), list(rounds)
        self.weights = autond_weights(rounds) if weights is None else np.asarray(weights, float)
        self.D = _as_bits(diffs, sampler.n_bits) if diff_bits is None else diff_bits
        self.boot_rng = boot_rng if boot_rng is not None else np.random.default_rng(0)
        self.checkpoint_path = checkpoint_path
        self.policy_name = policy_name
        m, R = len(diffs), len(rounds)
        self.status = np.array([ACTIVE] * m, dtype=object)
        self.status[np.asarray(diffs) == 0] = ELIM
        self.counts = np.zeros((m, R, sampler.n_bits), dtype=np.int64)
        self.t = np.zeros((m, R), dtype=np.int64)
        self.lower = np.full(m, -np.inf)
        self.upper = np.full(m, np.inf)
        self.noise_width = np.full(m, np.inf)  # interval width excluding the shared floor term
        self.store: list[list[tuple[np.ndarray, np.ndarray]]] = [[] for _ in range(R)]
        self.stage = 0
        self.history: list[dict] = []
        self.ledger_start = dict(sampler.ledger.as_dict())
        self.stop_reason = None

    # -- persistence ---------------------------------------------------------
    def _save(self):
        if not self.checkpoint_path:
            return
        tmp = self.checkpoint_path + ".tmp"
        state = {k: v for k, v in self.__dict__.items() if k not in ("sampler",)}
        state["sampler_rng"] = self.sampler.rng.bit_generator.state
        state["sampler_ledger"] = self.sampler.ledger.as_dict()
        state["boot_rng_state"] = self.boot_rng.bit_generator.state
        with open(tmp, "wb") as f:
            pickle.dump(state, f)
        os.replace(tmp, self.checkpoint_path)

    def _try_resume(self) -> bool:
        if not self.checkpoint_path or not os.path.exists(self.checkpoint_path):
            return False
        with open(self.checkpoint_path, "rb") as f:
            state = pickle.load(f)
        self.sampler.rng.bit_generator.state = state.pop("sampler_rng")
        led = state.pop("sampler_ledger")
        self.sampler.ledger.encryptions = led["encryptions"]
        self.sampler.ledger.round_evaluations = led["round_evaluations"]
        boot_state = state.pop("boot_rng_state")
        self.__dict__.update(state)
        self.boot_rng.bit_generator.state = boot_state
        return True

    # -- core ----------------------------------------------------------------
    def _used(self) -> int:
        return self.sampler.ledger.encryptions - self.ledger_start["encryptions"]

    def _bits_for(self, idx: np.ndarray, ri: int) -> np.ndarray:
        """All stored samples (in order) for candidates idx at round index ri."""
        n = self.sampler.n_bits
        parts = []
        for cand, packed in self.store[ri]:
            pos = np.searchsorted(cand, idx)
            ok = (pos < len(cand)) & (cand[np.minimum(pos, len(cand) - 1)] == idx)
            if not np.all(ok):
                break  # later stages do not contain these candidates
            parts.append(np.unpackbits(packed[pos], axis=-1, count=n))
        return np.concatenate(parts, axis=1)

    def _refresh_spectral(self, idx: np.ndarray) -> None:
        """Recompute unbiased C_{<=2} and its SE for candidates idx from all stored samples."""
        from .spectral import cap12_and_se
        if not hasattr(self, "spec_pr"):
            self.spec_pr = np.zeros(self.t.shape)
            self.spec_se = np.zeros(self.t.shape)
        for ri in range(len(self.rounds)):
            cap, se = cap12_and_se(self._bits_for(idx, ri))
            self.spec_pr[idx, ri], self.spec_se[idx, ri] = cap, se

    def _point(self) -> tuple[np.ndarray, np.ndarray]:
        if self.cfg.statistic == "spectral":
            pr = self.spec_pr.copy() if hasattr(self, "spec_pr") else np.zeros(self.t.shape)
            pr[np.asarray(self.diffs) == 0] = 0.0
            return pr, pr @ self.weights
        per_round = bias_from_counts(self.counts, np.maximum(self.t, 1))
        per_round[self.t == 0] = 0.0
        per_round[np.asarray(self.diffs) == 0] = 0.0
        return per_round, per_round @ self.weights

    def _update_intervals(self, idx: np.ndarray, point: np.ndarray, n_looks: int):
        cfg = self.cfg
        if cfg.statistic == "spectral":
            from scipy.stats import norm
            z = float(norm.ppf(1 - cfg.alpha / 2))
            se = np.sqrt((self.spec_se[idx] ** 2) @ (self.weights ** 2))
            self.lower[idx], self.upper[idx] = point[idx] - z * se, point[idx] + z * se
            self.noise_width[idx] = 2 * z * se
            return
        if cfg.interval == "mcdiarmid":
            alpha = cfg.alpha / (len(self.diffs) * max(n_looks, 1))
            lo, hi = intervals(point[idx], None, self.t[idx], self.weights, "mcdiarmid", alpha)
        else:
            bits = [self._bits_for(idx, ri) for ri in range(len(self.rounds))]
            boot = bootstrap_weighted(bits, self.weights, cfg.n_boot, self.boot_rng)
            lo, hi = intervals(point[idx], boot, self.t[idx], self.weights, cfg.interval, cfg.alpha)
        self.lower[idx], self.upper[idx] = lo, hi
        floor = 0.0
        if cfg.interval in ("bias_bounded", "mcdiarmid"):
            floor = np.sum(self.weights[None] * 0.5 / np.sqrt(self.t[idx]), axis=1)
        self.noise_width[idx] = (hi - lo) - floor

    def _decide(self, point: np.ndarray):
        cfg = self.cfg
        live = np.where(self.status != ELIM)[0]
        if len(live) <= cfg.k:
            self.status[live[self.status[live] == ACTIVE]] = ACCEPT
            return
        L, U = self.lower[live], self.upper[live]
        Ls, Us = np.sort(L), np.sort(U)
        n_better = len(live) - np.searchsorted(Ls, U, side="right")      # #{o: L_o > U_a}
        n_maybe_better = len(live) - np.searchsorted(Us, L, side="right") - (U > L)  # others with U_o > L_a
        s_k = np.sort(point[live])[::-1][cfg.k - 1]
        active = self.status[live] == ACTIVE
        elim = active & (n_better >= cfg.k)
        accept = active & ~elim & (n_maybe_better <= cfg.k - 1) if cfg.accept_clear else np.zeros_like(active)
        # The floor term is identical for all candidates at equal t and cannot reorder them,
        # so the tie tolerance is applied to the sampling-noise part of the width only.
        precise = active & ~elim & ~accept & (self.noise_width[live] <= cfg.tol * max(s_k, 1e-12))
        self.status[live[elim]] = ELIM
        self.status[live[accept]] = ACCEPT
        self.status[live[precise]] = PRECISE

    def run(self) -> ScoringResult:
        cfg = self.cfg
        targets = cfg.stage_targets()
        self._try_resume()
        R = len(self.rounds)
        while self.stage < len(targets):
            active = np.where(self.status == ACTIVE)[0]
            if len(active) == 0:
                self.stop_reason = "resolved"
                break
            t_now = int(self.t[active[0], 0])
            batch = targets[self.stage] - t_now
            if cfg.budget is not None:
                remaining = cfg.budget - self._used()
                batch = min(batch, remaining // (R * (len(active) + 1)))
                if batch < cfg.min_batch:
                    self.stop_reason = "budget"
                    break
            for ri, nr in enumerate(self.rounds):
                bits = self.sampler.draw(self.D[active], batch, nr)
                self.counts[active, ri] += bits.sum(axis=1)
                self.t[active, ri] += batch
                self.store[ri].append((active.copy(), np.packbits(bits, axis=-1)))
            if cfg.statistic == "spectral":
                self._refresh_spectral(active)
            per_round, point = self._point()
            self._update_intervals(active, point, n_looks=len(targets))
            self._decide(point)
            reached = t_now + batch
            self.stage += 1 if reached >= targets[self.stage] else 0
            st = {s: int(np.sum(self.status == s)) for s in (ACTIVE, ELIM, ACCEPT, PRECISE)}
            self.history.append({"stage": self.stage, "t": int(reached), "encryptions": self._used(), **st})
            self._save()
        else:
            self.stop_reason = self.stop_reason or "t_max"
        per_round, point = self._point()
        mask = self.status != ELIM
        used = {k_: self.sampler.ledger.as_dict()[k_] - self.ledger_start[k_] for k_ in self.ledger_start}
        return ScoringResult(
            policy=self.policy_name, diffs=self.diffs, rounds=self.rounds,
            weights=self.weights.tolist(), scores=point, per_round=per_round, samples=self.t.copy(),
            shortlist=top_k(self.diffs, point, cfg.k, mask), ledger=used,
            status=[str(s) for s in self.status], lower=self.lower.copy(), upper=self.upper.copy(),
            history=self.history, config={**asdict(cfg), "stop_reason": self.stop_reason})


# ----------------------------------------------------------------------------
# Ablation C2: successive halving (same stage schedule, no uncertainty)
# ----------------------------------------------------------------------------


def score_successive_halving(sampler, diffs: list[int], rounds: list[int], cfg: AdaptiveConfig,
                             eta: float = 2.0, weights=None, diff_bits=None) -> ScoringResult:
    """Same checkpoints as the adaptive scorer, but keeps the top 1/eta by point estimate
    (never fewer than K) instead of using intervals. Isolates the value of uncertainty."""
    weights = autond_weights(rounds) if weights is None else np.asarray(weights, float)
    D = _as_bits(diffs, sampler.n_bits) if diff_bits is None else diff_bits
    m, R = len(diffs), len(rounds)
    counts = np.zeros((m, R, sampler.n_bits), np.int64)
    t = np.zeros((m, R), np.int64)
    alive = np.asarray(diffs) != 0
    spectral = cfg.statistic == "spectral"
    stored = {} if spectral else None          # (cand, round) -> list of bit blocks
    spec = np.zeros((m, R))
    start = dict(sampler.ledger.as_dict())
    history = []
    stop = "t_max"
    for target in cfg.stage_targets():
        idx = np.where(alive)[0]
        batch = target - int(t[idx[0], 0])
        if cfg.budget is not None:
            remaining = cfg.budget - (sampler.ledger.encryptions - start["encryptions"])
            batch = min(batch, remaining // (R * (len(idx) + 1)))
            if batch < cfg.min_batch:
                stop = "budget"
                break
        for ri, nr in enumerate(rounds):
            bits = sampler.draw(D[idx], batch, nr)
            counts[idx, ri] += bits.sum(axis=1)
            t[idx, ri] += batch
            if spectral:
                from .spectral import cap12_and_se
                for row, c in enumerate(idx):
                    stored.setdefault((c, ri), []).append(bits[row])
                allb = np.stack([np.concatenate(stored[(c, ri)]) for c in idx])
                spec[idx, ri] = cap12_and_se(allb)[0]
        pr = spec.copy() if spectral else bias_from_counts(counts, np.maximum(t, 1))
        pr[t == 0] = 0
        sc = pr @ weights
        keep = max(cfg.k, int(math.ceil(len(idx) / eta)))
        if keep < len(idx):
            order = idx[np.argsort(-sc[idx], kind="stable")]
            alive[order[keep:]] = False
        history.append({"t": int(t[idx[0], 0]), "alive": int(alive.sum()),
                        "encryptions": sampler.ledger.encryptions - start["encryptions"]})
        if alive.sum() <= cfg.k:
            stop = "resolved"
            break
    pr = spec.copy() if spectral else bias_from_counts(counts, np.maximum(t, 1))
    pr[t == 0] = 0
    pr[np.asarray(diffs) == 0] = 0
    sc = pr @ weights
    used = {k_: sampler.ledger.as_dict()[k_] - start[k_] for k_ in start}
    return ScoringResult(policy="C2_successive_halving", diffs=list(diffs), rounds=list(rounds),
                         weights=weights.tolist(), scores=sc, per_round=pr, samples=t,
                         shortlist=top_k(list(diffs), sc, cfg.k, alive), ledger=used,
                         status=["alive" if a else "dropped" for a in alive], history=history,
                         config={**asdict(cfg), "eta": eta, "stop_reason": stop})

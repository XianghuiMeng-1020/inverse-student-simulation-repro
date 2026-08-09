"""Discrete multi-skill BKT with EM parameter fitting (PyBKT-style simplified)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class BKTParams:
    p_init: float = 0.3
    p_learn: float = 0.2
    p_guess: float = 0.1
    p_slip: float = 0.05


@dataclass
class MultiSkillBKT:
    """One BKT model per KC; EM fit on binary correctness sequences."""

    params_by_skill: dict[str, BKTParams] = field(default_factory=dict)

    def _forward_backward(
        self,
        obs: np.ndarray,
        p: BKTParams,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """obs: 0/1 correctness sequence. Returns log-likelihood, alpha, beta posteriors."""
        T = len(obs)
        if T == 0:
            return 0.0, np.array([]), np.array([])

        p_init, p_learn, p_guess, p_slip = p.p_init, p.p_learn, p.p_guess, p.p_slip
        alpha = np.zeros(T)
        beta = np.zeros(T)
        alpha[0] = p_init
        for t in range(1, T):
            p_known = alpha[t - 1] + (1 - alpha[t - 1]) * p_learn
            alpha[t] = p_known

        ll = 0.0
        for t in range(T):
            p_correct = alpha[t] * (1 - p_slip) + (1 - alpha[t]) * p_guess
            p_correct = np.clip(p_correct, 1e-6, 1 - 1e-6)
            ll += obs[t] * np.log(p_correct) + (1 - obs[t]) * np.log(1 - p_correct)

        beta[T - 1] = 1.0
        for t in range(T - 2, -1, -1):
            p_known = alpha[t] + (1 - alpha[t]) * p_learn
            emit0 = (1 - alpha[t]) * p_guess + alpha[t] * p_slip
            emit1 = (1 - alpha[t]) * (1 - p_guess) + alpha[t] * (1 - p_slip)
            emit0, emit1 = max(emit0, 1e-9), max(emit1, 1e-9)
            beta[t] = (
                beta[t + 1] * p_learn * (emit1 if obs[t + 1] else emit0)
                + beta[t + 1] * (1 - p_learn) * (emit1 if obs[t + 1] else emit0)
            )
            beta[t] = np.clip(beta[t], 1e-9, 1.0)

        return float(ll), alpha, beta

    def fit_skill(self, obs: np.ndarray, *, n_em: int = 25) -> BKTParams:
        if len(obs) < 2:
            return BKTParams()
        best_p = BKTParams()
        best_ll = -np.inf
        rng = np.random.default_rng(0)
        for _ in range(n_em):
            p = BKTParams(
                p_init=float(rng.uniform(0.1, 0.5)),
                p_learn=float(rng.uniform(0.05, 0.4)),
                p_guess=float(rng.uniform(0.05, 0.3)),
                p_slip=float(rng.uniform(0.02, 0.2)),
            )
            for _ in range(8):
                _, alpha, beta = self._forward_backward(obs, p)
                T = len(obs)
                # M-step simplified updates
                num_learn = 0.0
                den_learn = 0.0
                for t in range(T - 1):
                    p_trans = alpha[t] * (1 - p.p_learn) + (1 - alpha[t]) * p.p_learn
                    num_learn += p_trans * beta[t + 1] * p.p_learn
                    den_learn += p_trans * beta[t + 1]
                if den_learn > 1e-6:
                    p.p_learn = np.clip(num_learn / den_learn, 0.01, 0.99)
                p.p_init = np.clip(alpha[0], 0.05, 0.95)
                ll, _, _ = self._forward_backward(obs, p)
                if ll > best_ll:
                    best_ll = ll
                    best_p = BKTParams(**p.__dict__)
        return best_p

    def predict_mastery(self, obs: np.ndarray, p: BKTParams) -> float:
        """Posterior P(know) after observing sequence."""
        if len(obs) == 0:
            return p.p_init
        _, alpha, _ = self._forward_backward(obs, p)
        return float(alpha[-1])

    def fit_all(self, sequences: dict[str, list[int]]) -> None:
        for skill, obs_list in sequences.items():
            arr = np.asarray(obs_list, dtype=int)
            self.params_by_skill[skill] = self.fit_skill(arr)

    def mastery_vector(self, sequences: dict[str, list[int]]) -> dict[str, float]:
        out: dict[str, float] = {}
        for skill, obs_list in sequences.items():
            p = self.params_by_skill.get(skill, BKTParams())
            out[skill] = self.predict_mastery(np.asarray(obs_list, dtype=int), p)
        return out

"""Train / eval scikit-learn heads on sentence embeddings (real labels only)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


@dataclass(frozen=True)
class ClfMetrics:
    accuracy: float
    macro_f1: float
    weighted_f1: float
    n_train: int
    n_eval: int
    n_classes: int


def fit_logistic_multiclass(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int,
    max_iter: int = 500,
) -> LogisticRegression:
    clf = LogisticRegression(
        max_iter=max_iter,
        class_weight="balanced",
        random_state=seed,
        solver="lbfgs",
    )
    clf.fit(x_train, y_train)
    return clf


def eval_clf_full(
    clf: LogisticRegression,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
) -> ClfMetrics:
    pred = clf.predict(x_eval)
    n_cls = len(np.unique(np.concatenate([y_train, y_eval])))
    return ClfMetrics(
        accuracy=float(accuracy_score(y_eval, pred)),
        macro_f1=float(f1_score(y_eval, pred, average="macro", zero_division=0)),
        weighted_f1=float(f1_score(y_eval, pred, average="weighted", zero_division=0)),
        n_train=len(y_train),
        n_eval=len(y_eval),
        n_classes=int(n_cls),
    )


def top_k_labels(y: list[str], k: int) -> set[str]:
    from collections import Counter

    counts = Counter(y)
    return {lab for lab, _ in counts.most_common(k)}


def remap_rare_to_other(y: list[str], top: set[str]) -> list[str]:
    return [lab if lab in top else "OTHER" for lab in y]


def stratified_subset(
    x: np.ndarray,
    y: np.ndarray,
    max_n: int,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if max_n <= 0 or len(y) <= max_n:
        return x, y
    try:
        x_sub, _, y_sub, _ = train_test_split(
            x,
            y,
            train_size=max_n,
            stratify=y,
            random_state=seed,
        )
    except ValueError:
        x_sub, _, y_sub, _ = train_test_split(
            x,
            y,
            train_size=max_n,
            random_state=seed,
        )
    return x_sub, y_sub


def encode_texts(
    texts: list[str],
    model_name: str,
    *,
    batch_size: int = 32,
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return np.asarray(
        model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        ),
        dtype=np.float32,
    )


def metrics_to_dict(m: ClfMetrics) -> dict[str, Any]:
    return {
        "accuracy": m.accuracy,
        "macro_f1": m.macro_f1,
        "weighted_f1": m.weighted_f1,
        "n_train": m.n_train,
        "n_eval": m.n_eval,
        "n_classes": m.n_classes,
    }


def fit_predict_encode_pipeline(
    x_train: np.ndarray,
    y_raw_train: list[str],
    x_eval: np.ndarray,
    y_raw_eval: list[str],
    *,
    seed: int,
) -> tuple[ClfMetrics, dict[str, Any]]:
    train_lab = set(y_raw_train)
    keep_idx = [i for i, lab in enumerate(y_raw_eval) if lab in train_lab]
    if not keep_idx:
        msg = "No evaluation rows share a label with the training set after filtering."
        raise ValueError(msg)
    x_eval_f = x_eval[keep_idx]
    y_eval_f = [y_raw_eval[i] for i in keep_idx]

    le = LabelEncoder()
    y_train = le.fit_transform(y_raw_train)
    y_eval = le.transform(y_eval_f)
    clf = fit_logistic_multiclass(x_train, y_train, seed=seed)
    m = eval_clf_full(clf, x_train, y_train, x_eval_f, y_eval)
    aux = {"label_classes": le.classes_.tolist(), "n_eval_filtered": len(keep_idx)}
    return m, aux

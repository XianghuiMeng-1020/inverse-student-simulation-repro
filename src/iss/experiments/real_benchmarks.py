"""Benchmark definitions over processed parquet shards (real labels)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from iss.experiments.dialogue_text import (
    extract_last_talkmoves_label,
    loads_metadata,
    loads_turns,
    turns_to_plain_text,
)
from iss.experiments.sklearn_eval import (
    encode_texts,
    fit_predict_encode_pipeline,
    metrics_to_dict,
    remap_rare_to_other,
    stratified_subset,
    top_k_labels,
)


def _read_shard(path: Path) -> pd.DataFrame:
    if not path.exists():
        msg = f"missing parquet shard: {path}"
        raise FileNotFoundError(msg)
    return pd.read_parquet(path)


def _rows_bridge_target(df: pd.DataFrame, field: str) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []
    for _, row in df.iterrows():
        meta = loads_metadata(str(row["metadata_json"]))
        lab = str(meta.get(field, "")).strip()
        if not lab:
            continue
        turns = loads_turns(str(row["turns_json"]))
        text = turns_to_plain_text(turns, strip_talkmoves_suffix=False)
        if not text.strip():
            continue
        texts.append(text)
        labels.append(lab)
    return texts, labels


def run_bridge_error_type(
    processed_root: Path,
    *,
    embedding_model: str,
    seed: int,
    max_train: int,
    max_eval: int,
    e_topk: int,
) -> dict[str, Any]:
    train_df = _read_shard(processed_root / "bridge" / "train.parquet")
    val_df = _read_shard(processed_root / "bridge" / "validation.parquet")

    tr_xs, tr_y = _rows_bridge_target(train_df, "error_type")
    ev_xs, ev_y = _rows_bridge_target(val_df, "error_type")

    top = top_k_labels(tr_y, e_topk)
    tr_y2 = remap_rare_to_other(tr_y, top)
    ev_y2 = remap_rare_to_other(ev_y, top)

    x_tr = encode_texts(tr_xs, embedding_model)
    x_ev = encode_texts(ev_xs, embedding_model)

    x_tr, y_tr_arr = stratified_subset(x_tr, np.array(tr_y2), max_train, seed=seed)
    x_ev, y_ev_arr = stratified_subset(x_ev, np.array(ev_y2), max_eval, seed=seed + 1)
    y_tr_list = y_tr_arr.tolist()
    y_ev_list = y_ev_arr.tolist()

    m, aux = fit_predict_encode_pipeline(
        x_tr,
        y_tr_list,
        x_ev,
        y_ev_list,
        seed=seed,
    )
    return {
        "task": "bridge_error_type_e",
        "e_topk": e_topk,
        "split": "train -> validation",
        "metrics": metrics_to_dict(m),
        "embedding_model": embedding_model,
        "aux": aux,
    }


def run_bridge_z_what(
    processed_root: Path,
    *,
    embedding_model: str,
    seed: int,
    max_train: int,
    max_eval: int,
    zwhat_topk: int,
) -> dict[str, Any]:
    train_df = _read_shard(processed_root / "bridge" / "train.parquet")
    val_df = _read_shard(processed_root / "bridge" / "validation.parquet")
    tr_xs, tr_y = _rows_bridge_target(train_df, "z_what")
    ev_xs, ev_y = _rows_bridge_target(val_df, "z_what")
    top = top_k_labels(tr_y, zwhat_topk)
    tr_y2 = remap_rare_to_other(tr_y, top)
    ev_y2 = remap_rare_to_other(ev_y, top)

    x_tr = encode_texts(tr_xs, embedding_model)
    x_ev = encode_texts(ev_xs, embedding_model)

    x_tr, y_tr_arr = stratified_subset(x_tr, np.array(tr_y2), max_train, seed=seed)
    x_ev, y_ev_arr = stratified_subset(x_ev, np.array(ev_y2), max_eval, seed=seed + 1)

    m, aux = fit_predict_encode_pipeline(
        x_tr,
        y_tr_arr.tolist(),
        x_ev,
        y_ev_arr.tolist(),
        seed=seed,
    )
    return {
        "task": "bridge_z_what_topk_other",
        "topk": zwhat_topk,
        "split": "train -> validation",
        "metrics": metrics_to_dict(m),
        "embedding_model": embedding_model,
        "aux": aux,
    }


def run_bridge_error_type_train_to_test(
    processed_root: Path,
    *,
    embedding_model: str,
    seed: int,
    max_train: int,
    max_eval: int,
    e_topk: int,
) -> dict[str, Any]:
    """Held-out Bridge ``test`` split (stricter than validation)."""

    train_df = _read_shard(processed_root / "bridge" / "train.parquet")
    test_df = _read_shard(processed_root / "bridge" / "test.parquet")
    tr_xs, tr_y = _rows_bridge_target(train_df, "error_type")
    ev_xs, ev_y = _rows_bridge_target(test_df, "error_type")
    top = top_k_labels(tr_y, e_topk)
    tr_y2 = remap_rare_to_other(tr_y, top)
    ev_y2 = remap_rare_to_other(ev_y, top)
    x_tr = encode_texts(tr_xs, embedding_model)
    x_ev = encode_texts(ev_xs, embedding_model)
    x_tr, y_tr_arr = stratified_subset(x_tr, np.array(tr_y2), max_train, seed=seed)
    x_ev, y_ev_arr = stratified_subset(x_ev, np.array(ev_y2), max_eval, seed=seed + 1)
    m, aux = fit_predict_encode_pipeline(
        x_tr,
        y_tr_arr.tolist(),
        x_ev,
        y_ev_arr.tolist(),
        seed=seed,
    )
    return {
        "task": "bridge_error_type_e_train_to_test",
        "e_topk": e_topk,
        "split": "train -> test",
        "metrics": metrics_to_dict(m),
        "embedding_model": embedding_model,
        "aux": aux,
    }


def run_bridge_lesson_topic(
    processed_root: Path,
    *,
    embedding_model: str,
    seed: int,
    max_train: int,
    max_eval: int,
    topic_topk: int,
) -> dict[str, Any]:
    train_df = _read_shard(processed_root / "bridge" / "train.parquet")
    val_df = _read_shard(processed_root / "bridge" / "validation.parquet")
    tr_xs, tr_y = _rows_bridge_target(train_df, "lesson_topic")
    ev_xs, ev_y = _rows_bridge_target(val_df, "lesson_topic")
    top = top_k_labels(tr_y, topic_topk)
    tr_y2 = remap_rare_to_other(tr_y, top)
    ev_y2 = remap_rare_to_other(ev_y, top)
    x_tr = encode_texts(tr_xs, embedding_model)
    x_ev = encode_texts(ev_xs, embedding_model)
    x_tr, y_tr_arr = stratified_subset(x_tr, np.array(tr_y2), max_train, seed=seed)
    x_ev, y_ev_arr = stratified_subset(x_ev, np.array(ev_y2), max_eval, seed=seed + 1)
    m, aux = fit_predict_encode_pipeline(
        x_tr,
        y_tr_arr.tolist(),
        x_ev,
        y_ev_arr.tolist(),
        seed=seed,
    )
    return {
        "task": "bridge_lesson_topic_topk_other",
        "topk": topic_topk,
        "split": "train -> validation",
        "metrics": metrics_to_dict(m),
        "embedding_model": embedding_model,
        "aux": aux,
    }


def run_talkmoves_student_label(
    processed_root: Path,
    *,
    embedding_model: str,
    seed: int,
    max_train: int,
    max_eval: int,
    label_topk: int,
) -> dict[str, Any]:
    train_df = _read_shard(processed_root / "talkmoves" / "train_student.parquet")
    test_df = _read_shard(processed_root / "talkmoves" / "test_student.parquet")

    def collect(df: pd.DataFrame) -> tuple[list[str], list[str]]:
        xs: list[str] = []
        ys: list[str] = []
        for _, row in df.iterrows():
            turns = loads_turns(str(row["turns_json"]))
            lab = extract_last_talkmoves_label(turns)
            if not lab:
                continue
            text = turns_to_plain_text(turns, strip_talkmoves_suffix=True)
            if not text.strip():
                continue
            xs.append(text)
            ys.append(lab)
        return xs, ys

    tr_xs, tr_y = collect(train_df)
    ev_xs, ev_y = collect(test_df)
    top = top_k_labels(tr_y, label_topk)
    tr_y2 = remap_rare_to_other(tr_y, top)
    ev_y2 = remap_rare_to_other(ev_y, top)

    x_tr = encode_texts(tr_xs, embedding_model)
    x_ev = encode_texts(ev_xs, embedding_model)
    x_tr, y_tr_arr = stratified_subset(x_tr, np.array(tr_y2), max_train, seed=seed)
    x_ev, y_ev_arr = stratified_subset(x_ev, np.array(ev_y2), max_eval, seed=seed + 1)

    m, aux = fit_predict_encode_pipeline(
        x_tr,
        y_tr_arr.tolist(),
        x_ev,
        y_ev_arr.tolist(),
        seed=seed,
    )
    return {
        "task": "talkmoves_student_label",
        "topk": label_topk,
        "split": "train_student -> test_student",
        "metrics": metrics_to_dict(m),
        "embedding_model": embedding_model,
        "aux": aux,
    }


def run_dataset_source_classifier(
    processed_root: Path,
    *,
    embedding_model: str,
    seed: int,
    max_per_class: int,
) -> dict[str, Any]:
    """Four-way classifier: which corpus a dialogue comes from (real domain shift)."""

    shards: list[tuple[str, Path]] = [
        ("mathdial", processed_root / "mathdial" / "train.parquet"),
        ("bridge", processed_root / "bridge" / "train.parquet"),
        ("cima", processed_root / "cima" / "all.parquet"),
        ("talkmoves", processed_root / "talkmoves" / "train_student.parquet"),
    ]
    all_texts: list[str] = []
    all_labels: list[str] = []
    for ds_name, path in shards:
        df = _read_shard(path)
        cap = min(max_per_class, len(df)) if max_per_class > 0 else len(df)
        for _, row in df.head(cap).iterrows():
            turns = loads_turns(str(row["turns_json"]))
            text = turns_to_plain_text(turns, strip_talkmoves_suffix=True)
            if not text.strip():
                continue
            all_texts.append(text)
            all_labels.append(ds_name)

    x = encode_texts(all_texts, embedding_model)
    y = np.array(all_labels)

    from sklearn.model_selection import train_test_split

    try:
        x_tr, x_te, y_tr, y_te = train_test_split(
            x,
            y,
            test_size=0.25,
            stratify=y,
            random_state=seed,
        )
    except ValueError:
        x_tr, x_te, y_tr, y_te = train_test_split(
            x,
            y,
            test_size=0.25,
            random_state=seed,
        )
    m, aux = fit_predict_encode_pipeline(
        x_tr,
        y_tr.tolist(),
        x_te,
        y_te.tolist(),
        seed=seed,
    )
    return {
        "task": "dataset_source_classifier",
        "split": "stratified_holdout_on_pooled_train_shards",
        "metrics": metrics_to_dict(m),
        "embedding_model": embedding_model,
        "max_per_class": max_per_class,
        "aux": aux,
    }

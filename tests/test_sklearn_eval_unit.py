"""Fast tests for sklearn heads (random features, no sentence-transformers)."""

from __future__ import annotations

import numpy as np

from iss.experiments.sklearn_eval import fit_predict_encode_pipeline


def test_fit_predict_encode_pipeline_shapes() -> None:
    rng = np.random.default_rng(0)
    x_tr = rng.standard_normal((40, 8)).astype(np.float32)
    x_ev = rng.standard_normal((20, 8)).astype(np.float32)
    y_tr = ["a"] * 20 + ["b"] * 20
    y_ev = ["a"] * 10 + ["b"] * 10
    m, aux = fit_predict_encode_pipeline(x_tr, y_tr, x_ev, y_ev, seed=1)
    assert m.n_train == 40
    assert m.n_eval == 20
    assert set(aux["label_classes"]) == {"a", "b"}

import numpy as np
import pandas as pd

from autoprofile import profile, profile_text


def _kinds(df, **kw):
    return {f.kind for f in profile(df, **kw)}


def test_empty_frame():
    assert profile(pd.DataFrame()) == []


def test_detects_missing_duplicates_outliers_corr_skew():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    x[0] = 100.0
    df = pd.DataFrame({"x": x, "y": x * 2 + rng.normal(scale=0.01, size=200),
                       "cat": rng.choice(list("abcd"), 200)})
    df.loc[1, "cat"] = None
    df = pd.concat([df, df.iloc[[5]]])
    kinds = _kinds(df)
    assert {"missing", "duplicates", "outliers", "correlation", "top_bottom"} <= kinds


def test_skew_detected():
    df = pd.DataFrame({"v": [1] * 50 + [1000]})
    assert "skew" in _kinds(df)


def test_profile_text_and_dict():
    df = pd.DataFrame({"a": [1, None, 3]})
    assert "missing" in profile_text(df)
    assert profile(df)[0].to_dict()["kind"] == "missing"


def test_constant_column_no_crash():
    kinds = {f.kind for f in profile(pd.DataFrame({"a": [1, 1, 1], "b": [2, 2, 2]}))}
    assert kinds == {"duplicates"}

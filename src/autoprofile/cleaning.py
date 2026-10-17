"""Turn strings like '1.234,50 €', '$1,000.50' or '12.5%' into floats."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

_SYMBOLS = re.compile(r"[$€£¥%\s]")


def parse_number(value) -> float:
    """Parse one number in EU (1.234,50) or US (1,234.50) notation. Returns NaN if impossible."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    s = _SYMBOLS.sub("", str(value))
    if s in ("", "nan", "None", "-"):
        return np.nan
    if "." in s and "," in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")  # EU
        else:
            s = s.replace(",", "")  # US
    elif "," in s:
        head, _, tail = s.rpartition(",")
        # '1,234' / '1,234,567' -> thousands separator; '10,5' / '0,750' -> decimal comma
        if s.count(",") > 1 or (len(tail) == 3 and head.lstrip("-") not in ("0", "")):
            s = s.replace(",", "")
        else:
            s = s.replace(",", ".")
    elif s.count(".") > 1:
        s = s.replace(".", "")  # '1.234.567'
    try:
        return float(s)
    except ValueError:
        return np.nan


def clean_numeric_strings(df: pd.DataFrame, sample_size: int = 20, min_parse_rate: float = 0.8) -> pd.DataFrame:
    """Convert string columns containing currency symbols or '%' to numeric.

    A column is converted only if it contains a symbol in its first `sample_size`
    non-null values AND at least `min_parse_rate` of those values parse. Returns a copy.
    """
    out = df.copy()
    for col in out.columns:
        s = out[col]
        if not (s.dtype == object or pd.api.types.is_string_dtype(s)):
            continue
        sample = s.dropna().head(sample_size).astype(str)
        if sample.empty or not sample.str.contains(r"[$€£¥%]", regex=True).any():
            continue
        parsed_sample = sample.map(parse_number)
        if parsed_sample.notna().mean() < min_parse_rate:
            continue
        out[col] = s.map(parse_number)
    return out

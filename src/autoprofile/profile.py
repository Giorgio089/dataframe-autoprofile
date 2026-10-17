"""Rule-based exploration of a DataFrame. No LLM, no network."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass
class Finding:
    kind: str  # missing | top_bottom | outliers | correlation | skew | duplicates
    message: str
    columns: list
    value: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def profile(df: pd.DataFrame, corr_threshold: float = 0.7, max_cols: int = 3) -> list[Finding]:
    """Return notable findings about `df`."""
    findings: list[Finding] = []
    if df.empty:
        return findings

    missing = df.isna().sum()
    missing = missing[missing > 0]
    if len(missing):
        findings.append(Finding("missing", f"{int(missing.sum())} missing values in {len(missing)} columns",
                                list(missing.index), float(missing.sum())))

    dups = int(df.duplicated().sum())
    if dups:
        findings.append(Finding("duplicates", f"{dups} fully duplicated rows", [], float(dups)))

    numeric = df.select_dtypes(include="number").columns.tolist()
    categorical = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    if numeric and categorical:
        metric, cat = numeric[0], categorical[0]
        grouped = df.groupby(cat)[metric].sum().sort_values(ascending=False)
        if len(grouped) >= 3:
            findings.append(Finding(
                "top_bottom",
                f"'{cat}' by sum of '{metric}': top {list(grouped.head(3).index)}, bottom {list(grouped.tail(3).index)}",
                [cat, metric]))

    for col in numeric[:max_cols]:
        s = df[col].dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            n = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
            if n:
                pct = n / len(s) * 100
                findings.append(Finding("outliers", f"{n} IQR outliers in '{col}' ({pct:.1f}%)", [col], pct))

    if len(numeric) >= 2:
        corr = df[numeric].corr()
        mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
        pairs = corr.where(mask).stack()
        strong = pairs[pairs.abs() > corr_threshold].sort_values(key=abs, ascending=False).head(3)
        for (a, b), r in strong.items():
            findings.append(Finding("correlation", f"'{a}' and '{b}' strongly correlated (r={r:.2f})", [a, b], float(r)))

    for col in numeric[:2]:
        skew = df[col].skew()
        if pd.notna(skew) and abs(skew) > 2:
            side = "right" if skew > 0 else "left"
            findings.append(Finding("skew", f"'{col}' is strongly {side}-skewed (skew={skew:.1f})", [col], float(skew)))

    return findings


def profile_text(df: pd.DataFrame, **kwargs) -> str:
    """Findings as a bullet list, ready to paste into an LLM prompt."""
    return "\n".join(f"- [{f.kind}] {f.message}" for f in profile(df, **kwargs))

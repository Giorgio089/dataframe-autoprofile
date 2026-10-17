# dataframe-autoprofile

LLM-free, dependency-light profiling of pandas DataFrames, extracted from the
[Mistral Data Detective](https://github.com/Giorgio089/mistral-data-detective) multi-agent project.

- **`profile(df)`** – finds missing values, duplicate rows, IQR outliers, strong correlations, skewed distributions and top/bottom categories.
- **`profile_text(df)`** – the same as a bullet list you can drop into an LLM prompt as grounded context.
- **`clean_numeric_strings(df)`** – converts columns like `"1.234,50 €"`, `"$1,000.50"` or `"12,5%"` to floats (EU and US notation).

```python
import pandas as pd
from autoprofile import clean_numeric_strings, profile_text

df = clean_numeric_strings(pd.read_csv("marketing.csv"))
print(profile_text(df))
# - [missing] 3 missing values in 2 columns
# - [outliers] 4 IQR outliers in 'spend' (9.8%)
# - [correlation] 'clicks' and 'impressions' strongly correlated (r=0.93)
```

## Install

```bash
pip install git+https://github.com/Giorgio089/dataframe-autoprofile
```

## Notes

- A lone comma such as `"1,234"` is read as a thousands separator, `"10,5"` and `"0,750"` as decimal commas. This is a heuristic; check your data.
- Column conversion requires a currency/percent symbol in the sample and ≥80 % parseable values, so ID-like columns (`"001"`) are left alone.

MIT licensed.

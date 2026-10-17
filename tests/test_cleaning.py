import numpy as np
import pandas as pd
import pytest

from autoprofile import clean_numeric_strings, parse_number


@pytest.mark.parametrize("raw,expected", [
    ("1.234,50 €", 1234.5), ("$1,000.50", 1000.5), ("12.5%", 12.5), ("10,5", 10.5),
    ("1,234", 1234.0), ("1,234,567", 1234567.0), ("0,750", 0.75), ("1.234.567", 1234567.0),
    ("-3,5%", -3.5), (" 42 ", 42.0),
])
def test_parse_number(raw, expected):
    assert parse_number(raw) == pytest.approx(expected)


@pytest.mark.parametrize("raw", ["", "abc", None, "nan", np.nan])
def test_parse_number_invalid(raw):
    assert np.isnan(parse_number(raw))


def test_clean_numeric_strings_converts_only_symbol_columns():
    df = pd.DataFrame({"rev": ["1.000,50 €", "2.000,00 €", None], "ctr": ["1,5%", "2%", "3%"],
                       "name": ["a", "b", "c"], "code": ["001", "002", "003"]})
    out = clean_numeric_strings(df)
    assert out["rev"].tolist()[:2] == [1000.5, 2000.0]
    assert out["ctr"].tolist() == [1.5, 2.0, 3.0]
    assert out["name"].tolist() == ["a", "b", "c"]
    assert out["code"].tolist() == ["001", "002", "003"]
    assert df["rev"][0] == "1.000,50 €"  # input untouched


def test_text_column_with_stray_symbol_is_not_converted():
    df = pd.DataFrame({"note": ["50% off", "sale $", "new", "old"]})
    assert clean_numeric_strings(df)["note"].tolist() == df["note"].tolist()

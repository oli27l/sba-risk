"""Testy přípravy dat pro model: únik informací, dělení podle let, odvozené proměnné."""

import numpy as np
import pandas as pd

from sba_risk.features import (
    CATEGORICAL,
    FEATURES,
    LEAKAGE,
    NUMERIC,
    TARGET,
    X_y,
    add_features,
    split_by_year,
)


def loans(years: list[int], maturity: str = "2010-01-01") -> pd.DataFrame:
    """Vymyšlené vyčištěné úvěry schválené v daných letech."""
    n = len(years)
    return pd.DataFrame({
        "approval_date": pd.to_datetime([f"{y}-06-01" for y in years]),
        "maturity_date": pd.Timestamp(maturity),
        "gr_appv": 100_000.0,
        "sba_share": 0.75,
        "term_months": 84.0,
        "no_emp": 4,
        "create_job": 0,
        "retained_job": 2,
        "new_business": 0.0,
        "franchise": 0,
        "real_estate": 0,
        "low_doc": 0.0,
        "sector": "Maloobchod",
        "state": "IN",
        "bank_state": ["IN", "OH"] * (n // 2) + ["IN"] * (n % 2),
        "rev_line_cr": "N",
        "default": [0, 1] * (n // 2) + [0] * (n % 2),
        "chgoff_date": pd.NaT,
        "chgoff_prin_gr": 0.0,
        "disbursement_gross": 100_000.0,
        "disbursement_date": pd.to_datetime([f"{y}-08-01" for y in years]),
    })


def test_no_leakage_in_features():
    # Údaje známé až po schválení ani cíl nesmí být mezi vstupy modelu.
    assert not set(LEAKAGE) & set(FEATURES)
    assert TARGET not in FEATURES


def test_X_contains_only_features():
    X, y = X_y(add_features(loans([1995, 1996])))
    assert list(X.columns) == FEATURES
    assert not set(LEAKAGE) & set(X.columns)
    assert y.name == TARGET


def test_features_are_numeric_or_categorical():
    assert set(NUMERIC).isdisjoint(CATEGORICAL)
    assert len(FEATURES) == len(set(FEATURES))


def test_add_features():
    df = add_features(loans([1995, 1996]))
    assert np.isclose(df.loc[0, "log_gr_appv"], np.log(100_000))
    assert np.isclose(df.loc[0, "log_no_emp"], np.log(5))
    assert df["same_state"].tolist() == [1, 0]


def test_add_features_does_not_change_input():
    df = loans([1995])
    add_features(df)
    assert "log_gr_appv" not in df.columns


def test_split_by_year_puts_years_in_right_parts():
    df = loans([1989, 1990, 2000, 2001, 2003, 2004, 2006, 2007])
    parts = split_by_year(df)
    years = {name: part["approval_date"].dt.year.tolist() for name, part in parts.items()}
    assert years == {"train": [1990, 2000], "valid": [2001, 2003], "test": [2004, 2006]}


def test_split_parts_do_not_overlap():
    parts = split_by_year(loans(list(range(1990, 2007))))
    indexes = [set(part.index) for part in parts.values()]
    assert not indexes[0] & indexes[1]
    assert not indexes[1] & indexes[2]
    # Všechny trénovací roky jsou dřív než všechny testovací.
    assert parts["train"]["approval_date"].max() < parts["test"]["approval_date"].min()


def test_split_drops_loans_not_matured_by_2014():
    df = pd.concat([loans([1995], "2014-06-30"), loans([1995], "2014-07-01")], ignore_index=True)
    assert len(split_by_year(df)["train"]) == 1
    assert len(split_by_year(df, matured_only=False)["train"]) == 2

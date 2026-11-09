"""Testy popisných výpočtů (otázky 1A a 1B) na malých vymyšlených datech."""

import pandas as pd

from sba_risk.describe import (
    cluster_bootstrap_rate,
    default_rate_by_sector,
    matured_loans,
)


def loans(rows: list[tuple]) -> pd.DataFrame:
    """Úvěry zadané jako (odvětví, rok vyplacení, banka, odepsán, splatnost)."""
    df = pd.DataFrame(rows, columns=["sector", "year", "bank", "default", "maturity"])
    df["disbursement_date"] = pd.to_datetime(df["year"].astype(str) + "-06-01")
    df["approval_date"] = df["disbursement_date"]
    df["maturity_date"] = pd.to_datetime(df["maturity"])
    return df


def test_default_rate_by_sector():
    df = loans([
        ("Maloobchod", 2000, "A", 1, "2005-01-01"),
        ("Maloobchod", 2000, "A", 0, "2005-01-01"),
        ("Výroba", 2000, "B", 0, "2005-01-01"),
    ])
    table = default_rate_by_sector(df)
    assert table.loc["Maloobchod", "rate"] == 0.5
    assert table.loc["Výroba", "rate"] == 0.0
    assert table.loc["Maloobchod", "loans"] == 2


def test_default_rate_by_sector_keeps_only_1987_to_2010():
    df = loans([
        ("Výroba", 1986, "A", 1, "2000-01-01"),
        ("Výroba", 1987, "A", 0, "2000-01-01"),
        ("Výroba", 2010, "A", 0, "2015-01-01"),
        ("Výroba", 2011, "A", 1, "2015-01-01"),
    ])
    table = default_rate_by_sector(df)
    assert table.loc["Výroba", "loans"] == 2
    assert table.loc["Výroba", "rate"] == 0.0


def test_matured_loans_uses_cutoff():
    df = loans([
        ("Výroba", 2000, "A", 0, "2014-06-30"),
        ("Výroba", 2000, "A", 0, "2014-07-01"),
    ])
    assert len(matured_loans(df)) == 1


def test_bootstrap_is_reproducible_and_contains_estimate():
    df = loans([("Výroba", 2000, b, d, "2005-01-01")
                for b in "ABCDEFGHIJ" for d in (0, 0, 0, 1)])
    first = cluster_bootstrap_rate(df, n_boot=500, seed=1)
    second = cluster_bootstrap_rate(df, n_boot=500, seed=1)
    assert first == second
    assert first["rate"] == 0.25
    assert first["cluster_low"] <= first["rate"] <= first["cluster_high"]


def test_cluster_interval_wider_when_banks_differ():
    # Banky A–E odepisují všechno, F–J nic: úvěry uvnitř banky nejsou nezávislé,
    # takže shlukový interval musí být širší než naivní.
    rows = [("Výroba", 2000, b, 1, "2005-01-01") for b in "ABCDE" for _ in range(20)]
    rows += [("Výroba", 2000, b, 0, "2005-01-01") for b in "FGHIJ" for _ in range(20)]
    result = cluster_bootstrap_rate(loans(rows), n_boot=1000)
    naive_width = result["naive_high"] - result["naive_low"]
    cluster_width = result["cluster_high"] - result["cluster_low"]
    assert cluster_width > naive_width


def test_missing_bank_is_one_cluster():
    df = loans([("Výroba", 2000, None, 1, "2005-01-01"),
                ("Výroba", 2000, None, 0, "2005-01-01"),
                ("Výroba", 2000, "A", 0, "2005-01-01")])
    assert cluster_bootstrap_rate(df, n_boot=10)["clusters"] == 2

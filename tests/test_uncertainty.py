"""Testy nejistoty: logit, bootstrap po bankách, chyba v čase, interval, pokrytí."""

import json

import numpy as np
import pandas as pd
import pytest

from sba_risk.model import golden_loans, logit as make_logit
from sba_risk.uncertainty import (
    UNCERTAINTY_PATH,
    apply_interval,
    bank_bootstrap_deviations,
    coverage,
    expit,
    fit_weighted,
    interval_shifts,
    logit,
    plot_coverage,
    time_offsets,
)


def fake_loans(years: list[int], rates: list[float], n: int = 300, seed: int = 0) -> pd.DataFrame:
    """Vymyšlené úvěry: pro každý rok n úvěrů od 10 bank s daným podílem odepsaných."""
    rng = np.random.default_rng(seed)
    parts = []
    for year, rate in zip(years, rates):
        df = golden_loans().sample(n, replace=True, random_state=year).reset_index(drop=True)
        df["sba_share"] = rng.uniform(0.5, 0.9, n)
        df["approval_date"] = pd.Timestamp(f"{year}-06-01")
        df["bank"] = rng.choice([f"banka {i}" for i in range(10)], n)
        df["default"] = (rng.uniform(size=n) < rate).astype(int)
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def test_logit_and_expit_are_inverse():
    p = np.array([0.01, 0.2, 0.5, 0.9])
    assert np.allclose(expit(logit(p)), p)
    assert logit(0.5) == 0


def test_fit_weighted_equals_duplicated_rows():
    # Váha 2 musí dát (skoro) stejný model, jako kdyby byl úvěr v datech dvakrát.
    # Ne přesně: váhy dostane jen model, příprava dat (medián, standardizace) se učí bez vah.
    df = fake_loans([1995], [0.3])
    X, y = df[golden_loans().columns], df["default"]
    weights = np.array([2, 1, 0] * (len(df) // 3))
    weighted = fit_weighted(make_logit, X, y, weights)
    rows = np.repeat(np.arange(len(df)), weights)
    duplicated = make_logit().fit(X.iloc[rows], y.iloc[rows])
    assert np.allclose(weighted.predict_proba(X), duplicated.predict_proba(X), atol=1e-3)


def test_bank_bootstrap_is_reproducible_and_has_right_shape():
    df = fake_loans([1995, 1996], [0.2, 0.2])
    X_eval = df[golden_loans().columns].head(7)
    d1 = bank_bootstrap_deviations(make_logit, df, X_eval, n_boot=3, seed=1)
    d2 = bank_bootstrap_deviations(make_logit, df, X_eval, n_boot=3, seed=1)
    assert d1.shape == (3, 7)
    assert np.array_equal(d1, d2)


def test_time_offsets_are_positive_when_risk_grows():
    # Riziko každý rok roste → model naučený na minulosti ho podhodnotí → kladná odchylka.
    df = fake_loans([1990, 1991, 1992, 1993], [0.05, 0.10, 0.20, 0.40])
    offsets = time_offsets(make_logit, df, first_cutoff=1991, last_year=1993, horizon=2)
    assert offsets[["cutoff", "year"]].values.tolist() == [[1991, 1992], [1991, 1993], [1992, 1993]]
    assert (offsets["offset"] > 0).all()


def test_interval_shifts_quantiles():
    s = interval_shifts(np.zeros((1, 1)), np.arange(11.0), level=0.8)
    assert s["shift"] == 5
    assert np.isclose(s["low"], 1)
    assert np.isclose(s["high"], 9)


def test_apply_interval_orders_low_risk_high():
    out = apply_interval(np.array([0.01, 0.2, 0.7]), {"shift": 0.5, "low": -0.2, "high": 1.0})
    assert (out["low"] <= out["risk"]).all() and (out["risk"] <= out["high"]).all()
    assert ((out > 0) & (out < 1)).all().all()


def test_apply_interval_without_shift_keeps_prediction():
    p = np.array([0.1, 0.4])
    out = apply_interval(p, {"shift": 0.0, "low": 0.0, "high": 0.0})
    assert np.allclose(out["risk"], p)


def test_coverage_detects_good_and_bad_interval():
    df = fake_loans([2004], [0.3], n=2000)
    p = np.full(len(df), 0.3) + np.linspace(0, 1e-3, len(df))  # skoro konstantní predikce
    wide = coverage(df, p, {"shift": 0, "low": -1, "high": 1}, n_bins=4)
    narrow_wrong = coverage(df, p, {"shift": 2, "low": 1.9, "high": 2.1}, n_bins=4)
    assert wide["covered"].all()
    assert not narrow_wrong["covered"].any()


@pytest.mark.skipif(not UNCERTAINTY_PATH.exists(), reason="nejistota ještě není spočítaná")
def test_saved_interval_is_sensible():
    s = json.loads(UNCERTAINTY_PATH.read_text())
    assert s["level"] == 0.8
    assert s["low"] < s["shift"] < s["high"]
    # Bod 5 zadání: 80% intervaly mají na testu pokrýt zhruba 80 %.
    assert 0.7 <= s["test_coverage"] <= 0.9


def test_plot_coverage_saves_figure(tmp_path):
    df = fake_loans([2004, 2005], [0.2, 0.3], n=500)
    cells = coverage(df, np.linspace(0.05, 0.5, len(df)), {"shift": 0, "low": -1, "high": 1})
    plot_coverage(cells, tmp_path / "c.png")
    assert (tmp_path / "c.png").stat().st_size > 0

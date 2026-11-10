"""Testy modelů: základ, logistická regrese, boosting, metriky, uložený model."""

import joblib
import numpy as np
import pandas as pd
import pytest

from sba_risk.features import FEATURES, LEAKAGE
from sba_risk.model import (
    GOLDEN_PATH,
    GRIDS,
    MODEL_PATH,
    SectorRate,
    boosting,
    evaluate,
    golden_loans,
    input_ranges,
    logit,
    plot_calibration,
    tune,
)


def fake_data(n: int = 400, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """Vymyšlené úvěry, u kterých riziko roste s podílem záruky a je vyšší v Maloobchodu."""
    rng = np.random.default_rng(seed)
    X = golden_loans().sample(n, replace=True, random_state=seed).reset_index(drop=True)
    X["sba_share"] = rng.uniform(0.5, 0.9, n)
    X["sector"] = rng.choice(["Maloobchod", "Výroba"], n)
    p = 0.05 + 0.5 * (X["sba_share"] - 0.5) + 0.2 * (X["sector"] == "Maloobchod")
    y = pd.Series((rng.uniform(size=n) < p).astype(int), name="default")
    return X, y


def test_sector_rate_is_share_of_defaults_in_sector():
    X = pd.DataFrame({"sector": ["A", "A", "A", "B"]})
    y = pd.Series([1, 0, 0, 1])
    p = SectorRate().fit(X, y).predict_proba(X)[:, 1]
    assert np.allclose(p, [1 / 3, 1 / 3, 1 / 3, 1])


def test_sector_rate_unknown_sector_gets_overall_rate():
    model = SectorRate().fit(pd.DataFrame({"sector": ["A", "B"]}), pd.Series([0, 1]))
    assert model.predict_proba(pd.DataFrame({"sector": ["C"]}))[0, 1] == 0.5


@pytest.mark.parametrize("make_model", [logit, boosting])
def test_models_return_probabilities(make_model):
    X, y = fake_data()
    p = make_model().fit(X, y).predict_proba(X)
    assert p.shape == (len(X), 2)
    assert np.all((p >= 0) & (p <= 1))
    assert np.allclose(p.sum(axis=1), 1)


@pytest.mark.parametrize("make_model", [logit, boosting])
def test_models_handle_missing_and_unknown_values(make_model):
    # Model nesmí spadnout na chybějící hodnotu ani na kategorii, kterou nikdy neviděl.
    X, y = fake_data()
    model = make_model().fit(X, y)
    new = X.head(1).copy()
    new["new_business"] = np.nan
    # Kategorie musí zůstat textová (dtype str). Kdyby se z ní stalo číslo, model spadne –
    # o správné typy vstupů se postará veřejné rozhraní v kroku 8.
    new["rev_line_cr"] = pd.Series([None], dtype="str", index=new.index)
    new["state"] = "Atlantida"
    p = model.predict_proba(new)[0, 1]
    assert 0 < p < 1


@pytest.mark.parametrize("make_model", [logit, boosting])
def test_models_learn_the_signal(make_model):
    X, y = fake_data(2000)
    X_new, y_new = fake_data(2000, seed=1)
    p = make_model().fit(X, y).predict_proba(X_new)[:, 1]
    assert evaluate(y_new, p)["auc"] > 0.6


def test_boosting_is_reproducible():
    X, y = fake_data()
    p1 = boosting().fit(X, y).predict_proba(X)
    p2 = boosting().fit(X, y).predict_proba(X)
    assert np.array_equal(p1, p2)


def test_evaluate():
    y = pd.Series([0, 0, 1, 1])
    m = evaluate(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert m["auc"] == 1.0
    assert np.isclose(m["brier"], (0.01 + 0.04 + 0.04 + 0.01) / 4)
    assert m["actual"] == 0.5
    assert np.isclose(m["mean_pred"], 0.5)


def test_golden_loans_have_only_features():
    assert list(golden_loans().columns) == FEATURES
    assert not set(LEAKAGE) & set(golden_loans().columns)


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="model ještě není natrénovaný")
def test_saved_model_gives_saved_predictions():
    # Bod 1 zadání Part III: uložený model vrací pro vzorové úvěry pořád stejná čísla.
    golden = pd.read_csv(GOLDEN_PATH, keep_default_na=False, na_values=[""])
    model = joblib.load(MODEL_PATH)
    p = model.predict_proba(golden[FEATURES])[:, 1]
    assert np.allclose(p, golden["prediction"], rtol=0, atol=1e-12)


def test_tune_picks_setting_with_best_validation_auc():
    X, y = fake_data(600)
    X_va, y_va = fake_data(600, seed=1)
    params, results = tune("logit", X.assign(default=y), X_va.assign(default=y_va))
    assert len(results) == len(GRIDS["logit"])
    assert params == results.loc[results["auc"].idxmax(), "params"]


def test_input_ranges():
    X, _ = fake_data()
    df = X.assign(gr_appv=np.linspace(1_000, 2_000_000, len(X)), no_emp=5, create_job=0,
                  retained_job=np.arange(len(X)))
    ranges = input_ranges(df)
    low, high = ranges["numeric"]["gr_appv"]
    assert 1_000 < low < high < 2_000_000  # percentily, ne minimum a maximum
    assert ranges["categories"]["sector"] == ["Maloobchod", "Výroba"]


def test_plot_calibration_saves_figure(tmp_path):
    y = pd.Series([0, 1] * 50)
    plot_calibration(y, {"model": np.linspace(0.1, 0.9, 100)}, tmp_path / "c.png")
    assert (tmp_path / "c.png").stat().st_size > 0

"""Testy veřejného rozhraní predict_risk: správné odpovědi, špatné vstupy, varování."""

import numpy as np
import pandas as pd
import pytest

import sba_risk
from sba_risk import FIELDS, RiskWarning, predict_many, predict_risk
from sba_risk.api import REQUIRED, to_features
from sba_risk.features import FEATURES, LEAKAGE
from sba_risk.model import GOLDEN_PATH, golden_loans

# Stejné úvěry jako první čtyři v golden_loans(), ale tak, jak je zadá uživatel.
GOLDEN_INPUTS = [
    {"amount": 100_000, "sba_share": 0.75, "naics": 44, "state": "CA", "bank_state": "CA",
     "employees": 5, "new_jobs": 0, "retained_jobs": 5, "new_business": False, "revolving": "N"},
]
GOLDEN_INPUTS += [
    {**GOLDEN_INPUTS[0], "naics": "31", "state": "TX", "bank_state": "TX", "sba_share": 0.5},
    {**GOLDEN_INPUTS[0], "amount": 15_000, "new_business": True, "revolving": True, "naics": 722511},
    {**GOLDEN_INPUTS[0], "amount": 1_500_000, "employees": 60},
]


def loan(**changes) -> dict:
    return {**GOLDEN_INPUTS[0], **changes}


def test_version():
    assert sba_risk.__version__ == "0.1.0"


def test_inputs_give_same_features_as_golden_loans():
    for i, inputs in enumerate(GOLDEN_INPUTS):
        X, _ = to_features(inputs)
        pd.testing.assert_frame_equal(X, golden_loans().iloc[[i]].reset_index(drop=True),
                                      check_dtype=False)


@pytest.mark.filterwarnings("ignore::sba_risk.RiskWarning")  # 4. úvěr (1,5 mil.) je nad běžným rozsahem
def test_predict_risk_returns_saved_model_predictions():
    # Bod 1 zadání Part III: balíček vrací stejné predikce jako model.
    golden = pd.read_csv(GOLDEN_PATH)
    for i, inputs in enumerate(GOLDEN_INPUTS):
        assert predict_risk(inputs)["probability"] == pytest.approx(golden["prediction"][i], abs=1e-12)


def test_output():
    out = predict_risk(loan())
    assert 0 < out["low"] < out["risk"] < out["high"] < 1
    assert out["decision"] in ("schválit", "zamítnout")
    assert out["decision"] == ("schválit" if out["risk"] < out["break_even"] else "zamítnout")
    assert out["warnings"] == []


def test_same_input_same_output():
    assert predict_risk(loan()) == predict_risk(loan())


def test_higher_margin_is_more_willing_to_approve():
    assert predict_risk(loan(), margin=0.2)["break_even"] > predict_risk(loan(), margin=0.05)["break_even"]


def test_inputs_never_contain_leakage():
    # Bod 2 zadání: uživatel ani nemůže zadat údaje známé až po schválení.
    assert not {"term", "term_months", "chgoff_date", "chgoff_prin_gr", "disbursement_gross",
                "default"} & set(FIELDS)
    X, _ = to_features(loan())
    assert list(X.columns) == FEATURES
    assert not set(LEAKAGE) & set(X.columns)


@pytest.mark.parametrize("key", REQUIRED)
def test_missing_required_value_is_refused(key):
    with pytest.raises(ValueError, match=key):
        predict_risk(loan(**{key: None}))


@pytest.mark.parametrize("changes", [
    {"amount": -1}, {"amount": 0}, {"amount": "100000"}, {"amount": float("nan")},
    {"sba_share": 75}, {"sba_share": 0}, {"employees": -3}, {"new_jobs": True},
    {"state": "XX"}, {"state": "California"}, {"bank_state": 5},
    {"naics": 99}, {"naics": "abc"}, {"revolving": "maybe"}, {"new_business": "yes"},
    {"year": "2005"}, {"typo_amount": 1},
])
def test_bad_input_is_refused(changes):
    # Bod 3 zadání: nesmysl odmítnout, nikdy tiše nevrátit číslo.
    with pytest.raises(ValueError):
        predict_risk(loan(**changes))


def test_not_a_dict_is_refused():
    with pytest.raises(TypeError):
        predict_risk([100_000, 0.75])


def test_bad_margin_is_refused():
    with pytest.raises(ValueError):
        predict_risk(loan(), margin=10)


def test_percent_instead_of_share_gets_a_hint():
    with pytest.raises(ValueError, match="ne procenta"):
        predict_risk(loan(sba_share=75))


@pytest.mark.parametrize("changes, text", [
    ({"amount": 5_000_000}, "mimo běžný rozsah"),
    ({"employees": 5000}, "mimo běžný rozsah"),
    ({"revolving": None}, "podhodnocené"),
    ({"new_business": None}, "podhodnocené"),
    ({"year": 2024}, "mimo období"),
    ({"year": 1985}, "mimo období"),
])
def test_unusual_input_warns(changes, text):
    with pytest.warns(RiskWarning, match=text):
        out = predict_risk(loan(**changes))
    assert any(text in w for w in out["warnings"])
    assert 0 < out["risk"] < 1


def test_state_is_case_insensitive():
    assert predict_risk(loan(state="ca", bank_state=" ca")) == predict_risk(loan())


def test_naics_six_digits_equals_two_digits():
    assert predict_risk(loan(naics=445110)) == predict_risk(loan(naics="44"))


@pytest.mark.filterwarnings("ignore::sba_risk.RiskWarning")  # 4. úvěr (1,5 mil.) je nad běžným rozsahem
def test_predict_many():
    out = predict_many(GOLDEN_INPUTS)
    assert len(out) == len(GOLDEN_INPUTS)
    assert np.allclose(out["probability"], pd.read_csv(GOLDEN_PATH)["prediction"][:4])

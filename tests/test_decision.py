"""Testy rozhodnutí: práh podle zisku, skutečný zisk, pravidla, porovnání."""

import numpy as np
import pandas as pd

from sba_risk.decision import (
    break_even_risk,
    compare,
    loss_given_default,
    plot_profit_by_threshold,
    realized_profit,
    rules,
)


def loans() -> pd.DataFrame:
    """Tři vymyšlené úvěry: dva splacené, jeden odepsaný."""
    return pd.DataFrame({
        "gr_appv": [100_000.0, 200_000.0, 50_000.0],
        "sba_share": [0.5, 0.75, 0.8],
        "default": [0, 0, 1],
        "chgoff_prin_gr": [0.0, np.nan, 40_000.0],
    })


def test_break_even_risk():
    # 0,1 / (0,1 + 0,5 · 0,5) = 0,2857
    assert np.isclose(break_even_risk(0.5, margin=0.1, lgd=0.5), 0.1 / 0.35)


def test_expected_profit_is_zero_at_break_even():
    p = break_even_risk(0.6, margin=0.08, lgd=0.7)
    expected = (1 - p) * 0.08 - p * 0.7 * (1 - 0.6)
    assert np.isclose(expected, 0)


def test_higher_sba_guarantee_allows_more_risk():
    t = break_even_risk(np.array([0.5, 0.75, 1.0]), margin=0.1, lgd=0.6)
    assert t[0] < t[1] < t[2]
    assert t[2] == 1  # SBA ručí za vše → banka nemůže prodělat


def test_loss_given_default_uses_only_defaulted_loans():
    assert np.isclose(loss_given_default(loans()), 40_000 / 50_000)


def test_realized_profit():
    profit = realized_profit(loans(), margin=0.1)
    # Splacené: 10 % z částky. Odepsaný: banka ztratí 20 % (neručí SBA) z 40 000.
    assert np.allclose(profit, [10_000, 20_000, -8_000])


def test_rules_approve_all_and_extremes():
    share = np.array([0.5, 0.5])
    approved = rules(p=np.array([0.0, 0.99]), risk=np.array([0.0, 0.99]),
                     high=np.array([0.0, 0.99]), sba_share=share, margin=0.1, lgd=0.6)
    assert approved["schválit vše"].all()
    for name in ("zisk > 0, model bez posunu", "zisk > 0, posunuté riziko",
                 "zisk > 0, horní hranice intervalu"):
        assert approved[name].tolist() == [True, False]


def test_compare():
    df = loans()
    approved = {"schválit vše": np.array([True, True, True]),
                "bez odepsaného": np.array([True, True, False])}
    out = compare(df, approved, margin=0.1).set_index("pravidlo")
    assert np.isclose(out.loc["schválit vše", "zisk_mil_usd"], 0.022)
    assert out.loc["schválit vše", "proti_schvalit_vse_mil_usd"] == 0
    assert np.isclose(out.loc["bez odepsaného", "proti_schvalit_vse_mil_usd"], 0.008)
    assert out.loc["bez odepsaného", "odepsano_ze_schvalenych"] == 0
    assert np.isclose(out.loc["bez odepsaného", "schvaleno"], 2 / 3)


def test_plot_profit_by_threshold_saves_figure(tmp_path):
    plot_profit_by_threshold(loans(), np.array([0.1, 0.2, 0.9]), margin=0.1, lgd=0.6,
                             path=tmp_path / "p.png")
    assert (tmp_path / "p.png").stat().st_size > 0

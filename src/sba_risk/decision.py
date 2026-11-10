"""Rozhodnutí: schválit, nebo zamítnout úvěr (otázka 3 z Part II).

Banka u splaceného úvěru vydělá úrokovou marži, u odepsaného přijde o tu část odepsané
částky, za kterou neručí SBA. Marži v datech nemáme, proto zkoušíme několik variant.

Pravidla jen **zpřísňují**: zamítají typy úvěrů, které banky dřív schvalovaly. Jak by
dopadly úvěry, které banky zamítly, z dat nevíme (Part II, otázka 3).

Pravidla porovnáme se „schválit vše“ na úvěrech z let 2004–06. Skutečný výsledek
(`default`, `chgoff_prin_gr`) používáme jen na vyhodnocení, nikdy k rozhodnutí.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sba_risk.features import TARGET, X_y
from sba_risk.model import FIGURES

# Celkový výnos banky ze splaceného úvěru za celou dobu splácení, jako podíl z částky.
# Např. marže 2,5 % ročně × průměrně 4 roky ≈ 10 %. Neznáme ji, proto tři varianty.
MARGINS = (0.05, 0.10, 0.20)
LI_THRESHOLD = 0.5  # pravidlo z článku Li et al. (2018): zamítnout, když riziko > 50 %
DECISION_PATH = "reports/decision.csv"


def loss_given_default(df: pd.DataFrame) -> float:
    """Jaká část úvěru se při odepsání v průměru ztratí (odepsaná částka / schválená
    částka). Počítá se z minulých let; banka ji zná předem."""
    d = df[df[TARGET] == 1]
    return float((d["chgoff_prin_gr"] / d["gr_appv"]).mean())


def break_even_risk(sba_share, margin: float, lgd: float):
    """Riziko, při kterém je očekávaný zisk banky přesně nula.

    Očekávaný zisk = (1 − p) · marže · částka − p · LGD · (1 − záruka SBA) · částka.
    Položíme = 0 a vyjádříme p. Částka se vykrátí, takže práh závisí jen na záruce:
    čím víc ručí SBA, tím víc rizika si banka může dovolit."""
    return margin / (margin + lgd * (1 - np.asarray(sba_share)))


def realized_profit(df: pd.DataFrame, margin: float) -> np.ndarray:
    """Skutečný zisk banky z každého úvěru (jen na vyhodnocení, používá výsledek).
    Splacený: + marže · částka. Odepsaný: − (1 − záruka SBA) · odepsaná částka."""
    loss = (1 - df["sba_share"]) * df["chgoff_prin_gr"].fillna(0)
    gain = margin * df["gr_appv"]
    return np.where(df[TARGET] == 1, -loss, gain)


def rules(p: np.ndarray, risk: np.ndarray, high: np.ndarray, sba_share, margin: float,
          lgd: float) -> dict[str, np.ndarray]:
    """Rozhodovací pravidla. Vrací pro každé pravidlo pole True = schválit.

    p: predikce modelu, risk: posunutá predikce z kroku 6, high: horní hranice 80% intervalu."""
    threshold = break_even_risk(sba_share, margin, lgd)
    return {
        "schválit vše": np.ones(len(p), dtype=bool),
        "Li et al.: riziko modelu < 50 %": p < LI_THRESHOLD,
        "zisk > 0, model bez posunu": p < threshold,
        "zisk > 0, posunuté riziko": risk < threshold,
        "zisk > 0, horní hranice intervalu": high < threshold,
    }


def compare(df: pd.DataFrame, approved: dict[str, np.ndarray], margin: float) -> pd.DataFrame:
    """Pro každé pravidlo: kolik úvěrů schválí, kolik z nich se odepíše a skutečný zisk."""
    profit = realized_profit(df, margin)
    y = df[TARGET].to_numpy()
    base = profit.sum()
    rows = []
    for name, ok in approved.items():
        rows.append({
            "marze": margin,
            "pravidlo": name,
            "schvaleno": ok.mean(),
            "odepsano_ze_schvalenych": y[ok].mean() if ok.any() else np.nan,
            "zisk_mil_usd": profit[ok].sum() / 1e6,
            "proti_schvalit_vse_mil_usd": (profit[ok].sum() - base) / 1e6,
        })
    return pd.DataFrame(rows)


def plot_profit_by_threshold(df: pd.DataFrame, risk: np.ndarray, margin: float, lgd: float,
                             path) -> None:
    """Zisk na testu, když zamítneme vše nad pevným prahem rizika. Ukáže, kde leží
    pravidlo 50 % a kde náš práh podle zisku."""
    profit = realized_profit(df, margin)
    thresholds = np.linspace(0.05, 1.0, 96)
    totals = [profit[risk < t].sum() / 1e9 for t in thresholds]
    typical = float(np.median(break_even_risk(df["sba_share"], margin, lgd)))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(thresholds, totals)
    ax.axhline(profit.sum() / 1e9, color="grey", linestyle="--", label="schválit vše")
    ax.axvline(LI_THRESHOLD, color="tab:red", linestyle=":", label="práh 50 % (jako Li et al.)")
    ax.axvline(typical, color="tab:green", linestyle=":", label=f"práh podle zisku (medián {typical:.0%})")
    ax.set_xlabel("zamítnout, když posunuté riziko ≥ práh")
    ax.set_ylabel("zisk banky na testu (mld. USD)")
    ax.set_title(f"Úvěry 2004–06, marže {margin:.0%}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    import joblib

    from sba_risk.data import CLEAN_PATH
    from sba_risk.features import add_features, split_by_year
    from sba_risk.model import MODEL_PATH
    from sba_risk.uncertainty import UNCERTAINTY_PATH, apply_interval

    parts = split_by_year(add_features(pd.read_parquet(CLEAN_PATH)))
    test = parts["test"]
    lgd = loss_given_default(pd.concat([parts["train"], parts["valid"]]))
    print(f"Ztráta při odepsání (1990–2003): {lgd:.1%} schválené částky")

    p = joblib.load(MODEL_PATH).predict_proba(X_y(test)[0])[:, 1]
    interval = apply_interval(p, json.loads(UNCERTAINTY_PATH.read_text()))
    risk, high = interval["risk"].to_numpy(), interval["high"].to_numpy()

    results = pd.concat([
        compare(test, rules(p, risk, high, test["sba_share"], m, lgd), m) for m in MARGINS
    ], ignore_index=True)
    results.to_csv(DECISION_PATH, index=False)
    pd.set_option("display.width", 200)
    for m, r in results.groupby("marze"):
        print(f"\nMarže {m:.0%} (test 2004–06, {len(test):,} úvěrů):")
        print(r.drop(columns="marze").round(3).to_string(index=False))

    # Které typy úvěrů pravidlo přestane schvalovat (marže 10 %, posunuté riziko).
    ok = risk < break_even_risk(test["sba_share"], 0.10, lgd)
    by_sector = (test.assign(zamitnuto=~ok).groupby("sector")
                 .agg(uvery=(TARGET, "size"), odepsano=(TARGET, "mean"), zamitnuto=("zamitnuto", "mean"))
                 .sort_values("zamitnuto", ascending=False))
    print("\nPodíl zamítnutých podle odvětví (marže 10 %, posunuté riziko):")
    print(by_sector.round(3).to_string())

    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_profit_by_threshold(test, risk, 0.10, lgd, FIGURES / "7_profit_by_threshold.png")


if __name__ == "__main__":
    main()

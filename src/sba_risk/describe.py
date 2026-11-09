"""Popisné otázky z Part II.

1A (vzorek): kolik procent úvěrů vyplacených 1987–2010 skončilo odepsáním, podle odvětví.
1B (populace): pravděpodobnost odepsání úvěru, jehož splatnost skončila do června 2014,
    s 95% intervalem spolehlivosti počítaným bootstrapem po bankách.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sba_risk.data import CLEAN_PATH

FIGURES = Path("reports/figures")
MATURITY_CUTOFF = pd.Timestamp("2014-06-30")
MISSING_BANK = "(banka neuvedena)"


def start_year(df: pd.DataFrame) -> pd.Series:
    """Rok vyplacení; když datum vyplacení chybí, použije rok schválení."""
    return df["disbursement_date"].fillna(df["approval_date"]).dt.year


def default_rate_by_sector(df: pd.DataFrame, first: int = 1987, last: int = 2010) -> pd.DataFrame:
    """Otázka 1A: počet úvěrů, počet odepsaných a jejich podíl v každém odvětví."""
    year = start_year(df)
    sample = df[(year >= first) & (year <= last)]
    table = sample.groupby("sector")["default"].agg(loans="size", defaults="sum")
    table["rate"] = table["defaults"] / table["loans"]
    return table.sort_values("rate", ascending=False)


def default_rate_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """Podíl odepsaných podle roku vyplacení: ukazuje zkreslení posledních ročníků."""
    table = df.groupby(start_year(df))["default"].agg(loans="size", rate="mean")
    table.index.name = "year"
    return table


def matured_loans(df: pd.DataFrame, cutoff: pd.Timestamp = MATURITY_CUTOFF) -> pd.DataFrame:
    """Otázka 1B: úvěry, které podle plánu měly doběhnout do `cutoff`.

    Výběr závisí jen na datu a splatnosti, tedy na údajích známých při schválení,
    ne na výsledku. Proto nezkresluje podíl odepsaných jako vypadlé běžící úvěry.
    """
    return df[df["maturity_date"] <= cutoff]


def cluster_bootstrap_rate(
    df: pd.DataFrame, cluster: str = "bank", n_boot: int = 2000, seed: int = 0
) -> dict:
    """Podíl odepsaných s 95% intervalem: naivním a shlukovým (bootstrap po bankách).

    Naivní interval předpokládá, že úvěry jsou nezávislé. Úvěry jedné banky ale
    dopadají podobně, proto losujeme s opakováním celé banky, ne jednotlivé úvěry.
    """
    groups = df.groupby(df[cluster].fillna(MISSING_BANK))["default"].agg(["size", "sum"])
    sizes = groups["size"].to_numpy()
    defaults = groups["sum"].to_numpy()

    rate = defaults.sum() / sizes.sum()
    naive_se = np.sqrt(rate * (1 - rate) / sizes.sum())

    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(groups), size=(n_boot, len(groups)))
    boot_rates = defaults[picks].sum(axis=1) / sizes[picks].sum(axis=1)
    low, high = np.percentile(boot_rates, [2.5, 97.5])

    return {
        "loans": int(sizes.sum()),
        "clusters": len(groups),
        "rate": rate,
        "naive_low": rate - 1.96 * naive_se,
        "naive_high": rate + 1.96 * naive_se,
        "cluster_low": low,
        "cluster_high": high,
    }


def plot_figures(df: pd.DataFrame, sectors: pd.DataFrame) -> None:
    """Uloží dva grafy do reports/figures/."""
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    ordered = sectors.sort_values("rate")
    colors = ["#999999" if s == "Neuvedeno" else "#2a6f97" for s in ordered.index]
    ax.barh(ordered.index, ordered["rate"] * 100, color=colors)
    ax.set_xlabel("Podíl odepsaných úvěrů (%)")
    ax.set_title("Odepsané úvěry podle odvětví, vyplaceno 1987–2010")
    fig.tight_layout()
    fig.savefig(FIGURES / "1a_default_by_sector.png", dpi=150)
    plt.close(fig)

    years = default_rate_by_year(df).loc[1987:2014]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(years.index, years["rate"] * 100, marker="o", color="#2a6f97")
    ax.axvspan(2010.5, 2014.5, color="#cccccc", alpha=0.5, label="vynecháno v 1A")
    ax.set_xlabel("Rok vyplacení")
    ax.set_ylabel("Podíl odepsaných (%)")
    ax.set_title("Podíl odepsaných podle roku vyplacení")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "1a_default_by_year.png", dpi=150)
    plt.close(fig)


def main() -> None:
    df = pd.read_parquet(CLEAN_PATH)

    sectors = default_rate_by_sector(df)
    total = sectors["defaults"].sum() / sectors["loans"].sum()
    print("1A – úvěry vyplacené 1987–2010")
    print(f"Celkem {sectors['loans'].sum():,} úvěrů, odepsáno {total:.1%}\n")
    print(sectors.assign(rate=sectors["rate"].map("{:.1%}".format)).to_string())

    result = cluster_bootstrap_rate(matured_loans(df))
    print("\n1B – úvěry se splatností do 30. 6. 2014")
    print(f"{result['loans']:,} úvěrů od {result['clusters']:,} bank")
    print(f"Podíl odepsaných: {result['rate']:.2%}")
    print(f"95% interval naivní:          {result['naive_low']:.2%} – {result['naive_high']:.2%}")
    print(f"95% interval po bankách:      {result['cluster_low']:.2%} – {result['cluster_high']:.2%}")

    plot_figures(df, sectors)
    print(f"\nGrafy uloženy do {FIGURES}/")


if __name__ == "__main__":
    main()

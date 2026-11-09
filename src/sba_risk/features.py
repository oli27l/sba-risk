"""Příprava dat pro model (otázka 2 z Part II).

Model smí vidět jen to, co banka zná v okamžiku schválení úvěru. Data dělíme podle
roku schválení: učíme na starších letech a testujeme na pozdějších, stejně jako by
se model používal ve skutečnosti.
"""

import numpy as np
import pandas as pd

TARGET = "default"

# Proměnné známé při schválení, které model dostane.
NUMERIC = [
    "log_gr_appv",      # log schválené částky (USD)
    "sba_share",        # podíl záruky SBA (0–1)
    "term_months",      # splatnost v měsících
    "log_no_emp",       # log(1 + počet zaměstnanců)
    "create_job",       # nová pracovní místa podle žádosti
    "retained_job",     # udržená pracovní místa podle žádosti
    "new_business",     # 1 = nová firma
    "franchise",        # 1 = franšíza
    "real_estate",      # 1 = splatnost >= 20 let (kryto nemovitostí)
    "low_doc",          # 1 = program LowDoc
    "same_state",       # 1 = banka sídlí ve stejném státě jako firma
]
CATEGORICAL = [
    "sector",           # odvětví (včetně „Neuvedeno“)
    "state",            # stát firmy
    "rev_line_cr",      # revolvingový úvěr: Y / N / 0 / T
]
FEATURES = NUMERIC + CATEGORICAL

# Údaje známé až po schválení. Kdyby se dostaly do modelu, „věštil“ by z budoucnosti.
LEAKAGE = ["chgoff_date", "chgoff_prin_gr", "disbursement_gross", "disbursement_date"]

# Záměrně vynechané, i když jsou známé při schválení:
# - urban_rural: v letech 1990–2000 je u 99 % úvěrů „unknown“, model by se nic nenaučil.
# - approval_fy, approval_date: rok sám o sobě nejde použít na budoucí roky.
# - bank: nová banka by v trénovacích datech nebyla; tisíce kategorií.

# Dělení podle roku schválení (Part II, otázka 2).
SPLITS = {
    "train": (1990, 2000),
    "valid": (2001, 2003),
    "test": (2004, 2006),
}
MATURITY_CUTOFF = pd.Timestamp("2014-06-30")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Přidá odvozené proměnné. Každá se počítá jen z údajů jednoho úvěru,
    nikdy z celého datasetu, aby nevznikl únik informací mezi trénovacími a testovacími daty."""
    out = df.copy()
    out["log_gr_appv"] = np.log(out["gr_appv"])
    out["log_no_emp"] = np.log1p(out["no_emp"])
    out["same_state"] = (out["state"] == out["bank_state"]).astype(int)
    return out


def split_by_year(
    df: pd.DataFrame, splits: dict = SPLITS, matured_only: bool = True
) -> dict[str, pd.DataFrame]:
    """Rozdělí úvěry podle roku schválení na train / valid / test.

    `matured_only`: ponechá jen úvěry, které měly podle plánu doběhnout do června 2014.
    V datech chybí úvěry, které v roce 2014 ještě běžely. U úvěru se splatností po roce
    2014 tedy vidíme jen ty, které skončily dřív – předčasným splacením nebo odepsáním.
    Takový výběr závisí na výsledku a zkresluje ho. Výběr podle plánované splatnosti
    závisí jen na údajích známých při schválení.
    """
    year = df["approval_date"].dt.year
    if matured_only:
        df = df[df["maturity_date"] <= MATURITY_CUTOFF]
        year = year.loc[df.index]
    return {name: df[(year >= first) & (year <= last)] for name, (first, last) in splits.items()}


def X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Vrátí vstupy modelu (X) a cíl (y). X obsahuje jen sloupce z FEATURES."""
    return df[FEATURES], df[TARGET]


def main() -> None:
    from sba_risk.data import CLEAN_PATH

    df = add_features(pd.read_parquet(CLEAN_PATH))
    parts = split_by_year(df)
    print(f"{'část':6} {'roky':10} {'úvěry':>8} {'odepsáno':>9}")
    for name, part in parts.items():
        first, last = SPLITS[name]
        print(f"{name:6} {first}–{last}  {len(part):8,} {part[TARGET].mean():9.1%}")

    X, _ = X_y(parts["train"])
    print("\nChybějící hodnoty v trénovacích datech:")
    print(X.isna().mean()[lambda s: s > 0].map("{:.1%}".format).to_string())


if __name__ == "__main__":
    main()

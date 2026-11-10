"""Nejistota predikce (otázka 2 z Part II, bod 5 zadání Part III).

Interval pro pravděpodobnost odepsání skládáme ze dvou zdrojů nejistoty:
1. Data: kdybychom měli jiné banky, model by vyšel trochu jinak → bootstrap po bankách.
2. Čas: model se učí na minulosti, ale používá se na budoucí roky, kde je riziko jiné →
   jak moc se model v minulosti mýlil o 1–3 roky dopředu.

Obojí měříme v log-šancích (logit), kde se posuny sčítají. Výsledek jsou dvě čísla,
o která se logit predikce posune dolů a nahoru. Pokrytí 80% intervalu ověříme na testu.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sba_risk.features import SPLITS, TARGET, X_y
from sba_risk.model import ARTIFACTS, FIGURES, INFO_PATH, MODELS, SEED

UNCERTAINTY_PATH = ARTIFACTS / "uncertainty.json"
COVERAGE_PATH = "reports/coverage.csv"
MISSING_BANK = "(neuvedeno)"
LEVEL = 0.8


def logit(p):
    """Pravděpodobnost → log-šance: log(p / (1 − p)). 0,5 → 0; 0,1 → −2,2; 0,9 → 2,2."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def expit(z):
    """Log-šance → pravděpodobnost (opak funkce logit)."""
    return 1 / (1 + np.exp(-z))


def fit_weighted(make_model, X: pd.DataFrame, y: pd.Series, weights: np.ndarray):
    """Naučí model s vahami řádků. Váha 2 = úvěr je ve vzorku dvakrát, 0 = vůbec.
    Váhy dostane jen poslední krok pipeline (model), příprava dat se učí bez vah."""
    keep = weights > 0
    model = make_model()
    step = model.steps[-1][0]  # název posledního kroku pipeline, např. „histgradientboostingclassifier“
    return model.fit(X[keep], y[keep], **{f"{step}__sample_weight": weights[keep]})


def bank_bootstrap_deviations(
    make_model, df: pd.DataFrame, X_eval: pd.DataFrame, n_boot: int = 20, seed: int = SEED
) -> np.ndarray:
    """Bootstrap po bankách: n_boot× vylosuje s opakováním celé banky, naučí model a změří,
    o kolik se jeho predikce (v logitu) liší od modelu naučeného na všech datech.

    Místo kopírování dat dostane každý úvěr váhu = kolikrát byla jeho banka vylosována.
    Vrací matici odchylek: řádek = bootstrap, sloupec = úvěr z X_eval."""
    X, y = X_y(df)
    banks, _ = pd.factorize(df["bank"].fillna(MISSING_BANK))
    p_full = make_model().fit(X, y).predict_proba(X_eval)[:, 1]
    rng = np.random.default_rng(seed)
    deviations = []
    for _ in range(n_boot):
        drawn = rng.integers(0, banks.max() + 1, size=banks.max() + 1)
        weights = np.bincount(drawn, minlength=banks.max() + 1)[banks]
        p = fit_weighted(make_model, X, y, weights).predict_proba(X_eval)[:, 1]
        deviations.append(logit(p) - logit(p_full))
    return np.array(deviations)


def time_offsets(make_model, df: pd.DataFrame, first_cutoff: int, last_year: int,
                 horizon: int = 3) -> pd.DataFrame:
    """Jak moc se model mýlil o 1–3 roky dopředu v minulosti.

    Pro každý rok c: model naučený na letech 1990..c predikuje roky c+1..c+horizon.
    Odchylka = logit(skutečný podíl odepsaných) − logit(průměrná predikce). Kladná =
    model riziko podhodnotil. Používá jen roky do last_year, test se nedotkne."""
    year = df["approval_date"].dt.year
    first = SPLITS["train"][0]
    rows = []
    for cutoff in range(first_cutoff, last_year):
        model = make_model().fit(*X_y(df[(year >= first) & (year <= cutoff)]))
        for ahead in range(1, horizon + 1):
            target_year = cutoff + ahead
            if target_year > last_year:
                break
            X, y = X_y(df[year == target_year])
            p = model.predict_proba(X)[:, 1]
            rows.append({"cutoff": cutoff, "year": target_year, "ahead": ahead,
                         "actual": y.mean(), "mean_pred": p.mean(),
                         "offset": logit(y.mean()) - logit(p.mean())})
    return pd.DataFrame(rows)


def interval_shifts(deviations: np.ndarray, offsets: np.ndarray, level: float = LEVEL) -> dict:
    """Spojí oba zdroje: každou odchylku z bootstrapu sečte s každou odchylkou v čase
    a vezme kvantily. Např. pro 80 % je to 10. a 90. percentil."""
    combined = (deviations.ravel()[:, None] + offsets[None, :]).ravel()
    alpha = (1 - level) / 2
    return {
        "level": level,
        "shift": float(np.median(combined)),
        "low": float(np.quantile(combined, alpha)),
        "high": float(np.quantile(combined, 1 - alpha)),
    }


def apply_interval(p: np.ndarray, shifts: dict) -> pd.DataFrame:
    """Posunutá predikce a interval pro pravděpodobnosti p z modelu."""
    z = logit(p)
    return pd.DataFrame({
        "risk": expit(z + shifts["shift"]),
        "low": expit(z + shifts["low"]),
        "high": expit(z + shifts["high"]),
    })


def coverage(df: pd.DataFrame, p: np.ndarray, shifts: dict, n_bins: int = 10) -> pd.DataFrame:
    """Kontrola pokrytí. Výsledek jednoho úvěru je 0/1, interval je pro pravděpodobnost,
    takže porovnáváme skupiny: úvěry rozdělíme podle roku a podle predikce (10 skupin)
    a pro každou skupinu se ptáme, jestli skutečný podíl odepsaných leží v intervalu."""
    intervals = apply_interval(p, shifts)
    cells = pd.DataFrame({
        "year": df["approval_date"].dt.year.to_numpy(),
        "bin": pd.qcut(p, n_bins, labels=False, duplicates="drop"),
        "actual": df[TARGET].to_numpy(),
        "pred": p,
        **{c: intervals[c].to_numpy() for c in intervals},
    })
    out = cells.groupby(["year", "bin"]).mean().reset_index()
    out["covered"] = (out["actual"] >= out["low"]) & (out["actual"] <= out["high"])
    return out


def plot_coverage(cells: pd.DataFrame, path) -> None:
    years = sorted(cells["year"].unique())
    fig, axes = plt.subplots(1, len(years), figsize=(4 * len(years), 4), sharey=True)
    for ax, year in zip(np.atleast_1d(axes), years):
        c = cells[cells["year"] == year]
        ax.fill_between(c["bin"] + 1, c["low"], c["high"], alpha=0.3, label="80% interval")
        ax.plot(c["bin"] + 1, c["pred"], "--", color="grey", label="model bez posunu")
        ax.plot(c["bin"] + 1, c["actual"], "o", color="black", label="skutečnost")
        ax.set_title(f"schváleno {year}")
        ax.set_xlabel("skupina podle predikce (1 = nejnižší)")
    np.atleast_1d(axes)[0].set_ylabel("podíl odepsaných")
    np.atleast_1d(axes)[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    import joblib

    from sba_risk.data import CLEAN_PATH
    from sba_risk.features import add_features, split_by_year
    from sba_risk.model import MODEL_PATH

    info = json.loads(INFO_PATH.read_text())
    make_model = lambda: MODELS[info["model"]](**info["params"])  # noqa: E731
    parts = split_by_year(add_features(pd.read_parquet(CLEAN_PATH)))
    past = pd.concat([parts["train"], parts["valid"]])

    sample = past.sample(20_000, random_state=SEED)
    deviations = bank_bootstrap_deviations(make_model, past, X_y(sample)[0])
    print(f"Bootstrap po bankách: směrodatná odchylka v logitu {deviations.std():.3f}")

    offsets = time_offsets(make_model, past, first_cutoff=1992, last_year=SPLITS["valid"][1])
    print("\nChyba o 1–3 roky dopředu (kladná = model riziko podhodnotil):")
    print(offsets.round(3).to_string(index=False))

    shifts = interval_shifts(deviations, offsets["offset"].to_numpy())
    only_bootstrap = interval_shifts(deviations, np.zeros(1))
    print(f"\nPosuny v logitu: {shifts}")

    model = joblib.load(MODEL_PATH)
    p_test = model.predict_proba(X_y(parts["test"])[0])[:, 1]
    rows = []
    for name, s in {"jen bootstrap": only_bootstrap, "bootstrap + čas": shifts}.items():
        cells = coverage(parts["test"], p_test, s)
        rows.append({"interval": name, "pokryti": cells["covered"].mean(),
                     "prumerna_sirka": (cells["high"] - cells["low"]).mean()})
    cells.to_csv(COVERAGE_PATH, index=False)
    plot_coverage(cells, FIGURES / "6_coverage.png")
    print(f"\nPokrytí {LEVEL:.0%} intervalů na testu (skupiny rok × predikce):")
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    risk = apply_interval(p_test, shifts)["risk"]
    print(f"\nTest: průměrná predikce {p_test.mean():.3f}, po posunu {risk.mean():.3f}, "
          f"skutečnost {parts['test'][TARGET].mean():.3f}")

    UNCERTAINTY_PATH.write_text(json.dumps(
        {**shifts, "n_boot": len(deviations), "n_offsets": len(offsets),
         "test_coverage": float(rows[-1]["pokryti"])}, indent=2) + "\n")
    print(f"\nUloženo do {UNCERTAINTY_PATH}")


if __name__ == "__main__":
    main()

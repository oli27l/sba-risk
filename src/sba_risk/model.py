"""Modely rizika odepsání (otázka 2 z Part II).

Tři modely od nejjednoduššího po nejsložitější:
1. základ – průměrná míra odepsání v odvětví,
2. logistická regrese,
3. gradient boosting (rozhodovací stromy).

Nastavení každého modelu vybíráme na validačních datech (2001–03). Vybrané nastavení pak
naučíme znovu na train + valid (1990–2003) a jednou vyhodnotíme na testu (2004–06).
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from sba_risk.features import CATEGORICAL, FEATURES, NUMERIC, SPLITS, TARGET, X_y

SEED = 0
ARTIFACTS = Path(__file__).parent / "artifacts"
MODEL_PATH = ARTIFACTS / "model.joblib"
INFO_PATH = ARTIFACTS / "model_info.json"
GOLDEN_PATH = Path("tests/golden_predictions.csv")
METRICS_PATH = Path("reports/model_metrics.csv")
FIGURES = Path("reports/figures")

# Nastavení, která zkusíme na validačních datech.
GRIDS = {
    "logit": [{"C": c} for c in (0.01, 0.1, 1.0)],
    "boosting": [
        {"learning_rate": lr, "max_leaf_nodes": leaves}
        for lr in (0.05, 0.1)
        for leaves in (15, 31, 63)
    ],
}


class SectorRate:
    """Základní model: pravděpodobnost odepsání = podíl odepsaných v odvětví v trénovacích
    datech. Neznámé odvětví dostane průměr všech úvěrů."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SectorRate":
        self.rates_ = y.groupby(X["sector"].to_numpy()).mean()
        self.overall_ = y.mean()
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = X["sector"].map(self.rates_).fillna(self.overall_).to_numpy(dtype=float)
        return np.column_stack([1 - p, p])


def logit(C: float = 1.0):
    """Logistická regrese. Chybějící čísla doplní mediánem z trénovacích dat (a přidá
    sloupec „chybělo“), čísla standardizuje, kategorie převede na 0/1 sloupce (one-hot)."""
    numeric = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler())
    categorical = OneHotEncoder(handle_unknown="ignore")
    prep = ColumnTransformer([("num", numeric, NUMERIC), ("cat", categorical, CATEGORICAL)])
    return make_pipeline(prep, LogisticRegression(C=C, max_iter=2000))


def boosting(learning_rate: float = 0.1, max_leaf_nodes: int = 31):
    """Gradient boosting. Chybějící hodnoty a kategorie umí sám, kategorie jen očísluje.
    Neznámá kategorie (např. nový stát) se bere jako chybějící hodnota."""
    categorical = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=np.nan, encoded_missing_value=np.nan
    )
    prep = ColumnTransformer([("num", "passthrough", NUMERIC), ("cat", categorical, CATEGORICAL)])
    is_categorical = [False] * len(NUMERIC) + [True] * len(CATEGORICAL)
    return make_pipeline(prep, HistGradientBoostingClassifier(
        learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes,
        categorical_features=is_categorical,
        random_state=SEED,
    ))


MODELS = {"baseline": SectorRate, "logit": logit, "boosting": boosting}


def evaluate(y: pd.Series, p: np.ndarray) -> dict:
    """AUC: jak dobře model seřadí úvěry podle rizika (0,5 = náhoda, 1 = dokonale).
    Brier: průměr (p − y)², nižší je lepší; trestá i špatně „nastavené“ pravděpodobnosti.
    Průměrná predikce vs. skutečný podíl: jestli model trefuje celkovou úroveň rizika."""
    return {
        "auc": roc_auc_score(y, p),
        "brier": brier_score_loss(y, p),
        "log_loss": log_loss(y, p),
        "mean_pred": float(np.mean(p)),
        "actual": float(np.mean(y)),
    }


def tune(name: str, train: pd.DataFrame, valid: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Zkusí všechna nastavení z GRIDS na train, vybere to s nejvyšším AUC na valid.
    AUC, protože kvůli label shiftu (6,7 % → 16,1 %) jsou pravděpodobnosti stejně posunuté."""
    X_tr, y_tr = X_y(train)
    X_va, y_va = X_y(valid)
    rows = []
    for params in GRIDS.get(name, [{}]):
        model = MODELS[name](**params).fit(X_tr, y_tr)
        rows.append({"model": name, "params": params,
                     **evaluate(y_va, model.predict_proba(X_va)[:, 1])})
    results = pd.DataFrame(rows)
    return results.loc[results["auc"].idxmax(), "params"], results


def input_ranges(df: pd.DataFrame) -> dict:
    """Rozsahy a kategorie, které model viděl při učení. Veřejné rozhraní podle nich
    varuje, když nový úvěr leží mimo (bod 3 zadání Part III). Rozsah = 0,5. až 99,5.
    percentil, protože minimum a maximum jsou často chyby v datech (9 999 zaměstnanců)."""
    numeric = ["gr_appv", "sba_share", "no_emp", "create_job", "retained_job"]
    return {
        "numeric": {c: df[c].quantile([0.005, 0.995]).round(4).tolist() for c in numeric},
        "categories": {c: sorted(df[c].dropna().unique().tolist()) for c in CATEGORICAL},
    }


def golden_loans() -> pd.DataFrame:
    """Několik vymyšlených úvěrů, na kterých si uložíme predikce. Testy pak hlídají,
    že balíček pro ně vrací pořád stejná čísla (bod 1 zadání Part III)."""
    base = {
        "log_gr_appv": np.log(100_000), "sba_share": 0.75,
        "log_no_emp": np.log1p(5), "create_job": 0, "retained_job": 5,
        "new_business": 0.0, "franchise": 0, "low_doc": 0.0,
        "same_state": 1, "sector": "Maloobchod", "state": "CA", "rev_line_cr": "N",
    }
    changes = [
        {},
        {"sector": "Výroba", "state": "TX", "sba_share": 0.5},
        {"log_gr_appv": np.log(15_000), "new_business": 1.0,
         "rev_line_cr": "Y", "sector": "Ubytování a stravování"},
        {"log_gr_appv": np.log(1_500_000), "log_no_emp": np.log1p(60)},
        {"new_business": np.nan, "rev_line_cr": np.nan, "state": "XX", "sector": "Neuvedeno"},
    ]
    return pd.DataFrame([{**base, **c} for c in changes])[FEATURES]


def plot_calibration(y: pd.Series, preds: dict[str, np.ndarray], path: Path) -> None:
    """Kalibrační graf: úvěry rozdělíme do 10 skupin podle predikce a porovnáme průměrnou
    predikci se skutečným podílem odepsaných. Dobře kalibrovaný model leží na úhlopříčce."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", label="dokonalá kalibrace")
    for name, p in preds.items():
        actual, predicted = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(predicted, actual, marker="o", label=name)
    ax.set_xlabel("průměrná predikovaná pravděpodobnost odepsání")
    ax.set_ylabel("skutečný podíl odepsaných")
    ax.set_title("Kalibrace na testu (úvěry schválené 2004–06)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    from sba_risk.data import CLEAN_PATH
    from sba_risk.decision import loss_given_default
    from sba_risk.features import add_features, split_by_year

    parts = split_by_year(add_features(pd.read_parquet(CLEAN_PATH)))
    train_valid = pd.concat([parts["train"], parts["valid"]])
    X_trva, y_trva = X_y(train_valid)
    X_te, y_te = X_y(parts["test"])

    rows, test_preds, final = [], {}, {}
    for name in MODELS:
        params, results = tune(name, parts["train"], parts["valid"])
        print(f"\n{name}: validace (učeno na {SPLITS['train']})")
        print(results.drop(columns="model").round(4).to_string(index=False))

        model = MODELS[name](**params).fit(X_trva, y_trva)
        test_preds[name] = model.predict_proba(X_te)[:, 1]
        final[name] = model
        best_valid = results.loc[results["auc"].idxmax()]
        rows.append({"model": name, "params": json.dumps(params), "valid_auc": best_valid["auc"],
                     **{f"test_{k}": v for k, v in evaluate(y_te, test_preds[name]).items()}})

    metrics = pd.DataFrame(rows)
    print("\nTest (učeno na 1990–2003, hodnoceno na 2004–06):")
    print(metrics.round(4).to_string(index=False))
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(METRICS_PATH, index=False)

    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_calibration(y_te, test_preds, FIGURES / "5_calibration.png")

    # Model vybíráme podle validace, ne podle testu – test jen hlásíme.
    best = metrics.loc[metrics["valid_auc"].idxmax(), "model"]
    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump(final[best], MODEL_PATH)
    INFO_PATH.write_text(json.dumps({
        "model": best,
        "params": json.loads(metrics.set_index("model").loc[best, "params"]),
        "trained_on_years": [SPLITS["train"][0], SPLITS["valid"][1]],
        "train_default_rate": float(y_trva.mean()),
        "loss_given_default": loss_given_default(train_valid),
        "input_ranges": input_ranges(train_valid),
        "features": FEATURES,
        "sklearn_version": sklearn.__version__,
    }, indent=2, ensure_ascii=False) + "\n")

    golden = golden_loans()
    golden["prediction"] = final[best].predict_proba(golden)[:, 1]
    golden.to_csv(GOLDEN_PATH, index=False)
    print(f"\nUložen model „{best}“ do {MODEL_PATH}")


if __name__ == "__main__":
    main()

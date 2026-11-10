"""Veřejné rozhraní balíčku: `predict_risk(loan)`.

Uživatel zadá úvěr jako slovník s čitelnými údaji (částka, záruka, odvětví…). Funkce:
1. zkontroluje vstup – nesmysl odmítne (ValueError), neobvyklé věci ohlásí (RiskWarning),
2. převede ho na proměnné modelu,
3. vrátí pravděpodobnost odepsání, 80% interval a doporučení schválit / zamítnout.

Bod 3 zadání Part III: „warn or refuse, never silently return nonsense“.
"""

import json
import math
import warnings
from collections.abc import Mapping
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from sba_risk.data import SECTORS
from sba_risk.decision import break_even_risk
from sba_risk.features import CATEGORICAL, FEATURES
from sba_risk.model import INFO_PATH, MODEL_PATH, SPLITS
from sba_risk.uncertainty import UNCERTAINTY_PATH, apply_interval

# Údaje, které funkce přijímá. Jiné klíče odmítne (ochrana proti překlepům).
FIELDS = {
    "amount": "schválená částka v USD (povinné, > 0)",
    "sba_share": "podíl záruky SBA 0–1, např. 0.75 (povinné)",
    "naics": "kód odvětví NAICS, stačí první 2 číslice, např. '44' nebo 445110 (povinné)",
    "state": "stát firmy, dvě písmena, např. 'CA' (povinné)",
    "bank_state": "stát banky, dvě písmena (povinné)",
    "employees": "počet zaměstnanců (povinné)",
    "new_jobs": "počet nových pracovních míst podle žádosti (povinné)",
    "retained_jobs": "počet udržených pracovních míst podle žádosti (povinné)",
    "new_business": "True = nová firma (do 2 let)",
    "franchise": "True = franšíza (výchozí False)",
    "low_doc": "True = program LowDoc (výchozí False)",
    "revolving": "revolvingový úvěr: True / False nebo kód 'Y', 'N', '0', 'T'",
    "year": "rok schválení (nepovinné, jen kvůli varování)",
}
# Povinné jsou údaje, které v trénovacích datech nikdy nechyběly. Model neví, co s chybějící
# hodnotou – např. bez „retained_jobs“ spadne riziko z 20 % na 5 %, což je nesmysl.
REQUIRED = ("amount", "sba_share", "naics", "state", "bank_state",
            "employees", "new_jobs", "retained_jobs")
# Tyhle v datech občas chyběly, ale hlavně u starých úvěrů s nízkým rizikem.
MISSING_NOTE = ("Chybí „{}“ – v trénovacích datech chyběl hlavně u starých úvěrů s nízkým "
                "rizikem, riziko proto může být podhodnocené.")
REVOLVING_CODES = {"Y", "N", "0", "T"}
FIRST_YEAR = SPLITS["train"][0]
LAST_TESTED_YEAR = SPLITS["test"][1]


class RiskWarning(UserWarning):
    """Vstup je neobvyklý (chybí, je mimo rozsah, neznámá kategorie) → predikce je méně
    spolehlivá. Výsledek se vrátí, ale uživatel by o tom měl vědět."""


@lru_cache
def load_artifacts() -> tuple:
    """Načte model, informace o něm a posuny pro interval. Jen jednou (lru_cache)."""
    return (
        joblib.load(MODEL_PATH),
        json.loads(INFO_PATH.read_text()),
        json.loads(UNCERTAINTY_PATH.read_text()),
    )


def _number(loan: Mapping, key: str, minimum: float = 0) -> float:
    """Povinný číselný údaj. Není číslo (text, True/False, nekonečno) nebo je pod minimem → chyba."""
    value = loan[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"„{key}“ musí být číslo, ne {value!r}.")
    if value < minimum:
        raise ValueError(f"„{key}“ nesmí být menší než {minimum}, zadáno {value}.")
    return float(value)


def _fmt(value: float) -> str:
    """Číslo pro zprávu: 1407000 → '1,407,000', 0.5 → '0.5'."""
    return f"{value:,.0f}" if abs(value) >= 1000 else f"{value:g}"


def _flag(loan: Mapping, key: str, notes: list[str], default: bool | None = None) -> float:
    """Ano/ne údaj (True/False nebo 1/0). Chybí → výchozí hodnota, nebo NaN + varování."""
    value = loan.get(key, default)
    if value is None:
        notes.append(MISSING_NOTE.format(key))
        return np.nan
    if value not in (True, False):  # 1 a 0 projdou, protože 1 == True
        raise ValueError(f"„{key}“ musí být True nebo False, ne {value!r}.")
    return float(value)


def _state(value, key: str, known: list[str]) -> str:
    """Stát jako dvě písmena. Neznámý stát (model ho nikdy neviděl) → chyba, nejspíš překlep."""
    if not isinstance(value, str) or value.strip().upper() not in known:
        raise ValueError(f"„{key}“ musí být zkratka státu USA ze dvou písmen, např. 'CA', ne {value!r}.")
    return value.strip().upper()


def _sector(value) -> str:
    """Kód NAICS → název odvětví (první 2 číslice, stejně jako v data.py)."""
    code = str(value).strip()
    if not code.isdigit() or code[:2] not in SECTORS:
        raise ValueError(f"Neznámý kód odvětví NAICS {value!r}. Zadej např. '44' (maloobchod).")
    return SECTORS[code[:2]]


def _revolving(value, notes: list[str]) -> str | None:
    if value is None:
        notes.append(MISSING_NOTE.format("revolving"))
        return None
    if value is True or value is False:
        return "Y" if value else "N"
    if value in REVOLVING_CODES:
        return value
    raise ValueError(f"„revolving“ musí být True, False nebo jeden z kódů {sorted(REVOLVING_CODES)}.")


def to_features(loan: Mapping) -> tuple[pd.DataFrame, list[str]]:
    """Zkontroluje úvěr a převede ho na jeden řádek proměnných modelu (FEATURES).
    Vrací tabulku a seznam varování. Nesmyslný vstup → ValueError."""
    if not isinstance(loan, Mapping):
        raise TypeError("Úvěr zadej jako slovník, např. {'amount': 100000, 'sba_share': 0.75}.")
    unknown = set(loan) - set(FIELDS)
    if unknown:
        raise ValueError(f"Neznámé údaje {sorted(unknown)}. Povolené: {sorted(FIELDS)}.")
    for key in REQUIRED:
        if loan.get(key) is None:
            raise ValueError(f"Chybí povinný údaj „{key}“: {FIELDS[key]}.")

    notes: list[str] = []
    amount = _number(loan, "amount")
    if amount <= 0:
        raise ValueError(f"„amount“ musí být kladná částka, zadáno {amount}.")
    sba_share = _number(loan, "sba_share")
    if not 0 < sba_share <= 1:
        hint = " Zadej podíl, ne procenta (75 % = 0.75)." if 1 < sba_share <= 100 else ""
        raise ValueError(f"„sba_share“ musí být mezi 0 a 1, zadáno {sba_share}.{hint}")

    _, info, _ = load_artifacts()
    ranges = info["input_ranges"]
    employees = _number(loan, "employees")
    state = _state(loan["state"], "state", ranges["categories"]["state"])
    bank_state = _state(loan["bank_state"], "bank_state", ranges["categories"]["state"])

    row = {
        "log_gr_appv": np.log(amount),
        "sba_share": sba_share,
        "log_no_emp": np.log1p(employees),
        "create_job": _number(loan, "new_jobs"),
        "retained_job": _number(loan, "retained_jobs"),
        "new_business": _flag(loan, "new_business", notes),
        "franchise": _flag(loan, "franchise", notes, default=False),
        "low_doc": _flag(loan, "low_doc", notes, default=False),
        "same_state": float(state == bank_state),
        "sector": _sector(loan["naics"]),
        "state": state,
        "rev_line_cr": _revolving(loan.get("revolving"), notes),
    }

    # Název údaje pro uživatele → název sloupce v datech, podle kterého známe rozsah.
    in_data = {"amount": "gr_appv", "sba_share": "sba_share", "employees": "no_emp",
               "new_jobs": "create_job", "retained_jobs": "retained_job"}
    for key, column in in_data.items():
        low, high = ranges["numeric"][column]
        if not low <= loan[key] <= high:
            notes.append(f"„{key}“ = {_fmt(loan[key])} je mimo běžný rozsah trénovacích dat "
                         f"({_fmt(low)}–{_fmt(high)}) – model tu extrapoluje.")

    year = loan.get("year")
    if year is not None:
        if isinstance(year, bool) or not isinstance(year, int):
            raise ValueError(f"„year“ musí být celé číslo, ne {year!r}.")
        if not FIRST_YEAR <= year <= LAST_TESTED_YEAR:
            notes.append(f"Rok {year} je mimo období {FIRST_YEAR}–{LAST_TESTED_YEAR}, na kterém byl "
                         "model učen a ověřen. Riziko i interval se mohly od té doby změnit.")

    X = pd.DataFrame({c: pd.Series([row[c]], dtype="str" if c in CATEGORICAL else float)
                      for c in FEATURES})
    return X, notes


def predict_risk(loan: Mapping, margin: float = 0.10) -> dict:
    """Riziko odepsání jednoho úvěru s ručením SBA.

    loan: slovník s údaji z FIELDS, povinné jsou `amount` a `sba_share`.
    margin: celkový výnos banky ze splaceného úvěru jako podíl z částky (výchozí 10 %).

    Vrací slovník:
      probability – pravděpodobnost odepsání podle modelu (bez posunu, viz krok 5),
      risk        – posunutá pravděpodobnost (zohledňuje, že model riziko podhodnocuje),
      low, high   – 80% interval pro risk,
      break_even  – riziko, nad kterým se bance úvěr nevyplatí,
      decision    – "schválit" / "zamítnout" (risk vs. break_even),
      warnings    – seznam varování (stejná varování se vypíší i jako RiskWarning).

    Příklad:
        >>> predict_risk({"amount": 100_000, "sba_share": 0.75, "naics": 44, "state": "CA"})
    """
    if not isinstance(margin, (int, float)) or not 0 < margin < 1:
        raise ValueError(f"„margin“ musí být mezi 0 a 1, zadáno {margin!r}.")
    X, notes = to_features(loan)
    model, info, shifts = load_artifacts()
    p = model.predict_proba(X)[:, 1]
    interval = apply_interval(p, shifts).iloc[0]
    threshold = float(break_even_risk(X["sba_share"].iloc[0], margin, info["loss_given_default"]))
    for note in notes:
        warnings.warn(note, RiskWarning, stacklevel=2)
    return {
        "probability": float(p[0]),
        "risk": float(interval["risk"]),
        "low": float(interval["low"]),
        "high": float(interval["high"]),
        "break_even": threshold,
        "decision": "schválit" if interval["risk"] < threshold else "zamítnout",
        "warnings": notes,
    }


def predict_many(loans: list[Mapping], margin: float = 0.10) -> pd.DataFrame:
    """`predict_risk` pro seznam úvěrů. Vrací tabulku, jeden řádek = jeden úvěr."""
    return pd.DataFrame([predict_risk(loan, margin) for loan in loans])

# sba-risk

Kalkulačka rizika odepsání úvěrů s ručením SBA (Small Business Administration).
Seminární práce 4EK481/4EK666 Modelling in Data Science, Part III: Production.

## Instalace

Potřebuješ [uv](https://docs.astral.sh/uv/) (`brew install uv`).

```bash
uv sync
```

Data: viz [data/README.md](data/README.md).

## Použití v Pythonu

```python
from sba_risk import predict_risk

predict_risk({
    "amount": 100_000,      # schválená částka v USD
    "sba_share": 0.75,      # podíl záruky SBA
    "naics": 44,            # kód odvětví (stačí 2 číslice)
    "state": "CA",          # stát firmy
    "bank_state": "CA",     # stát banky
    "employees": 5,
    "new_jobs": 0,
    "retained_jobs": 5,
    "new_business": False,
    "revolving": "N",
})
# {'probability': 0.202, 'risk': 0.300, 'low': 0.183, 'high': 0.472,
#  'break_even': 0.392, 'decision': 'schválit', 'warnings': []}
```

- `probability` – pravděpodobnost odepsání podle modelu,
- `risk`, `low`, `high` – posunutá pravděpodobnost a její 80% interval,
- `decision` – schválit / zamítnout podle očekávaného zisku banky (marže 10 %, lze změnit
  parametrem `margin`).

Popis všech vstupů: `from sba_risk import FIELDS`. Nesmyslný vstup vyvolá `ValueError`,
neobvyklý vstup varování `RiskWarning`.

## Znovu spočítat výsledky

```bash
uv run python -m sba_risk.data   # vyčistí CSV a uloží data/sba_clean.parquet
uv run python -m sba_risk.describe   # otázky 1A a 1B, grafy do reports/figures/
uv run python -m sba_risk.features   # rozdělení na train / valid / test
uv run python -m sba_risk.model      # naučí modely, uloží nejlepší do src/sba_risk/artifacts/
uv run python -m sba_risk.uncertainty   # 80% interval a kontrola pokrytí na testu
uv run python -m sba_risk.decision   # pravidla schvalování a zisk banky na 2004–06
```

## Testy

```bash
uv run pytest                    # všech 110 testů (~1 min, včetně instalace do nového prostředí)
uv run pytest -m "not slow"      # bez instalace (~5 s)
uv run pytest -m "not slow" --cov=sba_risk   # kolik kódu testy spouštějí (75 %)
```

Testy nepotřebují data z Kaggle (používají vymyšlené úvěry a uložený model).

| Požadavek (Part III) | Jak je ověřený | Soubor |
|---|---|---|
| 1. Stejné odpovědi jako model | 5 uložených úvěrů a jejich predikce (golden); `predict_risk` je vrací s přesností 10⁻¹², i po instalaci do nového prostředí | `test_api.py`, `test_model.py`, `test_production.py` |
| 2. Jen to, co je známo při predikci | uložený model bere přesně povolené proměnné; nic z `LEAKAGE` (odepsání, vyplacení, `Term`); vstupy `predict_risk` je ani nedovolí zadat | `test_features.py`, `test_api.py`, `test_production.py` |
| 3. Špatné vstupy | 17 nesmyslů → `ValueError`; chybějící povinný údaj → chyba; mimo rozsah, rok mimo data, chybějící nepovinný údaj → `RiskWarning` | `test_api.py` |
| 4. Reprodukovatelnost | pevný seed; zamčené verze (`uv.lock`, `scikit-learn==1.9.1`), jiná verze → varování; instalace z balíčku do čistého prostředí | `test_model.py`, `test_uncertainty.py`, `test_production.py` |
| 5. Poctivá nejistota | 80% intervaly pokrývají na testu (2004–06) 83 % skupin; interval jen z bootstrapu by pokryl 13 % | `test_uncertainty.py` |
| 6. Kód od AI je můj kód | každá funkce má testy na malých vymyšlených datech, ve kterých znám správný výsledek; pokrytí kódu 75 % (zbytek jsou skripty `main()` nad skutečnými daty) | `tests/` |

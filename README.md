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
uv run pytest                    # spustí testy
```

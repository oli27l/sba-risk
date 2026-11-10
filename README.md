# sba-risk

Kalkulačka rizika odepsání úvěrů s ručením SBA (Small Business Administration).
Seminární práce 4EK481/4EK666 Modelling in Data Science, Part III: Production.

## Instalace

Potřebuješ [uv](https://docs.astral.sh/uv/) (`brew install uv`).

```bash
uv sync
```

Data: viz [data/README.md](data/README.md).

## Použití

```bash
uv run python -m sba_risk.data   # vyčistí CSV a uloží data/sba_clean.parquet
uv run python -m sba_risk.describe   # otázky 1A a 1B, grafy do reports/figures/
uv run python -m sba_risk.features   # rozdělení na train / valid / test
uv run python -m sba_risk.model      # naučí modely, uloží nejlepší do src/sba_risk/artifacts/
uv run pytest                    # spustí testy
```

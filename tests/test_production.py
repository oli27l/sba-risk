"""Produkční testy podle bodů 1–6 zadání Part III.

Většina bodů je testovaná už v ostatních souborech (odkazy níže). Tady jsou testy,
které ověřují balíček jako celek – tak, jak ho dostane cizí uživatel.

| Bod zadání                   | Kde je otestovaný                                                  |
|------------------------------|--------------------------------------------------------------------|
| 1. stejné odpovědi jako model | test_api.py (golden), test_model.py, zde: čistá instalace          |
| 2. jen to, co je známo        | test_features.py, test_api.py, zde: vstupy uloženého modelu        |
| 3. špatné vstupy              | test_api.py (chyby a varování)                                     |
| 4. reprodukovatelnost         | test_model.py, test_uncertainty.py (seed), zde: verze, instalace   |
| 5. poctivá nejistota          | test_uncertainty.py (pokrytí na testu 70–90 %)                     |
| 6. kód od AI je můj kód       | všechny testy + pokrytí kódu testy (pytest --cov)                  |
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest
import sklearn

from sba_risk.api import version_note
from sba_risk.features import FEATURES, LEAKAGE
from sba_risk.model import GOLDEN_PATH, INFO_PATH, MODEL_PATH

ROOT = Path(__file__).parent.parent


def test_saved_model_takes_exactly_the_features():
    # Bod 2: uložený model (ne jen náš seznam) bere přesně FEATURES a nic z LEAKAGE.
    model = joblib.load(MODEL_PATH)
    assert list(model.feature_names_in_) == FEATURES
    assert not set(LEAKAGE) & set(model.feature_names_in_)


def test_installed_sklearn_is_the_one_model_was_saved_with():
    # Bod 4: zaznamenané verze. Model uložený se scikit-learn X načítáme se stejnou X.
    info = json.loads(INFO_PATH.read_text())
    assert sklearn.__version__ == info["sklearn_version"]
    assert version_note(info) is None


def test_different_sklearn_version_warns():
    note = version_note({"sklearn_version": "1.9.1"}, installed="2.0.0")
    assert "1.9.1" in note and "2.0.0" in note


def test_pyproject_pins_sklearn_version():
    info = json.loads(INFO_PATH.read_text())
    assert f'"scikit-learn=={info["sklearn_version"]}"' in (ROOT / "pyproject.toml").read_text()


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("uv") is None, reason="potřebuje uv")
def test_clean_install_gives_same_predictions(tmp_path):
    """Bod 1 a 4: postaví balíček, nainstaluje ho do úplně nového prostředí mimo projekt
    (jako cizí uživatel) a ověří, že vrací golden predikce."""
    run = lambda *cmd: subprocess.run(cmd, check=True, capture_output=True, text=True,  # noqa: E731
                                      cwd=tmp_path)
    run("uv", "build", "--wheel", "--out-dir", str(tmp_path / "dist"), str(ROOT))
    wheel = next((tmp_path / "dist").glob("sba_risk-*.whl"))
    run("uv", "venv", "--python", f"{sys.version_info.major}.{sys.version_info.minor}", "venv")
    python = tmp_path / "venv" / "bin" / "python"
    run("uv", "pip", "install", "--python", str(python), str(wheel))

    script = (
        "import json, sba_risk\n"
        "from sba_risk import predict_risk\n"
        "loan = {'amount': 100_000, 'sba_share': 0.75, 'naics': 44, 'state': 'CA', "
        "'bank_state': 'CA', 'employees': 5, 'new_jobs': 0, 'retained_jobs': 5, "
        "'new_business': False, 'revolving': 'N'}\n"
        "print(json.dumps({'file': sba_risk.__file__, 'p': predict_risk(loan)['probability']}))\n"
    )
    out = json.loads(run(str(python), "-c", script).stdout)
    assert str(ROOT) not in out["file"]  # opravdu nainstalovaná kopie, ne složka projektu
    assert out["p"] == pytest.approx(pd.read_csv(GOLDEN_PATH)["prediction"][0], abs=1e-12)

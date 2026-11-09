"""Načtení a vyčištění dat SBAnational.csv.

Jeden řádek = jeden úvěr. Výstupem funkce `clean` je tabulka s čitelnými názvy
sloupců, správnými datovými typy a cílovou proměnnou `default` (1 = odepsán).
"""

from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path("data/SBAnational.csv")
CLEAN_PATH = Path("data/sba_clean.parquet")

# Poslední rok, ze kterého data pocházejí. Datum po něm je chyba dvouciferného roku.
LAST_DATA_YEAR = 2014

# Názvy odvětví podle prvních dvou číslic kódu NAICS.
SECTORS = {
    "11": "Zemědělství, lesnictví, rybolov",
    "21": "Těžba",
    "22": "Energetika a vodárenství",
    "23": "Stavebnictví",
    "31": "Výroba",
    "32": "Výroba",
    "33": "Výroba",
    "42": "Velkoobchod",
    "44": "Maloobchod",
    "45": "Maloobchod",
    "48": "Doprava a skladování",
    "49": "Doprava a skladování",
    "51": "Informace a média",
    "52": "Finance a pojišťovnictví",
    "53": "Nemovitosti a pronájem",
    "54": "Odborné a technické služby",
    "55": "Řízení podniků",
    "56": "Administrativní a podpůrné služby",
    "61": "Vzdělávání",
    "62": "Zdravotní a sociální péče",
    "71": "Kultura, zábava a rekreace",
    "72": "Ubytování a stravování",
    "81": "Ostatní služby",
    "92": "Veřejná správa",
}
UNKNOWN_SECTOR = "Neuvedeno"


def parse_money(s: pd.Series) -> pd.Series:
    """Převede text typu '$60,000.00 ' na číslo 60000.0."""
    cleaned = s.str.replace(r"[$,\s]", "", regex=True)
    return pd.to_numeric(cleaned.replace("", np.nan))


def parse_date(s: pd.Series, max_year: pd.Series | int) -> pd.Series:
    """Převede text typu '28-Feb-97' na datum.

    Dvouciferný rok je nejednoznačný: pandas čte '61' jako 2061. Pokud vyjde rok
    vyšší než `max_year`, posuneme datum o 100 let zpět (2061 -> 1961).
    """
    dates = pd.to_datetime(s.replace("", np.nan), format="%d-%b-%y")
    too_late = dates.dt.year > max_year
    return dates.where(~too_late, dates - pd.DateOffset(years=100))


def yes_no(s: pd.Series) -> pd.Series:
    """'Y' -> 1, 'N' -> 0, cokoli jiného (neplatné kódy) -> chybějící hodnota."""
    return s.map({"Y": 1.0, "N": 0.0})


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Načte surové CSV. Vše jako text, abychom každý sloupec převedli vědomě."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Z surových dat udělá čistou tabulku. Úvěry bez známého výsledku vyřadí."""
    raw = raw[raw["MIS_Status"].isin(["P I F", "CHGOFF"])]

    df = pd.DataFrame(index=raw.index)
    df["loan_id"] = raw["LoanNr_ChkDgt"]
    df["bank"] = raw["Bank"].replace("", np.nan)
    df["bank_state"] = raw["BankState"].replace("", np.nan)
    df["state"] = raw["State"].replace("", np.nan)

    # Odvětví: první dvě číslice NAICS; kód 0 znamená, že odvětví chybí.
    naics2 = raw["NAICS"].str[:2]
    df["sector"] = naics2.map(SECTORS).fillna(UNKNOWN_SECTOR)

    # ApprovalFY obsahuje i hodnotu '1976A', proto bereme jen první 4 znaky.
    df["approval_fy"] = raw["ApprovalFY"].str[:4].astype(int)
    # Rok schválení je FY nebo FY-1 (fiskální rok začíná v říjnu), nikdy později.
    df["approval_date"] = parse_date(raw["ApprovalDate"], df["approval_fy"])
    disbursed = parse_date(raw["DisbursementDate"], LAST_DATA_YEAR)
    # Vyplacení víc než rok PŘED schválením je chyba v datech (10 úvěrů, např. 1920).
    impossible = disbursed < df["approval_date"] - pd.DateOffset(years=1)
    df["disbursement_date"] = disbursed.mask(impossible)

    # Splatnost 0 měsíců nedává smysl, bereme ji jako chybějící údaj (806 úvěrů).
    df["term_months"] = raw["Term"].astype(float).replace(0, np.nan)
    df["no_emp"] = raw["NoEmp"].astype(int)
    # Počet vytvořených a udržených pracovních míst podle žádosti (známo při schválení).
    df["create_job"] = raw["CreateJob"].astype(int)
    df["retained_job"] = raw["RetainedJob"].astype(int)
    # NewExist: 1 = existující firma, 2 = nová firma, 0 a prázdné = neznámo.
    df["new_business"] = raw["NewExist"].map({"1": 0.0, "2": 1.0})
    # FranchiseCode 0 a 1 znamenají „není franšíza“, ostatní kódy jsou franšízy.
    df["franchise"] = (~raw["FranchiseCode"].isin(["0", "1"])).astype(int)
    df["urban_rural"] = raw["UrbanRural"].map({"1": "urban", "2": "rural", "0": "unknown"})
    # RevLineCr: kromě Y/N i kódy '0' (31 % úvěrů) a 'T'. Nejsou to překlepy, jejich
    # četnost se mění v čase, proto je necháme jako vlastní kategorie. Zbytek -> chybí.
    df["rev_line_cr"] = raw["RevLineCr"].where(raw["RevLineCr"].isin(["Y", "N", "0", "T"]))
    df["low_doc"] = yes_no(raw["LowDoc"])

    df["gr_appv"] = parse_money(raw["GrAppv"])
    df["sba_appv"] = parse_money(raw["SBA_Appv"])
    df["sba_share"] = df["sba_appv"] / df["gr_appv"]
    # Úvěry na 20 a více let bývají kryté nemovitostí (Li et al., 2018).
    df["real_estate"] = (df["term_months"] >= 240).astype(int)

    # Splatnost: kdy měl úvěr doběhnout. Potřebujeme pro otázku 1B.
    start = df["disbursement_date"].fillna(df["approval_date"])
    df["maturity_date"] = start + pd.to_timedelta(df["term_months"] * 30.44, unit="D")

    # Cíl: 1 = odepsán (CHGOFF), 0 = splacen (P I F).
    df["default"] = (raw["MIS_Status"] == "CHGOFF").astype(int)

    # Údaje známé až PO schválení. Nesmí jít do modelu (únik informací),
    # ale hodí se pro rozhodovací otázku (kolik banka ztratila).
    df["chgoff_date"] = parse_date(raw["ChgOffDate"], LAST_DATA_YEAR)
    df["chgoff_prin_gr"] = parse_money(raw["ChgOffPrinGr"])
    df["disbursement_gross"] = parse_money(raw["DisbursementGross"])

    return df.reset_index(drop=True)


def main() -> None:
    """Načte CSV, vyčistí ho a uloží do rychlého formátu parquet."""
    df = clean(load_raw())
    df.to_parquet(CLEAN_PATH, index=False)
    print(f"Uloženo {len(df):,} úvěrů do {CLEAN_PATH}")


if __name__ == "__main__":
    main()

"""Testy čištění dat. Pracují na malých ručně vytvořených tabulkách, ne na CSV."""

import pandas as pd

from sba_risk.data import UNKNOWN_SECTOR, clean, parse_date, parse_money, yes_no


def raw_loan(**changes) -> dict:
    """Jeden surový úvěr tak, jak vypadá v CSV (vše jako text)."""
    row = {
        "LoanNr_ChkDgt": "1000014003",
        "Name": "ABC HOBBYCRAFT",
        "City": "EVANSVILLE",
        "State": "IN",
        "Zip": "47711",
        "Bank": "FIFTH THIRD BANK",
        "BankState": "OH",
        "NAICS": "451120",
        "ApprovalDate": "28-Feb-97",
        "ApprovalFY": "1997",
        "Term": "84",
        "NoEmp": "4",
        "NewExist": "2",
        "CreateJob": "0",
        "RetainedJob": "0",
        "FranchiseCode": "1",
        "UrbanRural": "0",
        "RevLineCr": "N",
        "LowDoc": "Y",
        "ChgOffDate": "",
        "DisbursementDate": "28-Feb-99",
        "DisbursementGross": "$60,000.00 ",
        "BalanceGross": "$0.00 ",
        "MIS_Status": "P I F",
        "ChgOffPrinGr": "$0.00 ",
        "GrAppv": "$60,000.00 ",
        "SBA_Appv": "$48,000.00 ",
    }
    row.update(changes)
    return row


def test_parse_money():
    s = pd.Series(["$60,000.00 ", "$1,234,567.89", "$0.00 ", ""])
    result = parse_money(s)
    assert result.tolist()[:3] == [60000.0, 1234567.89, 0.0]
    assert pd.isna(result.iloc[3])


def test_parse_date_fixes_two_digit_year():
    # '61' by pandas přečetl jako 2061; s horní hranicí 1962 musí vyjít 1961.
    s = pd.Series(["7-Dec-61", "28-Feb-97"])
    result = parse_date(s, max_year=pd.Series([1962, 1997]))
    assert result.dt.year.tolist() == [1961, 1997]


def test_yes_no_invalid_codes_become_missing():
    result = yes_no(pd.Series(["Y", "N", "0", "T", "`"]))
    assert result.iloc[0] == 1 and result.iloc[1] == 0
    assert result.iloc[2:].isna().all()


def test_clean_one_loan():
    df = clean(pd.DataFrame([raw_loan()]))
    loan = df.iloc[0]
    assert loan["gr_appv"] == 60000.0
    assert loan["sba_share"] == 0.8
    assert loan["sector"] == "Maloobchod"
    assert loan["new_business"] == 1.0
    assert loan["franchise"] == 0
    assert loan["urban_rural"] == "unknown"
    assert loan["approval_date"] == pd.Timestamp("1997-02-28")
    assert loan["default"] == 0


def test_clean_target_and_drops_unknown_outcome():
    raw = pd.DataFrame([
        raw_loan(MIS_Status="P I F"),
        raw_loan(MIS_Status="CHGOFF"),
        raw_loan(MIS_Status=""),
    ])
    df = clean(raw)
    assert len(df) == 2
    assert df["default"].tolist() == [0, 1]


def test_clean_missing_naics_is_own_group():
    df = clean(pd.DataFrame([raw_loan(NAICS="0")]))
    assert df.loc[0, "sector"] == UNKNOWN_SECTOR


def test_clean_fiscal_year_with_letter():
    df = clean(pd.DataFrame([raw_loan(ApprovalFY="1976A", ApprovalDate="15-Mar-76")]))
    assert df.loc[0, "approval_fy"] == 1976
    assert df.loc[0, "approval_date"].year == 1976


def test_clean_impossible_disbursement_becomes_missing():
    df = clean(pd.DataFrame([raw_loan(DisbursementDate="14-Apr-20")]))
    assert pd.isna(df.loc[0, "disbursement_date"])


def test_clean_term_zero_is_missing():
    df = clean(pd.DataFrame([raw_loan(Term="0")]))
    assert pd.isna(df.loc[0, "term_months"])
    assert pd.isna(df.loc[0, "maturity_date"])


def test_clean_rev_line_cr_keeps_known_codes():
    codes = ["Y", "N", "0", "T", "`"]
    df = clean(pd.DataFrame([raw_loan(RevLineCr=c) for c in codes]))
    assert df["rev_line_cr"].tolist()[:4] == ["Y", "N", "0", "T"]
    assert pd.isna(df.loc[4, "rev_line_cr"])


def test_clean_jobs():
    df = clean(pd.DataFrame([raw_loan(CreateJob="3", RetainedJob="7")]))
    assert df.loc[0, "create_job"] == 3
    assert df.loc[0, "retained_job"] == 7


def test_clean_real_estate_flag():
    raw = pd.DataFrame([raw_loan(Term="240"), raw_loan(Term="239")])
    assert clean(raw)["real_estate"].tolist() == [1, 0]

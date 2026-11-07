"""SEC EDGAR access and conservative annual XBRL extraction.

Requires a real identifying User-Agent (app name and contact email).
Values are returned with their source tags and filing dates for verification.
"""
from __future__ import annotations

import json
import re
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

import streamlit as st

BASE = "https://data.sec.gov"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

METRICS = {
    "Revenue": {"tags": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet", "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueGoodsNet"], "duration": True},
    "Net income": {"tags": ["NetIncomeLoss", "ProfitLoss"], "duration": True},
    "Operating income": {"tags": ["OperatingIncomeLoss"], "duration": True},
    "Total assets": {"tags": ["Assets"], "duration": False},
    "Total liabilities": {"tags": ["Liabilities"], "duration": False},
    "Stockholders equity": {"tags": ["StockholdersEquity"], "duration": False},
    "Operating cash flow": {"tags": ["NetCashProvidedByUsedInOperatingActivities"], "duration": True},
    "Capital expenditure": {"tags": ["PaymentsToAcquirePropertyPlantAndEquipment"], "duration": True},
    "Depreciation and amortization": {"tags": ["DepreciationDepletionAndAmortization"], "duration": True},
}


def valid_agent(agent: str) -> bool:
    return bool(re.search(r"[^\s@]+@[^\s@]+\.[^\s@]+", agent) and len(agent.strip()) >= 12)


@st.cache_data(ttl=3600, show_spinner=False)
def sec_json(url: str, agent: str) -> dict:
    if not valid_agent(agent):
        raise ValueError("Enter a genuine SEC User-Agent containing your app name and contact email.")
    req = Request(url, headers={"User-Agent": agent.strip(), "Accept": "application/json", "Accept-Encoding": "identity"})
    try:
        with urlopen(req, timeout=25) as resp:
            return json.load(resp)
    except HTTPError as e:
        raise RuntimeError(f"SEC returned HTTP {e.code}. If 403, verify your User-Agent and try again later. URL: {url}") from e
    except (URLError, TimeoutError) as e:
        raise RuntimeError(f"Unable to reach the SEC. Check internet access and try again: {e}") from e


def lookup_ticker(ticker: str, agent: str) -> dict:
    ticker = ticker.strip().upper()
    if not re.fullmatch(r"[A-Z0-9.\-]{1,12}", ticker):
        raise ValueError("Use a valid US ticker, e.g. AAPL, MSFT, or BLK.")
    mapping = sec_json(TICKERS_URL, agent)
    for value in mapping.values():
        if str(value.get("ticker", "")).upper() == ticker:
            return {"name": value["title"], "ticker": ticker, "cik": int(value["cik_str"])}
    raise ValueError(f"Ticker {ticker!r} not found in the SEC ticker mapping. Try AAPL, MSFT, or BLK.")


def get_company_data(cik: int, agent: str):
    cik10 = f"{cik:010d}"
    facts = sec_json(f"{BASE}/api/xbrl/companyfacts/CIK{cik10}.json", agent)
    submissions = sec_json(f"{BASE}/submissions/CIK{cik10}.json", agent)
    return facts, submissions


def _observations(facts: dict, tag: str, duration: bool) -> list[dict]:
    concept = facts.get("facts", {}).get("us-gaap", {}).get(tag, {})
    units = concept.get("units", {})
    values = units.get("USD", [])
    rows = []
    for x in values:
        if x.get("form") not in ("10-K", "10-K/A") or x.get("fp") != "FY":
            continue
        if not isinstance(x.get("val"), (int, float)) or not isinstance(x.get("fy"), int):
            continue
        if duration:
            # Fiscal full-year data; reject instant values and shorter spans.
            if not (x.get("start") and x.get("end")):
                continue
            try:
                from datetime import date
                length = (date.fromisoformat(x["end"]) - date.fromisoformat(x["start"])).days
            except (ValueError, TypeError):
                continue
            if not (330 <= length <= 380):
                continue
        else:
            if x.get("start"):
                continue
        rows.append({"value": x["val"], "year": x["fy"], "end": x.get("end", ""),
                     "filed": x.get("filed", ""), "accn": x.get("accn", ""), "tag": tag,
                     "form": x.get("form", "")})
    return rows


def annual_financials(facts: dict, max_years=5):
    """Choose one US-GAAP tag per metric; prefer annual data from most recent filings.

    FY matching: pick observations whose fiscal year matches their reported period end
    calendar year, or one year earlier (common for early FY ends). FY in SEC data can be
    filing fiscal year rather than the comparative period; prefer `frame=CY...` where
    present but still rely on explicit end labels for display.
    """
    all_data = {}
    for metric, spec in METRICS.items():
        selected_tag = None
        observations = []
        for tag in spec["tags"]:
            candidates = _observations(facts, tag, spec["duration"])
            # Prioritize recent annual coverage instead of longest historical coverage.
            # Some older XBRL concepts contain many years but stopped being used.
            def coverage(rows):
                years = {int(r["end"][:4]) for r in rows if r.get("end", "")[:4].isdigit()}
                latest = max(years, default=0)
                recent_count = sum(y >= latest - max_years + 1 for y in years)
                return (latest, recent_count, len(years))
            if coverage(candidates) > coverage(observations):
                selected_tag, observations = tag, candidates
        if not observations:
            continue
        # Use end year for comparable annual periods; for 52/53-week fiscal years this
        # keeps the company's own FY-end date as the precise point of reference.
        by_year = {}
        for r in observations:
            end_year = int(r["end"][:4]) if r["end"][:4].isdigit() else None
            if end_year is None:
                continue
            current = by_year.get(end_year)
            # Prefer a record from the filing nearest to that FY, then latest amended
            # filing as a tie breaker. This reduces comparative-year duplication.
            filing_gap = abs(r["year"] - end_year)
            key = (filing_gap, -int(r["filed"].replace("-", "") or 0))
            if current is None or key < current[0]:
                by_year[end_year] = (key, r)
        all_data[metric] = {year: value[1] for year, value in by_year.items()}
    years = sorted({y for data in all_data.values() for y in data}, reverse=True)[:max_years]
    records = []
    for year in years:
        record = {"Year": year}
        for metric in METRICS:
            obs = all_data.get(metric, {}).get(year)
            record[metric] = obs["value"] / 1e9 if obs else None
        records.append(record)
    return records, all_data


def latest_10k(submissions: dict):
    recent = submissions.get("filings", {}).get("recent", {})
    for i, form in enumerate(recent.get("form", [])):
        if form in ("10-K", "10-K/A"):
            return {"date": recent.get("filingDate", [])[i], "accession": recent.get("accessionNumber", [])[i], "form": form}
    return None

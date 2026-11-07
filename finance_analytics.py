"""Deterministic financial ratios and an explicitly simplified FCF DCF demonstration.

Input amounts are USD billions; missing observations are never treated as zero.
This is educational analytics, not a certified valuation or investment advice.
"""
from __future__ import annotations

import math
import pandas as pd


def safe_div(num, denom):
    if num is None or denom is None or pd.isna(num) or pd.isna(denom) or denom == 0:
        return None
    return float(num) / float(denom)


def compute_financial_ratios(records: list[dict]) -> pd.DataFrame:
    """Compute common accounting ratios, preserving unavailable fields as NA."""
    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame()
    df = df.sort_values('Year').copy()
    metrics = []
    previous = None
    for row in df.to_dict('records'):
        rev = row.get('Revenue')
        net = row.get('Net income')
        op = row.get('Operating income')
        assets = row.get('Total assets')
        equity = row.get('Stockholders equity')
        ocf = row.get('Operating cash flow')
        capex = row.get('Capital expenditure')
        prev_rev = previous.get('Revenue') if previous else None
        prev_assets = previous.get('Total assets') if previous else None
        prev_equity = previous.get('Stockholders equity') if previous else None
        avg_assets = (float(assets) + float(prev_assets)) / 2 if assets is not None and prev_assets is not None and pd.notna(assets) and pd.notna(prev_assets) else None
        avg_equity = (float(equity) + float(prev_equity)) / 2 if equity is not None and prev_equity is not None and pd.notna(equity) and pd.notna(prev_equity) else None
        growth_base = safe_div(rev, prev_rev)
        metrics.append({
            'Year': int(row['Year']),
            'Revenue ($B)': rev,
            'Revenue growth (%)': 100 * (growth_base - 1) if growth_base is not None else None,
            'Operating margin (%)': 100 * v if (v := safe_div(op, rev)) is not None else None,
            'Net profit margin (%)': 100 * v if (v := safe_div(net, rev)) is not None else None,
            'ROA, average assets (%)': 100 * v if (v := safe_div(net, avg_assets)) is not None else None,
            'ROE, average equity (%)': 100 * v if (v := safe_div(net, avg_equity)) is not None else None,
            'Cash flow less capex ($B)': float(ocf) - float(capex)
                if ocf is not None and capex is not None and not pd.isna(ocf) and not pd.isna(capex)
                else None,
        })
        previous = row
    return pd.DataFrame(metrics).sort_values('Year', ascending=False).reset_index(drop=True)


def simple_cashflow_dcf(base_cashflow: float, discount_rate: float, growth_rate: float,
                        terminal_growth_rate: float, years: int = 5) -> dict:
    """Illustrative perpetuity DCF of an explicitly supplied annual cash-flow proxy.

    It discounts an operating-cash-flow-minus-capex proxy, not a fully adjusted
    unlevered FCFF or FCFE. The result is a PV of forecast cash flows + terminal
    value, NOT an appraised enterprise/equity value or a share-price target.
    """
    vals = (base_cashflow, discount_rate, growth_rate, terminal_growth_rate)
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals):
        raise ValueError('Inputs must be finite numbers.')
    if base_cashflow <= 0:
        raise ValueError('Base cash flow must be positive for this simplified DCF.')
    if years < 1 or years > 15:
        raise ValueError('Forecast years must be between 1 and 15.')
    if discount_rate <= terminal_growth_rate:
        raise ValueError('Discount rate must be greater than terminal growth.')
    if discount_rate <= -1 or growth_rate <= -1:
        raise ValueError('Growth and discount rates must be greater than -100%.')
    flows = []
    for t in range(1, years + 1):
        flow = base_cashflow * (1 + growth_rate) ** t
        pv = flow / (1 + discount_rate) ** t
        flows.append({'Forecast year': t, 'Cash flow ($B)': flow, 'Present value ($B)': pv})
    terminal = flows[-1]['Cash flow ($B)'] * (1 + terminal_growth_rate) / (discount_rate - terminal_growth_rate)
    terminal_pv = terminal / (1 + discount_rate) ** years
    explicit_pv = sum(x['Present value ($B)'] for x in flows)
    return {
        'forecast': flows,
        'pv_forecast': explicit_pv,
        'terminal_value': terminal,
        'pv_terminal': terminal_pv,
        'illustrative_pv': explicit_pv + terminal_pv,
        'terminal_pct': terminal_pv / (explicit_pv + terminal_pv) * 100,
    }


def financial_prompt_context(company: dict, records: list[dict], ratios: pd.DataFrame) -> str:
    """Bounded structured evidence for Ollama to explain (not invent) numbers."""
    data = ratios.head(5).where(pd.notna(ratios.head(5)), None).to_dict('records') if not ratios.empty else []
    def compact(value):
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return 'N/A'
        if isinstance(value, float):
            return f'{value:.4f}'
        return str(value)
    lines = [f"Company: {company['name']} ({company['ticker']})", 'Source: SEC EDGAR 10-K Company Facts (USD billions unless labelled %)']
    for row in records[:5]:
        lines.append(f"FY{row['Year']}: " + '; '.join(f'{key}={compact(value)}' for key, value in row.items() if key != 'Year'))
    for row in data:
        lines.append(f"Calculated FY{row['Year']}: " + '; '.join(f'{key}={compact(value)}' for key, value in row.items() if key != 'Year'))
    return '\n'.join(lines)

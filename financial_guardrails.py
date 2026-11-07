"""Verified SEC question answering; arithmetic stays in Python, never an LLM.

Scope: common annual metrics and two-year comparisons. Unknown questions return
None for qualitative LLM explanation; known-but-unavailable metrics are blocked.
"""
from __future__ import annotations

import math
import re

METRIC_TERMS = [
    ("EBITDA", r"\bebitda\b"),
    ("Revenue growth (%)", r"\b(?:revenue|sales)\s+(?:growth|change)\b|\bgrowth\s+(?:in|of)\s+(?:revenue|sales)\b"),
    ("Operating margin (%)", r"\boperating\s+(?:profit\s+)?margin\b"),
    ("Net profit margin (%)", r"\b(?:net\s+)?profit\s+margin\b|\bnet\s+margin\b"),
    ("ROE, average equity (%)", r"\b(?:roe|return\s+on\s+equity)\b"),
    ("ROA, average assets (%)", r"\b(?:roa|return\s+on\s+assets)\b"),
    ("Cash flow less capex ($B)", r"\b(?:free\s+cash\s+flow|fcf|cash\s+flow\s+less\s+capex)\b"),
    ("Operating income", r"\boperating\s+(?:income|profit)\b"),
    ("Net income", r"\bnet\s+(?:income|earnings|profit)\b"),
    ("Revenue", r"\b(?:revenue|net\s+sales|sales)\b"),
]

DISPLAY = {
    "EBITDA": ("EBITDA ($B)", "USD billions"),
    "Revenue growth (%)": ("Revenue growth (%)", "%"),
    "Operating margin (%)": ("Operating margin (%)", "%"),
    "Net profit margin (%)": ("Net profit margin (%)", "%"),
    "ROE, average equity (%)": ("ROE, average equity (%)", "%"),
    "ROA, average assets (%)": ("ROA, average assets (%)", "%"),
    "Cash flow less capex ($B)": ("Operating cash flow less capex ($B)", "USD billions"),
    "Operating income": ("Operating income ($B)", "USD billions"),
    "Net income": ("Net income ($B)", "USD billions"),
    "Revenue": ("Revenue ($B)", "USD billions"),
}


def available(x):
    try:
        return x is not None and math.isfinite(float(x))
    except (ValueError, TypeError):
        return False


def determine_metric(question):
    q = question.lower()
    for label, pattern in METRIC_TERMS:
        if re.search(pattern, q, re.IGNORECASE):
            return label
    return None


def _years(question, available_years):
    asked = [int(x) for x in re.findall(r"\b(?:20\d{2})\b", question)]
    asked = sorted(set(y for y in asked if y in available_years), reverse=True)
    if len(asked) >= 2:
        return asked[:2]
    if len(asked) == 1:
        others = [y for y in sorted(available_years, reverse=True) if y != asked[0]]
        return [asked[0]] + others[:1]
    return sorted(available_years, reverse=True)[:2]


def _format(v, pct):
    return f"{v:.2f}{'%' if pct else 'B'}"


def verified_answer(question: str, records: list[dict], ratios) -> str | None:
    """Answers supported quantitative questions deterministically.

    Returns None only for questions unrelated to supported quantitative metrics,
    so the caller may ask the LLM for a *qualitative* interpretation.
    """
    if question.startswith("Please name the financial metric"):
        return question
    metric = determine_metric(question)
    q = question.lower()
    if metric is None:
        # Catch common missing metrics rather than sending them to an LLM for arithmetic.
        if re.search(r"\b(?:pe\s+ratio|p/e|eps|ebit|interest\s+coverage|gross\s+margin)\b", q):
            return "**Not available.** This metric has not been validated or implemented in the SEC analytics layer. I won't estimate it from unrelated figures."
        return None

    by_year = {int(r['Year']): r for r in records}
    calculated = {int(r['Year']): r for r in ratios.to_dict('records')} if not ratios.empty else {}
    years = _years(question, set(by_year))
    if not years:
        return "No annual SEC financial figures are available for this company."

    values = {}
    missing = []
    for year in years:
        raw = by_year[year]
        derived = calculated.get(year, {})
        if metric == "EBITDA":
            op = raw.get('Operating income')
            da = raw.get('Depreciation and amortization')
            if available(op) and available(da):
                values[year] = float(op) + float(da)
            else:
                absent = [name for name, value in [('operating income', op), ('depreciation and amortization', da)] if not available(value)]
                missing.append(f"FY{year}: {', '.join(absent)}")
        elif metric in derived:
            v = derived[metric]
            if available(v):
                values[year] = float(v)
            else:
                missing.append(f"FY{year}: {metric}")
        elif metric in raw:
            v = raw[metric]
            if available(v):
                values[year] = float(v)
            else:
                missing.append(f"FY{year}: {metric}")
        else:
            missing.append(f"FY{year}: {metric}")

    name, unit = DISPLAY[metric]
    lines = [f"### {name}", "Source: SEC Company Facts, annual 10-K observations. Figures are computed or selected using Python."]
    if metric == 'EBITDA':
        lines.append("**Definition used:** EBITDA = operating income + depreciation and amortization (D&A). This is an analytical approximation; verify D&A scope and accounting adjustments against the filing.")
    if metric == 'Cash flow less capex ($B)':
        lines.append("**Definition:** Operating cash flow − capital expenditure. This is a cash-flow proxy; it is not necessarily adjusted FCFF or FCFE.")
    if missing:
        lines.extend(["**Insufficient verified data for a full comparison.** Missing: " + '; '.join(missing) + ".", "Missing amounts are **not** assumed to be zero."])
    if not values:
        return '\n\n'.join(lines)
    pct = unit == '%'
    for year, v in values.items():
        display = f"{v:.2f}%" if pct else f"${v:.2f}B"
        lines.append(f"- **FY{year}: {display}** [SEC Company Facts, FY{year}]")
    if len(values) == 2 and not missing:
        later, earlier = sorted(values, reverse=True)
        delta = values[later] - values[earlier]
        if pct:
            lines.append(f"**Change FY{earlier} → FY{later}: {delta:+.2f} percentage points.** (Percentage-point change, not percent growth.)")
        else:
            lines.append(f"**Change FY{earlier} → FY{later}: {delta:+.2f} USD billions.**")
            if values[earlier] != 0:
                lines.append(f"**Relative change:** {delta / abs(values[earlier]) * 100:+.2f}%.")
    return '\n\n'.join(lines)


FOLLOWUP_PATTERN = re.compile(
    r"\b(?:how (?:did|do|was|is|you) (?:you )?(?:calculate|calculated|compute|computed|derive|derived)|"
    r"show (?:me )?(?:the )?(?:formula|calculation|working)|explain (?:that|the calculation)|"
    r"what(?:'s| is) (?:the )?formula|why (?:did|has) (?:it|that) (?:change|increase|decrease))\b",
    re.IGNORECASE,
)


def resolve_followup_question(question: str, history: list[dict]) -> str:
    """Rehydrate shorthand reference from most recent relevant user question.

    Only a supported metric can be inherited. Inheriting years helps with
    'show me the formula', without sending that question to the LLM.
    """
    if determine_metric(question) or not FOLLOWUP_PATTERN.search(question):
        return question
    for msg in reversed(history):
        if msg.get('role') != 'user':
            continue
        previous = msg.get('content', '')
        metric = determine_metric(previous)
        if metric:
            years = re.findall(r"\b20\d{2}\b", previous)
            period = ' from ' + ' to '.join(years[:2]) if len(years) >= 2 else ''
            return f"Explain how to calculate {metric}{period}"
    return "Please name the financial metric you want explained (for example, operating margin or revenue growth)."


def explain_previous_calculation(question: str, records: list[dict], ratios) -> str | None:
    """Human-readable deterministic formula for explicit how-calculated queries."""
    metric = determine_metric(question)
    if metric is None or not re.search(r"\b(?:calculate|calculated|compute|formula|derive|explain how)\b", question.lower()):
        return None
    by_year = {int(r['Year']): r for r in records}
    years = _years(question, set(by_year))
    if not years:
        return 'No SEC data is available.'
    formulas = {
        'Operating margin (%)': ('Operating income / Revenue × 100', 'Operating income', 'Revenue'),
        'Net profit margin (%)': ('Net income / Revenue × 100', 'Net income', 'Revenue'),
        'ROA, average assets (%)': ('Net income / average total assets × 100', 'Net income', None),
        'ROE, average equity (%)': ('Net income / average shareholders equity × 100', 'Net income', None),
    }
    if metric == 'Revenue growth (%)':
        later, earlier = sorted(years, reverse=True)[:2]
        a = by_year[later].get('Revenue')
        b = by_year[earlier].get('Revenue')
        if not (available(a) and available(b) and b != 0):
            return '**Not available:** Revenue for both years is required.'
        v = (float(a) / float(b) - 1)*100
        return (f"### Revenue growth — calculation\n\n"
                f"**Formula:** (Current-year revenue − Previous-year revenue) / Previous-year revenue × 100\n\n"
                f"FY{later}: ${float(a):.3f}B; FY{earlier}: ${float(b):.3f}B.\n\n"
                f"**Calculation:** ({float(a):.3f} − {float(b):.3f}) / {float(b):.3f} × 100 = **{v:.2f}%**.\n\n"
                f"Source: SEC Company Facts, FY{earlier} and FY{later}.")
    if metric == 'EBITDA':
        return verified_answer(question, records, ratios)
    if metric not in formulas:
        return verified_answer(question, records, ratios)
    label, num_col, den_col = formulas[metric]
    lines = [f"### How {metric} was calculated", f"**Formula:** {label}."]
    for year in years:
        r=by_year[year]
        num=r.get(num_col)
        if den_col:
            den=r.get(den_col)
        else:
            previous=by_year.get(year-1, {})
            balance_key='Total assets' if 'ROA' in metric else 'Stockholders equity'
            first=r.get(balance_key)
            older=previous.get(balance_key)
            den=(float(first)+float(older))/2 if available(first) and available(older) else None
        if not available(num) or not available(den) or float(den)==0:
            lines.append(f"FY{year}: Not available; required numerator or denominator is missing.")
        else:
            value=float(num)/float(den)*100
            lines.append(f"- FY{year}: ${float(num):.3f}B / ${float(den):.3f}B × 100 = **{value:.2f}%**.")
    lines.append('Source: SEC Company Facts. Figures and calculations are processed by Python.')
    return '\n\n'.join(lines)


INTERPRET_PATTERN = re.compile(
    r"\b(?:good|bad|positive|negative|improvement|worse|better|interpret|meaning|mean|significant)\b",
    re.IGNORECASE,
)


def interpret_previous_metric(question: str, history: list[dict], records: list[dict], ratios) -> str | None:
    """Ground subjective one-line follow-ups in the last supported metric.

    Deliberately conservative. This describes direction, not investment merit.
    """
    if not INTERPRET_PATTERN.search(question):
        return None
    metric = determine_metric(question)
    if metric is None:
        for msg in reversed(history):
            if msg.get('role') == 'user':
                metric = determine_metric(msg.get('content', ''))
                if metric is not None:
                    break
    if metric is None:
        return 'Which financial metric would you like me to interpret? For example, ROA or operating margin.'
    calculated = {int(r['Year']): r for r in ratios.to_dict('records')} if not ratios.empty else {}
    raw = {int(r['Year']): r for r in records}
    if metric == 'EBITDA':
        return '**EBITDA interpretation unavailable:** Verify the operating-income and depreciation/amortisation components first.'
    source = calculated if any(metric in row for row in calculated.values()) else raw
    points = [(year, row.get(metric)) for year, row in sorted(source.items(), reverse=True)]
    points = [(year, float(v)) for year, v in points if available(v)]
    if len(points) < 2:
        return f'**Not enough verified data** to assess how {metric} changed over time.'
    (latest_year, latest), (prior_year, prior) = points[:2]
    difference = latest - prior
    unit = 'percentage points' if '(%)' in metric else 'USD billions'
    fmt = lambda value: f'{value:.2f}%' if '(%)' in metric else f'${value:.2f}B'
    direction = 'increased' if difference > 0 else 'decreased' if difference < 0 else 'was unchanged'
    caveats = {
        'ROA, average assets (%)': 'A higher ROA generally indicates improved net income generation relative to average assets; asset changes and one-off items can also affect it.',
        'ROE, average equity (%)': 'A higher ROE may indicate better equity profitability, but leverage, buybacks and equity changes can also increase it.',
        'Operating margin (%)': 'A higher operating margin generally suggests stronger operating profitability relative to revenue.',
        'Net profit margin (%)': 'A higher net margin generally suggests improved after-tax profitability relative to revenue.',
        'Revenue growth (%)': 'Higher revenue growth indicates faster reported sales expansion, but not necessarily stronger profitability.',
    }
    explanation = caveats.get(metric, 'The direction of this metric alone does not determine business quality or investment attractiveness.')
    if difference == 0:
        assessment = 'The metric was broadly unchanged.'
    else:
        assessment = ('This is generally favourable for this profitability measure.' if difference > 0 and metric in caveats and metric != 'Revenue growth (%)' else
                      'This is generally less favourable for this profitability measure.' if difference < 0 and metric in caveats and metric != 'Revenue growth (%)' else
                      'This change requires interpretation alongside other financial results.')
    return (f'### Interpreting {metric}\n\n'
            f'{metric} {direction} from **{fmt(prior)} (FY{prior_year})** to **{fmt(latest)} (FY{latest_year})**, '
            f'a change of **{difference:+.2f} {unit}**.\n\n'
            f'{assessment} {explanation}\n\n'
            '**This is not an investment recommendation.** Source: SEC Company Facts; calculations performed in Python.')

#!/usr/bin/env python3
"""
format.py - Deterministic phrase formatters for SSOT markers.
Every sentence or phrase containing a protected fact lives here.
"""

from python.ssot.load import SSOTData


def money(amount):
    """Format a numeric USD amount like 1000 -> '$1,000'."""
    val = float(amount)
    if val.is_integer():
        return f"${int(val):,}"
    return f"${val:,.2f}"


def tier_price_list(data: SSOTData, offering_id: str) -> str:
    """Return '$1,000 / $750 / $500 / $500 / $250 / Custom' computed from base x tier multipliers."""
    off = data.offerings[offering_id]
    base = float(off["tier1_base_price_usd"])
    parts = []
    for t in data.tiers_list:
        if t["tier"] == 6:
            parts.append("Custom")
        else:
            parts.append(money(base * float(t["multiplier"])))
    return " / ".join(parts)


def seats_phrase(data: SSOTData, offering_id: str) -> str:
    """Return 'up to 10 participants (+$100 per extra seat, max 20)'."""
    cap = data.offerings[offering_id]["cohort_capacity"]
    inc = cap["included_seats"]
    fee = money(cap["additional_seat_fee_usd"])
    mx = cap["max_seats"]
    return f"up to {inc} participants (+{fee} per extra seat, max {mx})"


def duration_phrase(data: SSOTData, offering_id: str) -> str:
    """Return canonical duration meaning from offering schedule."""
    return data.offerings[offering_id]["schedule"]["duration_meaning"]


def installment_phrase(data: SSOTData, offering_id: str) -> str:
    """Return '10 equal installments of 10%' computed from milestone_schedule."""
    ms = data.offerings[offering_id]["payment_rules"]["milestone_schedule"]
    count = len(ms)
    pct = ms[0]["percentage"] if ms else 0
    return f"{count} equal installments of {pct}%"


def reply_sla_phrase(data: SSOTData) -> str:
    """Return canonical contact reply SLA."""
    return data.raw["contact"]["direct_communication"]["reply_sla"]


def tier_price(data: SSOTData, spec: str) -> str:
    """Return single-tier price for 'offering_id:tier_num', e.g. '$1,000' or 'Custom'."""
    offering_id, tier_str = [p.strip() for p in spec.split(":", 1)]
    off = data.offerings[offering_id]
    tier = data.tiers[tier_str]
    if int(tier["tier"]) == 6:
        return "Custom"
    return money(float(off["tier1_base_price_usd"]) * float(tier["multiplier"]))


def extra_seat_price_list(data: SSOTData, offering_id: str) -> str:
    """Return '+$100 / +$75 / +$50 / +$50 / +$25 / Custom' for extra seats across tiers."""
    fee = float(data.offerings[offering_id]["cohort_capacity"]["additional_seat_fee_usd"])
    parts = []
    for t in data.tiers_list:
        if t["tier"] == 6:
            parts.append("Custom")
        else:
            parts.append(f"+{money(fee * float(t['multiplier']))}")
    return " / ".join(parts)


def extra_seat_tier_price(data: SSOTData, spec: str) -> str:
    """Return single-tier extra-seat fee for 'offering_id:tier_num', e.g. '$75' or 'Custom'."""
    offering_id, tier_str = [p.strip() for p in spec.split(":", 1)]
    fee = float(data.offerings[offering_id]["cohort_capacity"]["additional_seat_fee_usd"])
    tier = data.tiers[tier_str]
    if int(tier["tier"]) == 6:
        return "Custom"
    return money(fee * float(tier["multiplier"]))


def installment_amount_list(data: SSOTData, offering_id: str) -> str:
    """Return per-installment parity list, e.g. '$10 / $7.50 / $5 / $5 / $2.50 / Custom'."""
    off = data.offerings[offering_id]
    base = float(off["tier1_base_price_usd"])
    ms = off["payment_rules"]["milestone_schedule"]
    pct = (float(ms[0]["percentage"]) / 100.0) if ms else 0.10
    parts = []
    for t in data.tiers_list:
        if t["tier"] == 6:
            parts.append("Custom")
        else:
            parts.append(money(base * pct * float(t["multiplier"])))
    return " / ".join(parts)


def tier_precedence_phrase(data: SSOTData) -> str:
    """Return canonical tier precedence rule."""
    return data.raw["tiers"]["precedence_rule"]


def quarterly_cap_phrase(data: SSOTData) -> str:
    """Return canonical quarterly flagship cohort cap."""
    cap = data.raw["pricing"]["capacity"]["flagship_cohorts_per_quarter"]
    return f"{cap} flagship cohorts per quarter"


FORMATTERS = {
    "money": lambda data, arg: money(float(arg)),
    "tier_price_list": tier_price_list,
    "tier_price": tier_price,
    "extra_seat_price_list": extra_seat_price_list,
    "extra_seat_tier_price": extra_seat_tier_price,
    "installment_amount_list": installment_amount_list,
    "seats_phrase": seats_phrase,
    "duration_phrase": duration_phrase,
    "installment_phrase": installment_phrase,
    "reply_sla_phrase": lambda data: reply_sla_phrase(data),
    "tier_precedence_phrase": lambda data: tier_precedence_phrase(data),
    "quarterly_cap_phrase": lambda data: quarterly_cap_phrase(data),
}


def call_formatter(data: SSOTData, expr: str) -> str:
    """Evaluate 'fmt.func_name(arg)' or 'fmt.func_name()'."""
    if "(" not in expr or not expr.endswith(")"):
        raise ValueError(f"Invalid formatter expression: {expr!r}")
    name, rest = expr.split("(", 1)
    arg = rest[:-1].strip().strip("'\"")
    if name not in FORMATTERS:
        raise KeyError(f"Unknown formatter: {name!r}")
    fn = FORMATTERS[name]
    if arg:
        return str(fn(data, arg))
    return str(fn(data))

"""Displayed video metadata parsing and request filters (no external API)."""
import re
from decimal import Decimal, InvalidOperation

COUNT = r"(?P<number>(?:[0-9][0-9,.]*\s*)?(?:천|백|십)|[0-9][0-9,.]*)\s*(?P<unit>억|만|천|[KMB])?"
FACTORS = {"": 1, "천": 1000, "만": 10000, "억": 100000000, "k": 1000, "m": 1000000, "b": 1000000000}


def count_value(number, unit):
    try:
        number = number.replace(",", "").replace(" ", "")
        scale = {"천": 1000, "백": 100, "십": 10}.get(number[-1:], 1)
        if scale != 1:
            number = number[:-1] or "1"
        return int(Decimal(number) * scale * FACTORS[(unit or "").lower()])
    except (InvalidOperation, KeyError):
        return None


def displayed_views(text):
    match = re.search(r"조회수\s*" + COUNT, text, re.IGNORECASE)
    if not match:
        match = re.search(COUNT + r"\s*views\b", text, re.IGNORECASE)
    if not match:
        match = re.fullmatch(r"\s*" + COUNT + r"\s*(?:회)?\s*", text, re.IGNORECASE)
    return None if not match else {"value": count_value(match["number"], match["unit"]), "text": match[0].strip()}


def view_requirement(command):
    match = re.search(r"조회수(?:가|는)?\s*" + COUNT + r"\s*(?:회)?\s*(?P<operator>이상|초과|이하|미만|넘는|넘은|넘게|넘어|넘으면|넘고|보다\s*많은|보다\s*적은)", command, re.IGNORECASE)
    if not match:
        return None
    operator = match["operator"]
    if operator.startswith("넘") or re.fullmatch(r"보다\s*많은", operator):
        operator = "초과"
    elif re.fullmatch(r"보다\s*적은", operator):
        operator = "미만"
    return {"value": count_value(match["number"], match["unit"]), "operator": operator}


def meets_views(value, requirement):
    if not requirement:
        return True
    if value is None:
        return False
    limit = requirement["value"]
    return {"이상": value >= limit, "초과": value > limit, "이하": value <= limit, "미만": value < limit}[requirement["operator"]]


def card_title(label):
    separator = re.search(r"\s+-\s+(?:\d+\s*분(?:\s*\d+\s*초)?|\d+:\d+(?::\d+)?)\s+-", label)
    return label[:separator.start()].strip() if separator else label.split(" - ", 1)[0].strip()

import json
import re
from typing import Optional
from pricer.data.schema import Item
from pricer.config import (
    MIN_CHARS,
    MIN_PRICE,
    MAX_PRICE,
    MAX_TEXT_EACH,
    MAX_TEXT_TOTAL,
    REMOVAL_KEYWORDS,
)


def simplify(text_list) -> str:
    """Return a clean string without excess whitespace, truncated to MAX_TEXT_EACH."""
    return re.sub(r"\s+", " ", str(text_list)).strip()[:MAX_TEXT_EACH]


def scrub(title, description, features, details) -> str:
    """Scrub non-informative tokens and filter non-informational rows."""
    text = simplify(f"{title} {description} {features} {details}")
    for removal in REMOVAL_KEYWORDS:
        text = text.replace(removal, "")
    return text[:MAX_TEXT_TOTAL]


def extract_weight(details_str: str) -> Optional[float]:
    """Extract weight in ounces if present in details."""
    if not details_str:
        return 0.0
    match = re.search(r"(\d+(\.\d+)?)\s*(ounces|oz|pounds|lbs|g|kg)", details_str, re.IGNORECASE)
    if not match:
        return 0.0
    val = float(match.group(1))
    unit = match.group(3).lower()
    if unit in ("pounds", "lbs"):
        return val * 16.0
    elif unit in ("g",):
        return val * 0.035274
    elif unit in ("kg",):
        return val * 35.274
    return val


def parse(datapoint: dict, category: str) -> Optional[Item]:
    """Parse raw Amazon dataset datapoint into a clean Item object."""
    try:
        price_val = datapoint.get("price")
        if price_val is None or str(price_val).strip() == "":
            return None
        price = float(str(price_val).replace("$", "").replace(",", "").strip())
        if price < MIN_PRICE or price > MAX_PRICE:
            return None

        title = simplify(datapoint.get("title", ""))
        features = [simplify(f) for f in datapoint.get("features", [])]
        description = [simplify(d) for d in datapoint.get("description", [])]

        raw_details = datapoint.get("details", {})
        if isinstance(raw_details, str):
            try:
                raw_details = json.loads(raw_details)
            except Exception:
                raw_details = {}
        details = {simplify(k): simplify(v) for k, v in raw_details.items()}

        scrubbed = scrub(title, description, features, details)
        if len(scrubbed) < MIN_CHARS:
            return None

        weight = extract_weight(str(details))

        return Item(
            title=title,
            price=price,
            category=category,
            features=features,
            description=description,
            details=details,
            weight=weight,
        )
    except Exception:
        return None

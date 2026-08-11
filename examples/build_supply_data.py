"""Turn the raw Olist tables into one CSV per entity type declared in supply-v2.

Source: the Olist Brazilian e-commerce dataset, published by Olist itself
(github.com/olist/work-at-olist-data). Real marketplace records, anonymised by the publisher.

The output is the shape the platform reads: ``<out>/<EntityType>.csv``, one file per entity, keyed
so the relations in Ω can be walked by their declared ``via`` fields. Nothing is joined into a wide
table here on purpose — a pre-joined table would answer the multi-hop questions before the
ontology got a chance to, which is precisely what we want to demonstrate it doing.

Two derived columns are computed rather than read, and both are stated in the ontology's
descriptions:

  Product.volume_cm3   length x height x width
  Seller.on_time_rate  share of that seller's delivered lines that arrived by the promised date

A caveat worth carrying into any reading of the result: ``days_late`` is measured against the
marketplace's own estimated delivery date, so a seller can look punctual because it was given a
generous promise. This is a measurement-validity limit, not a data-quality one.

The source tables are not kept in the repository (about 44MB, and freely downloadable);
``--download`` fetches them first.

Usage:
    python examples/build_supply_data.py --download
    LOKA_SUPPLY_DATA=examples/supply_data uvicorn loka_api.app:app
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from collections import defaultdict
from datetime import datetime
from typing import Any

_SOURCE = "https://raw.githubusercontent.com/olist/work-at-olist-data/master/datasets"
_TABLES = (
    "olist_customers_dataset",
    "olist_order_items_dataset",
    "olist_orders_dataset",
    "olist_products_dataset",
    "olist_sellers_dataset",
)


def download(raw: str) -> None:
    """Fetch the source tables from the publisher's own repository."""
    import urllib.request

    os.makedirs(raw, exist_ok=True)
    for name in _TABLES:
        target = os.path.join(raw, f"{name}.csv")
        if os.path.exists(target):
            print(f"  {name}.csv already present")
            continue
        print(f"  fetching {name}.csv ...")
        urllib.request.urlretrieve(f"{_SOURCE}/{name}.csv", target)


def bulky_threshold(ontology_path: str) -> float:
    """The weight above which a product is a BulkyProduct — read from the ShipStandard guard.

    Taking it from Ω rather than restating it here is what keeps the subtype boundary and the
    action's eligibility rule the same rule: change the guard and the data reclassifies with it.

    The platform's loader is used when it is importable, since it validates the ontology on the
    way past. This script also has to run as a deployment step, before anything is installed, so
    it falls back to reading the guard out of the file directly — the value still comes from Ω
    either way, which is the part that matters.
    """
    with open(ontology_path) as f:
        text = f.read()
    try:
        from loka_ontology import load_ontology_str

        action = next(a for a in load_ontology_str(text).actions if a.name == "ShipStandard")
        guard = action.guard
    except ImportError:
        m = re.search(r"guard:\s*[\"']weight_g\s*<=\s*(-?\d+(?:\.\d+)?)[\"']", text)
        if m is None:
            raise SystemExit(
                f"could not read the ShipStandard weight guard from {ontology_path}"
            ) from None
        return float(m.group(1))
    return float(guard.rsplit("<=", 1)[1])


def _num(value: str | None) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _read(path: str) -> list[dict[str, str]]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _days_between(later: str, earlier: str) -> float | None:
    fmt = "%Y-%m-%d %H:%M:%S"
    try:
        delta = datetime.strptime(later, fmt) - datetime.strptime(earlier, fmt)
        return delta.total_seconds() / 86400
    except (TypeError, ValueError):
        return None


def build(raw: str, out: str, *, ontology: str) -> dict[str, int]:
    threshold = bulky_threshold(ontology)
    os.makedirs(out, exist_ok=True)
    customers = _read(os.path.join(raw, "olist_customers_dataset.csv"))
    sellers = _read(os.path.join(raw, "olist_sellers_dataset.csv"))
    products = _read(os.path.join(raw, "olist_products_dataset.csv"))
    orders = _read(os.path.join(raw, "olist_orders_dataset.csv"))
    items = _read(os.path.join(raw, "olist_order_items_dataset.csv"))

    # --- Order: days_late is delivery minus the promised date; only delivered orders have one ---
    order_rows: list[dict[str, Any]] = []
    days_late_by_order: dict[str, float] = {}
    for o in orders:
        late = _days_between(
            o.get("order_delivered_customer_date", ""), o.get("order_estimated_delivery_date", "")
        )
        row = {
            "order_id": o["order_id"],
            "customer_id": o["customer_id"],
            "status": o.get("order_status", ""),
            "days_late": "" if late is None else f"{late:.3f}",
        }
        order_rows.append(row)
        if late is not None:
            days_late_by_order[o["order_id"]] = late

    # --- Seller.on_time_rate: over that seller's delivered lines ---
    seen: set[tuple[str, str]] = set()
    tally: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # seller -> [on_time, total]
    for it in items:
        key = (it["seller_id"], it["order_id"])
        if key in seen:
            continue  # one order counts once per seller, not once per line
        seen.add(key)
        late = days_late_by_order.get(it["order_id"])
        if late is None:
            continue
        counts = tally[it["seller_id"]]
        counts[1] += 1
        if late <= 0:
            counts[0] += 1

    seller_rows = []
    for s in sellers:
        on_time, total = tally.get(s["seller_id"], [0, 0])
        seller_rows.append({
            "seller_id": s["seller_id"],
            "seller_state": s.get("seller_state", ""),
            "on_time_rate": f"{on_time / total:.4f}" if total else "",
        })

    # --- Product / BulkyProduct: the subtype is decided by the same threshold the guard uses ---
    plain, bulky = [], []
    for p in products:
        w = _num(p.get("product_weight_g"))
        dims = [_num(p.get(f"product_{d}_cm")) for d in ("length", "height", "width")]
        volume = dims[0] * dims[1] * dims[2] if all(d is not None for d in dims) else None  # type: ignore[operator]
        row = {
            "product_id": p["product_id"],
            "weight_g": "" if w is None else f"{w:.1f}",
            "volume_cm3": "" if volume is None else f"{volume:.1f}",
            "category": p.get("product_category_name", ""),
        }
        (bulky if (w is not None and w > threshold) else plain).append(row)

    item_rows = [
        {
            "item_id": f"{it['order_id']}#{it['order_item_id']}",
            "order_id": it["order_id"],
            "product_id": it["product_id"],
            "seller_id": it["seller_id"],
            "freight_value": it.get("freight_value", ""),
            "price": it.get("price", ""),
        }
        for it in items
    ]

    customer_rows = [
        {"customer_id": c["customer_id"], "customer_state": c.get("customer_state", "")}
        for c in customers
    ]

    tables = {
        "Seller": seller_rows,
        "Product": plain,
        "BulkyProduct": bulky,
        "OrderItem": item_rows,
        "Order": order_rows,
        "Customer": customer_rows,
    }
    for name, rows in tables.items():
        path = os.path.join(out, f"{name}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return {name: len(rows) for name, rows in tables.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default="examples/olist_raw")
    ap.add_argument("--out", default="examples/supply_data")
    ap.add_argument("--ontology", default="examples/supply_ontology.yaml")
    ap.add_argument("--download", action="store_true", help="fetch the source tables first")
    args = ap.parse_args()
    if args.download:
        download(args.raw)
    print(f"bulky threshold from ontology: {bulky_threshold(args.ontology):.0f} g")
    for name, n in build(args.raw, args.out, ontology=args.ontology).items():
        print(f"  {name:<14} {n:>7} rows")

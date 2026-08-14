"""Build the supply dataset from the published Olist tables. One script, two steps.

    step 1   olist_raw/  ->  supply_full/     one CSV per entity type in Ω
    step 2   supply_full/ ->  supply_sample/  a small cut, committed to the repository

Source: the Olist Brazilian e-commerce dataset, published by Olist itself
(github.com/olist/work-at-olist-data). Real marketplace records, anonymised by the publisher.
Five tables, ~44MB, downloaded by ``--download``; not kept in the repository.

STEP 1 is not a rename. Ω declares six entity types and the source has five tables, because
OrderItem is a modelling decision rather than a table: an order can hold several lines and one
product may be sold by several sellers, so the seller belongs to the line, not to the product.
Four columns are also computed here, since the source does not carry them:

  Seller.on_time_rate     share of that seller's delivered orders that arrived by the promise
  Seller.delivered_lines  the denominator of that rate
  Order.days_late         delivered date minus promised date
  Product.volume_cm3      length x height x width

Those must be computed over the WHOLE dataset, which is why this step exists separately from
step 2. A seller's punctuality worked out from the 5 of their 300 orders that happened to fall
in a sample is not that seller's punctuality.

STEP 2 exists because the full set is ~31MB and stays out of the repository, which left a clone
unable to run the supply scenario at all — its endpoints returned 503 and its tests skipped, so
the half of Ω a single table cannot exercise (relations, ⪯, norms) was unreachable without
running this script first. The committed sample is 1.4MB and needs no network.

The sample is chosen by following Ω's relations rather than by sampling each table: take a block
of orders, then exactly the lines belonging to them, then exactly the products, sellers and
customers those reference. Sampling tables independently would be smaller and would leave an
OrderItem pointing at a Product that is not there — every declared relation dead-ending, which
at runtime is indistinguishable from missing data. Orders are taken in file order, not at
random, so the sample is byte-identical on every machine.

A caveat worth carrying into any reading of the result: ``days_late`` is measured against the
marketplace's own estimated delivery date, so a seller can look punctual because it was given a
generous promise. This is a measurement-validity limit, not a data-quality one.

Usage:
    python examples/build_supply_data.py --download     # both steps
    LOKA_SUPPLY_DATA=examples/supply_full uvicorn loka_api.app:app   # serve the full set
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
            # The denominator, carried alongside the rate. A rate of 0.67 over three orders and
            # over three hundred are not the same fact, and a norm that forbids acting on thin
            # evidence has nothing to test unless the count survives into the data.
            "delivered_lines": str(total),
        })

    # --- Product / BulkyProduct: the subtype is decided by the same threshold the guard uses ---
    # ⪯ is containment, not partition. Every BulkyProduct IS a Product, so a bulky row is
    # written to BOTH files: Product carries the whole extent, BulkyProduct the subtype's.
    # Splitting them into disjoint sets — which this did — makes the relation
    # OrderItem --of_product--> Product dead-end on exactly the heavy items, and makes "all
    # products" silently exclude the very rows the ShipStandard guard exists to catch.
    all_products, bulky = [], []
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
        all_products.append(row)
        if w is not None and w > threshold:
            bulky.append(row)

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
        "Product": all_products,
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


_SAMPLE_ENTITIES = ("Order", "OrderItem", "Product", "BulkyProduct", "Seller", "Customer")


def _read_with_header(path: str) -> tuple[list[str], list[dict[str, str]]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def _write(path: str, header: list[str], rows: list[dict[str, str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def cut_sample(src: str, out: str, n_orders: int) -> dict[str, int]:
    tables = {
        name: _read_with_header(os.path.join(src, f"{name}.csv"))
        for name in _SAMPLE_ENTITIES
    }

    order_hdr, orders = tables["Order"]
    kept_orders = orders[:n_orders]
    order_ids = {r["order_id"] for r in kept_orders}

    item_hdr, items = tables["OrderItem"]
    kept_items = [r for r in items if r["order_id"] in order_ids]

    # Only what the kept items and orders actually reference — the closure, not a second sample.
    product_ids = {r["product_id"] for r in kept_items}
    seller_ids = {r["seller_id"] for r in kept_items}
    customer_ids = {r["customer_id"] for r in kept_orders}

    prod_hdr, products = tables["Product"]
    kept_products = [r for r in products if r["product_id"] in product_ids]
    bulky_hdr, bulky = tables["BulkyProduct"]
    kept_bulky = [r for r in bulky if r["product_id"] in product_ids]
    seller_hdr, sellers = tables["Seller"]
    kept_sellers = [r for r in sellers if r["seller_id"] in seller_ids]
    cust_hdr, customers = tables["Customer"]
    kept_customers = [r for r in customers if r["customer_id"] in customer_ids]

    os.makedirs(out, exist_ok=True)
    written = {
        "Order": (order_hdr, kept_orders),
        "OrderItem": (item_hdr, kept_items),
        "Product": (prod_hdr, kept_products),
        "BulkyProduct": (bulky_hdr, kept_bulky),
        "Seller": (seller_hdr, kept_sellers),
        "Customer": (cust_hdr, kept_customers),
    }
    for name, (hdr, rows) in written.items():
        _write(os.path.join(out, f"{name}.csv"), hdr, rows)

    _assert_closed(written)
    return {name: len(rows) for name, (_, rows) in written.items()}


def _assert_closed(written: dict[str, tuple[list[str], list[dict[str, str]]]]) -> None:
    """Every foreign key resolves inside the sample. Checked here, not assumed: a sample that
    silently loses closure produces a demo where a declared relation dead-ends at runtime."""
    ids = {
        "order": {r["order_id"] for r in written["Order"][1]},
        "product": {r["product_id"] for r in written["Product"][1]},
        "seller": {r["seller_id"] for r in written["Seller"][1]},
        "customer": {r["customer_id"] for r in written["Customer"][1]},
    }
    for row in written["OrderItem"][1]:
        for key, bucket in (("order_id", "order"), ("product_id", "product"),
                            ("seller_id", "seller")):
            if row[key] not in ids[bucket]:
                raise SystemExit(f"OrderItem {row['item_id']} references missing {key}")
    for row in written["Order"][1]:
        if row["customer_id"] not in ids["customer"]:
            raise SystemExit(f"Order {row['order_id']} references missing customer")
    for row in written["BulkyProduct"][1]:
        if row["product_id"] not in ids["product"]:
            raise SystemExit("BulkyProduct instance is not an instance of Product (violates ⪯)")



if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default="examples/olist_raw")
    ap.add_argument("--full", default="examples/supply_full")
    ap.add_argument("--sample", default="examples/supply_sample")
    ap.add_argument("--sample-orders", type=int, default=4000)
    ap.add_argument("--ontology", default="examples/supply_ontology.yaml")
    ap.add_argument("--download", action="store_true", help="fetch the source tables first")
    args = ap.parse_args()

    if args.download:
        download(args.raw)
    if not os.path.isdir(args.raw):
        raise SystemExit(f"{args.raw} not found — re-run with --download")

    print(f"bulky threshold from ontology: {bulky_threshold(args.ontology):.0f} g")
    print(f"\nstep 1  Olist tables -> one CSV per entity  ({args.full})")
    for name, n in build(args.raw, args.full, ontology=args.ontology).items():
        print(f"  {name:<14} {n:>7,} rows")

    print(f"\nstep 2  cut a committed sample, closed under Ω's relations  ({args.sample})")
    for name, n in cut_sample(args.full, args.sample, args.sample_orders).items():
        print(f"  {name:<14} {n:>7,} rows")
    size = sum(
        os.path.getsize(os.path.join(args.sample, f"{n}.csv")) for n in _SAMPLE_ENTITIES
    )
    print(f"  referentially closed, {size / 1e6:.1f} MB — this is what the repository carries")

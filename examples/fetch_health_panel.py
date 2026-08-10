"""Fetch the real World Bank health panel used by the health demo.

Pulls 8 WDI indicators (2000-2023) for all real countries, keeps complete-case rows, and writes
examples/health_panel.csv. Re-run to refresh. No API key needed.

    python examples/fetch_health_panel.py
"""

from __future__ import annotations

import csv
import json
import os
import urllib.request

# column name -> World Bank indicator code
COLS = [
    ("under5_mortality", "SH.DYN.MORT"),          # outcome
    ("health_exp_per_capita", "SH.XPD.CHEX.PC.CD"),  # policy dial
    ("gdp_per_capita", "NY.GDP.PCAP.CD"),         # control: income
    ("immunization_dpt", "SH.IMM.IDPT"),          # control: immunization
    ("sanitation_access", "SH.STA.BASS.ZS"),      # control: sanitation
    ("water_access", "SH.H2O.BASW.ZS"),           # control: water
    ("fertility_rate", "SP.DYN.TFRT.IN"),         # control: demography
    ("urban_pct", "SP.URB.TOTL.IN.ZS"),           # control: urbanization
]
_BASE = "https://api.worldbank.org/v2"


def _indicator(code: str) -> dict[tuple[str, str], float]:
    url = f"{_BASE}/country/all/indicator/{code}?format=json&per_page=20000&date=2000:2023"
    data = json.loads(urllib.request.urlopen(url, timeout=60).read())
    rows = data[1] or []
    return {(r["countryiso3code"], r["date"]): r["value"] for r in rows if r["value"] is not None}


def _real_countries() -> dict[str, str]:
    data = json.loads(urllib.request.urlopen(f"{_BASE}/country?format=json&per_page=400", timeout=60).read())
    return {c["id"]: c["name"] for c in data[1] if c.get("region", {}).get("value") != "Aggregates"}


def main() -> None:
    real = _real_countries()
    series = {name: _indicator(code) for name, code in COLS}

    keys = set(series[COLS[0][0]])
    for name, _ in COLS[1:]:
        keys &= set(series[name])  # complete cases: every indicator present

    rows = []
    for iso, yr in keys:
        if iso in real:
            rows.append([iso, real[iso], int(yr)] + [round(series[n][(iso, yr)], 3) for n, _ in COLS])
    rows.sort(key=lambda r: (r[0], r[2]))

    out = os.path.join(os.path.dirname(__file__), "health_panel.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iso3", "country", "year"] + [n for n, _ in COLS])
        w.writerows(rows)
    print(f"wrote {len(rows)} rows, {len({r[0] for r in rows})} countries -> {out}")


if __name__ == "__main__":
    main()

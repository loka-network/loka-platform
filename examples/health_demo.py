"""Health scenario demo: raise a country's health spending -> project child mortality.

Real World Bank panel (examples/health_panel.csv). For a target country it prints, at several
health-spending levels, TWO projections:

  - NAIVE      : spending -> mortality with no controls (the seductive big-drop number);
  - CONTROLLED : holding income / sanitation / immunization / water / fertility / urbanization
                 fixed at the country's real values (the honest, modest number).

The contrast is the point: a naive model oversells "more money -> far fewer deaths"; controlling
for the real drivers shows spending alone is a modest lever — and the answer is labelled
observational, not an identified causal effect.

    PYTHONPATH=services/api python examples/health_demo.py [ISO3] [spend levels...]
    e.g. PYTHONPATH=services/api python examples/health_demo.py ZMB 100 150 200
"""

from __future__ import annotations

import csv
import os
import sys

from loka_api.projection import controlled_projection

CONTROLS = [
    "gdp_per_capita", "immunization_dpt", "sanitation_access",
    "water_access", "fertility_rate", "urban_pct",
]
LOG = ["health_exp_per_capita", "gdp_per_capita"]
OUTCOME, DIAL = "under5_mortality", "health_exp_per_capita"


def main() -> None:
    iso = (sys.argv[1] if len(sys.argv) > 1 else "ZMB").upper()
    levels = [float(x) for x in sys.argv[2:]] or [100.0, 150.0, 200.0]

    path = os.path.join(os.path.dirname(__file__), "health_panel.csv")
    panel = list(csv.DictReader(open(path)))
    country_rows = [r for r in panel if r["iso3"] == iso]
    if not country_rows:
        print(f"no rows for {iso}; try ZMB / NGA / THA / IND ...")
        return
    target = max(country_rows, key=lambda r: int(r["year"]))
    name, year = target["country"], target["year"]

    print(f"== {name} ({iso}), latest year {year} ==")
    print(f"   current health spending: ${float(target[DIAL]):.0f}/capita")
    print(f"   current under-5 mortality: {float(target[OUTCOME]):.1f} per 1,000")
    print(f"   panel: {len(panel)} country-years, {len({r['iso3'] for r in panel})} countries\n")

    print(f"   {'spend $':>8} | {'NAIVE (no controls)':>24} | {'CONTROLLED (drivers fixed)':>30}")
    print("   " + "-" * 70)
    for n in levels:
        naive = controlled_projection(
            panel, outcome=OUTCOME, dial=DIAL, controls=[], target=target,
            new_dial=n, log_cols=[DIAL],
        )
        ctrl = controlled_projection(
            panel, outcome=OUTCOME, dial=DIAL, controls=CONTROLS, target=target,
            new_dial=n, log_cols=LOG,
        )
        nv = f"{naive['projected_outcome']:.1f}  [{naive['interval_95'][0]:.0f},{naive['interval_95'][1]:.0f}]"
        cv = f"{ctrl['projected_outcome']:.1f}  [{ctrl['interval_95'][0]:.0f},{ctrl['interval_95'][1]:.0f}]"
        print(f"   {n:>8.0f} | {nv:>24} | {cv:>30}")

    print(
        "\n   Reading: the naive column oversells 'more money -> far fewer child deaths'.\n"
        "   Controlling for income/sanitation/immunization, spending alone is a modest lever.\n"
        "   Both are OBSERVATIONAL projections (association), not identified causal effects."
    )


if __name__ == "__main__":
    main()

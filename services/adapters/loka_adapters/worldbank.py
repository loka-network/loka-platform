"""World Bank adapter — a read-only MemoryAdapter over live World Bank public APIs.

A real data source (no credentials required): the projects API (funding side) and the WDI
indicator API (outcome side). It implements the same read-only, scoped, streaming, lineage-
tagged contract as every other adapter, so it drops straight into ``WorldState.ingest_from``.

The HTTP fetch is injectable (``fetch=``) so the row-mapping is unit-testable offline; the
default hits the live endpoints. Which endpoint a query uses is driven by the predicate's
``entity_type`` (Grant/Program → projects, Outcome → WDI) and its ``filters``.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime

from loka_schemas.adapter import AdapterError, AuthenticationError, Certificate, ScopeError, Session
from loka_schemas.data import Lineage, TypedPredicate, TypedRow

_PROJECTS_URL = "https://search.worldbank.org/api/v2/projects"
_WDI_URL = "https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"

Fetch = Callable[[str], object]  # url -> parsed JSON


def _http_get_json(url: str, timeout: float = 30.0) -> object:
    req = urllib.request.Request(url, headers={"User-Agent": "loka-adapter/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (trusted public API)
        return json.load(resp)


class WorldBankAdapter:
    """Read-only adapter over World Bank Projects + WDI."""

    #: entity types served from the projects endpoint (funding side)
    PROJECT_ENTITIES = frozenset({"Grant", "Program"})

    def __init__(
        self,
        adapter_id: str = "worldbank",
        *,
        fetch: Fetch | None = None,
        now: Callable[[], datetime] | None = None,
        timeout: float = 30.0,
    ) -> None:
        """``timeout`` bounds one HTTP read. A whole-panel pull returns tens of thousands of rows
        and needs longer than a single-country one, so the caller that knows the size sets it."""
        self._adapter_id = adapter_id
        self._fetch = fetch or (lambda url: _http_get_json(url, timeout))
        self._now = now or (lambda: datetime.now(UTC))

    async def authenticate(self, cert: Certificate) -> Session:
        if not cert.subject:
            raise AuthenticationError("certificate has no subject")
        return Session(subject=cert.subject, scopes=cert.scopes, established_at=self._now())

    def _lineage(self, source: str) -> Lineage:
        return Lineage(source=source, adapter_id=self._adapter_id, retrieved_at=self._now())

    async def query(
        self, predicate: TypedPredicate, session: Session
    ) -> AsyncIterator[TypedRow]:
        if not ("*" in session.scopes or predicate.entity_type in session.scopes):
            raise ScopeError(f"session {session.subject} not scoped for {predicate.entity_type}")
        if predicate.entity_type in self.PROJECT_ENTITIES:
            for row in self._query_projects(predicate):
                yield row
        elif predicate.entity_type == "Outcome":
            for row in self._query_wdi(predicate):
                yield row
        else:
            raise AdapterError(f"WorldBankAdapter has no source for {predicate.entity_type}")

    def _query_projects(self, predicate: TypedPredicate) -> list[TypedRow]:
        f = predicate.filters
        params = {"format": "json", "rows": str(f.get("rows", 20))}
        params["countrycode"] = str(f.get("country", "ZM"))
        if "qterm" in f:
            params["qterm"] = str(f["qterm"])
        if "sector" in f:
            params["sector"] = str(f["sector"])
        data = self._fetch(f"{_PROJECTS_URL}?{urllib.parse.urlencode(params)}")
        if not isinstance(data, dict):
            raise AdapterError("projects response is not an object")
        projects = data.get("projects")
        if not isinstance(projects, dict):
            return []
        lineage = self._lineage("worldbank_projects")
        rows: list[TypedRow] = []
        for pid, p in projects.items():
            if not isinstance(p, dict):
                continue
            amount = str(p.get("totalamt", "0")).replace(",", "") or "0"
            countries = p.get("countryname") or [""]
            rows.append(
                TypedRow(
                    entity_type=predicate.entity_type,
                    values={
                        "id": str(pid),
                        "name": str(p.get("project_name", ""))[:60],
                        "amount_usd": float(amount),
                        "status": str(p.get("projectstatusdisplay", "")),
                        "country": str(countries[0] if isinstance(countries, list) else countries),
                    },
                    lineage=lineage,
                )
            )
        return rows

    def _query_wdi(self, predicate: TypedPredicate) -> list[TypedRow]:
        f = predicate.filters
        indicator = str(f.get("indicator", ""))
        if not indicator:
            raise AdapterError("Outcome query needs an 'indicator' filter (e.g. EG.ELC.ACCS.ZS)")
        # ``country`` accepts an ISO3 code or "all"; the API paginates, and its default page is
        # small enough that a multi-country pull would silently truncate, so the page size is
        # explicit. The country is carried on each row: without it a panel spanning countries
        # cannot tell whose observation it is holding.
        country = str(f.get("country", "ZMB"))
        url = _WDI_URL.format(country=country, indicator=indicator)
        params = {
            "format": "json",
            "date": str(f.get("date", "2010:2022")),
            "per_page": str(f.get("per_page", 20000 if country == "all" else 100)),
        }
        data = self._fetch(f"{url}?{urllib.parse.urlencode(params)}")
        if not isinstance(data, list) or len(data) < 2 or not isinstance(data[1], list):
            return []
        lineage = self._lineage("worldbank_wdi")
        rows: list[TypedRow] = []
        for r in data[1]:
            if not isinstance(r, dict) or r.get("value") is None:
                continue
            year = str(r.get("date"))
            iso3 = str(r.get("countryiso3code") or "").strip()
            rows.append(
                TypedRow(
                    entity_type="Outcome",
                    values={
                        "id": f"{iso3 or country}:{indicator}:{year}",
                        "country": iso3 or country,
                        "indicator_code": indicator,
                        "value": float(r["value"]),
                        "as_of": year,
                    },
                    lineage=lineage,
                )
            )
        return rows

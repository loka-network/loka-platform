"""Method — an entry in KB.METHODS.

The professor's query workflow (slide 7) splits a query into a DATA request (asks -> retrieve
from KB.DATA) or a METHOD request (orders -> apply a method from KB.METHODS). A ``Method`` is
the typed descriptor of one such method; the callable that implements it lives with the query
service, so this contract stays data-only and shareable.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Method:
    """Typed descriptor of a KB.METHODS entry (name + I/O types)."""

    name: str
    description: str
    in_types: tuple[str, ...] = ()
    out_type: str = "object"

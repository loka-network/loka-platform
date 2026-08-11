"""Test helper: take a built ontology through review to publication.

A generated ontology enters as a draft and cannot authorize an answer. Tests that want to query
a built KB must walk the same path a reviewer walks — which is the point, so the tests document
the lifecycle rather than bypassing it.
"""

from __future__ import annotations

from typing import Any


def publish_built_ontology(client: Any, built: dict[str, Any], version: str = "reviewed-v1") -> str:
    """Review-and-publish the ontology a /build-kb response proposed. Returns the ontology id."""
    oid = built["ontology_id"]
    edited = client.put(f"/ontology/{oid}", json={"ontology_yaml": built["ontology_yaml"]})
    assert edited.status_code == 200, edited.text
    published = client.post(f"/ontology/{oid}/publish", json={"version": version})
    assert published.status_code == 200, published.text
    return str(oid)

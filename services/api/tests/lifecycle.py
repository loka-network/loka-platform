"""Test helper: take a built ontology through review to publication.

A generated ontology enters as a draft and cannot authorize an answer. Tests that want to query
a built KB must walk the same path a reviewer walks — which is the point, so the tests document
the lifecycle rather than bypassing it.
"""

from __future__ import annotations

from typing import Any


def publish_built_ontology(
    client: Any,
    built: dict[str, Any],
    version: str = "reviewed-v1",
    *,
    declare: dict[str, list[tuple[str, str]]] | None = None,
) -> str:
    """Review-and-publish the ontology a /build-kb response proposed. Returns the ontology id.

    ``declare`` adds typed properties to an entity during review — the edit a reviewer makes when
    the checklist points out that a builder reading prose could not infer them. Data for a
    property Ω does not declare is refused at ingest, so a test that ingests values has to
    declare them here, exactly as an operator would.
    """
    oid = built["ontology_id"]
    yaml_text = built["ontology_yaml"]
    for entity, props in (declare or {}).items():
        block = "".join(f"      - {{name: {n}, type: {t}}}\n" for n, t in props)
        yaml_text = yaml_text.replace(
            f"  - type: {entity}\n", f"  - type: {entity}\n    properties:\n{block}", 1
        )
    edited = client.put(f"/ontology/{oid}", json={"ontology_yaml": yaml_text})
    assert edited.status_code == 200, edited.text
    published = client.post(f"/ontology/{oid}/publish", json={"version": version})
    assert published.status_code == 200, published.text
    return str(oid)

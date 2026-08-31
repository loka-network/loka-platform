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
    if declare:
        # Edited through the parser rather than by string replacement. The draft is written in
        # the verb-signature notation, where an entity that already declares attributes and one
        # that declares none are different shapes; a textual insert gets one of them wrong, and
        # gets it wrong silently — the property simply never arrives, and the failure surfaces
        # later as data the ontology does not admit.
        import yaml as _yaml

        doc = _yaml.safe_load(yaml_text)
        for entity, props in declare.items():
            body = doc["entities"].get(entity) or {}
            has = body.get("has") or {}
            has.update({n: {"type": t} for n, t in props})
            body["has"] = has
            doc["entities"][entity] = body
        yaml_text = _yaml.safe_dump(doc, sort_keys=False, allow_unicode=True)
    edited = client.put(f"/ontology/{oid}", json={"ontology_yaml": yaml_text})
    assert edited.status_code == 200, edited.text
    published = client.post(f"/ontology/{oid}/publish", json={"version": version})
    assert published.status_code == 200, published.text
    return str(oid)

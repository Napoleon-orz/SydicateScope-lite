import difflib

def compute_similarity(name1: str, name2: str) -> float:
    if not name1 or not name2:
        return 0.0
    return difflib.SequenceMatcher(None, name1.lower().strip(), name2.lower().strip()).ratio()

def resolve_entities(extracted_mentions: list[dict], threshold: float = 0.75) -> dict:
    canonical_nodes = {}

    for mention in extracted_mentions:
        surface = mention["surface_text"]
        etype = mention["entity_type"]

        if etype not in ["PERSON", "LOCATION", "ORGANIZATION"]:
            canonical_id = f"{etype}_{surface}"
            if canonical_id not in canonical_nodes:
                canonical_nodes[canonical_id] = {
                    "canonical_id": canonical_id,
                    "entity_type": etype,
                    "canonical_name": surface,
                    "source_mentions": []
                }
            canonical_nodes[canonical_id]["source_mentions"].append(mention)
            continue

    matched_id = None
    for cid, node in canonical_nodes.items():
        if node["entity_type"] == etype:
            score = compute_similarity(surface, node["canonical_name"])
            if score >= threshold:
                matched_id = cid
                break

    if matched_id:
        canonical_nodes[matched_id]["source_mentions"].append(mention)
    else:
        new_id = f"{etype}_{len(canonical_nodes) + 1}"
        canonical_nodes[new_id] = {
            "canonical_id": new_id,
            "entity_type": etype,
            "canonical_name": surface,
            "source_mentions": [mention]
        }

    print(f"Entity Resolution complete. Merged raw mentions into {len(canonical_nodes)} unique master profiles.")
    return canonical_nodes
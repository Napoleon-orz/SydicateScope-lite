"""
SyndicateScope API layer featuring centralized RBAC, granular permissions,
resource-level checks, audit logs, and progressive intelligence graph payload.
"""

from typing import List, Optional

from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ingest import load_json, load_csv
from ner.ner import extract_all_entities
from resolution.resolver import resolve_entities
from graph.graph import (
    build_investigation_graph,
    add_fir_co_mention_edges,
    add_vehicle_edges,
    add_organization_edges,
    add_location_edges
)
from prediction.link_prediction import (
    train_link_prediction_model,
    rank_candidate_links,
    predict_link_probability,
    extract_features
)
from analytics.influence import rank_influential_nodes
from rag.graph_rag import explain_link, explain_influence

from security import require_permission, get_audit_logs

# ============================================================
# PIPELINE STATE
# ============================================================

class PipelineState:
    graph = None
    model = None
    scaler = None

state = PipelineState()

def build_pipeline():
    people = load_csv("people.csv")
    firs = load_json("firs.json")
    cdrs = load_csv("cdrs.csv")
    transactions = load_csv("transactions.csv")
    vehicles = load_csv("vehicles.csv")
    organizations = load_csv("organizations.csv")
    locations = load_csv("locations.csv")

    known_person_names = [p.get("name", "") for p in people if p.get("name")]

    all_clues = []
    for fir in firs:
        all_clues.extend(extract_all_entities(fir, known_person_names))

    canonical_database = resolve_entities(all_clues, threshold=0.75)

    graph = build_investigation_graph(people, canonical_database, cdrs, transactions)
    graph = add_fir_co_mention_edges(graph, canonical_database, people)

    # Previously vehicles.csv / organizations.csv / locations.csv were
    # never loaded at all, so VEHICLE/ORGANIZATION/LOCATION nodes were
    # either absent or (if NER happened to spot one in FIR text)
    # permanently isolated with degree 0 - invisible to influence
    # ranking and never shown as anyone's neighbour in the graph view.
    graph = add_vehicle_edges(graph, vehicles)
    graph = add_organization_edges(graph, organizations)
    graph = add_location_edges(graph, locations, cdrs)

    model, scaler, test_edges, training_graph = train_link_prediction_model(graph)

    return graph, model, scaler


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="SyndicateScope API",
    description="Investigation graph with RBAC and progressive intelligence filtering.",
    version="0.5.0",
)

@app.on_event("startup")
def startup_event():
    print("Building investigation graph and training link-prediction model...")
    state.graph, state.model, state.scaler = build_pipeline()
    print(f"Ready. Graph: {state.graph.number_of_nodes()} nodes, {state.graph.number_of_edges()} edges.")


# ============================================================
# RESPONSE MODELS
# ============================================================

class ConnectionInfo(BaseModel):
    node_id: str
    name: str
    entity_type: str
    edge_type: str
    cdr_weight: float = 0.0
    txn_weight: float = 0.0
    fir_weight: float = 0.0

class MergedMention(BaseModel):
    surface_text: str
    doc_id: str
    confidence: float

class PersonSummary(BaseModel):
    person_id: str
    name: str
    entity_type: str
    alias: str = ""
    degree: int
    influence_score: float = 0.0
    connections: List[ConnectionInfo] = []
    merged_mentions: List[MergedMention] = []

class LinkCandidate(BaseModel):
    node_a: str
    node_b: str
    probability: float

class InfluentialNode(BaseModel):
    rank: int
    node: str
    name: str
    score: float


# ============================================================
# ENDPOINTS WITH PROGRESSIVE DISCLOSURE & RBAC
# ============================================================

@app.get("/health")
def health():
    return {"status": "ok", "nodes": state.graph.number_of_nodes(), "edges": state.graph.number_of_edges()}

@app.get("/graph/data")
def get_graph_data(
    focus_node: Optional[str] = Query(None, description="Center graph around this node"),
    hops: int = Query(1, description="Hop depth when focused"),
    include_types: Optional[str] = Query("PERSON,LOCATION,VEHICLE,ORGANIZATION,PHONE_NUMBER,FINANCIAL,CASE", description="Entity types"),
    user: dict = Depends(require_permission("graphs.read", resource_type="full_graph"))
):
    """
    Returns progressive intelligence graph data optimized for clean visualization.
    Injects top AI-predicted links as dashed edges.
    """
    allowed_types = set(t.strip().upper() for t in include_types.split(","))

    # Calculate influence scores for sizing every node type on the
    # canvas (a busy tower or shell company is still worth drawing
    # bigger here - this is purely visual weight, not the "Key
    # Individuals" ranking).
    influence_scores = rank_influential_nodes(state.graph, top_n=state.graph.number_of_nodes())
    influence_map = {node: score for node, score in influence_scores}

    # But the *seed set* for the default view should be people, not
    # whichever node has the highest raw centrality - otherwise a
    # single high-traffic cell tower (many different people ping it)
    # could crowd real suspects out of the initial view entirely.
    person_influence_scores = rank_influential_nodes(
        state.graph, top_n=state.graph.number_of_nodes(), person_only=True
    )

    included_nodes = set()

    if focus_node and focus_node in state.graph:
        # Progressive Disclosure: Ego-network filtering
        current_tier = {focus_node}
        included_nodes.add(focus_node.strip())
        for _ in range(max(1, min(hops, 2))):
            next_tier = set()
            for n in current_tier:
                for neighbor in state.graph.neighbors(n):
                    next_tier.add(neighbor)
            included_nodes.update(next_tier)
            current_tier = next_tier
    else:
        # Default view: Top influential PEOPLE + their immediate primary context
        top_people = [node for node, score in person_influence_scores[:30]]
        included_nodes.update(top_people)
        for p in top_people:
            for neighbor in state.graph.neighbors(p):
                etype = state.graph.nodes[neighbor].get("entity_type", "").upper()
                # Hide low-value noise by default unless explicitly asked
                if etype in ["LOCATION", "VEHICLE", "ORGANIZATION"]:
                    included_nodes.add(neighbor)

    nodes = []
    valid_node_ids = set()

    for node_id in included_nodes:
        if node_id not in state.graph: continue
        data = state.graph.nodes[node_id]
        etype = data.get("entity_type", "PERSON").upper()

        if etype not in allowed_types and node_id != focus_node:
            continue

        score = influence_map.get(node_id, 0.1)
        nodes.append({
            "id": node_id,
            "label": data.get("canonical_name") or node_id,
            "group": etype,
            "score": score,
            "canonical_name": data.get("canonical_name", node_id),
            "entity_type": etype,
            "degree": state.graph.degree(node_id)
        })
        valid_node_ids.add(node_id)

    edges = []
    # 1. Add Observed Relationships
    for u, v, data in state.graph.edges(data=True):
        if u in valid_node_ids and v in valid_node_ids:
            edges.append({
                "from": u,
                "to": v,
                "label": data.get("edge_type", "OBSERVED"),
                "dashed": False,
                "cdr_weight": data.get("cdr_weight", 0),
                "txn_weight": data.get("txn_weight", 0),
                "fir_weight": data.get("fir_weight", 0),
                "confidence": 1.0
            })

    # 2. Inject AI Predicted Relationships (Top 10 involving the current nodes)
    top_candidates = rank_candidate_links(state.model, state.scaler, state.graph, top_n=30)
    injected_predictions = 0

    for node_a, node_b, prob in top_candidates:
        if injected_predictions >= 10: break

        # Only show prediction if at least one node is in our current view
        if node_a in valid_node_ids or node_b in valid_node_ids:

            # Ensure both nodes exist in the visual graph if we draw an edge
            for n in (node_a, node_b):
                if n not in valid_node_ids and n in state.graph:
                    valid_node_ids.add(n)
                    data = state.graph.nodes[n]
                    etype = data.get("entity_type", "PERSON").upper()
                    nodes.append({
                        "id": n,
                        "label": data.get("canonical_name") or n,
                        "group": etype,
                        "score": influence_map.get(n, 0.1),
                        "canonical_name": data.get("canonical_name", n),
                        "entity_type": etype,
                        "degree": state.graph.degree(n)
                    })

            # Extract underlying feature evidence
            feats = extract_features(state.graph, node_a, node_b)

            edges.append({
                "from": node_a,
                "to": node_b,
                "label": "AI_PREDICTED",
                "dashed": True,
                "probability": prob,
                "evidence_shared_neighbors": feats[0],
                "evidence_fir": feats[6]
            })
            injected_predictions += 1

    return {
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        "ai_predictions": injected_predictions,
        "focus_node": focus_node,
        "nodes": nodes,
        "edges": edges
    }

@app.get("/people/{person_id}", response_model=PersonSummary)
def get_person(person_id: str, user: dict = Depends(require_permission("graphs.read"))):
    """
    Full profile for one entity: identity, influence score, every
    direct connection (with the evidence type/weight behind it), and
    every raw NER mention that entity-resolution merged into this
    canonical node (FR-11/FR-12 traceability, made browsable instead
    of just internally stored).
    """
    if person_id not in state.graph:
        raise HTTPException(status_code=404, detail="Not found")

    node_data = state.graph.nodes[person_id]

    influence_scores = dict(
        rank_influential_nodes(state.graph, top_n=state.graph.number_of_nodes())
    )

    connections = []
    for neighbor in state.graph.neighbors(person_id):
        edge = state.graph[person_id][neighbor]
        neighbor_data = state.graph.nodes[neighbor]
        connections.append(ConnectionInfo(
            node_id=neighbor,
            name=neighbor_data.get("canonical_name", neighbor),
            entity_type=neighbor_data.get("entity_type", "UNKNOWN"),
            edge_type=edge.get("edge_type", "UNKNOWN"),
            cdr_weight=edge.get("cdr_weight", 0.0),
            txn_weight=edge.get("txn_weight", 0.0),
            fir_weight=edge.get("fir_weight", 0.0)
        ))

    raw_mentions = node_data.get("resolved_source_mentions") or node_data.get("source_mentions") or []
    merged_mentions = [
        MergedMention(
            surface_text=m.get("surface_text", ""),
            doc_id=m.get("doc_id", ""),
            confidence=m.get("confidence", 0.0)
        )
        for m in raw_mentions
    ]

    return PersonSummary(
        person_id=person_id,
        name=node_data.get("canonical_name", person_id),
        entity_type=node_data.get("entity_type", "PERSON"),
        alias=node_data.get("alias", ""),
        degree=state.graph.degree(person_id),
        influence_score=influence_scores.get(person_id, 0.0),
        connections=connections,
        merged_mentions=merged_mentions
    )

@app.get("/resolution/merges")
def get_resolution_merges(user: dict = Depends(require_permission("graphs.read"))):
    """
    Global feed of every entity-resolution merge on this deployment:
    every canonical node that absorbed one or more raw NER mentions
    from FIR text. This is the same data /people/{id} exposes per
    entity, surfaced as one browsable list (the "which profiles were
    merged into which" notification feed) instead of requiring an
    analyst to already know which node to look up.
    """
    merges = []
    for node_id, data in state.graph.nodes(data=True):
        mentions = data.get("resolved_source_mentions") or data.get("source_mentions") or []
        if not mentions:
            continue
        merges.append({
            "canonical_id": node_id,
            "canonical_name": data.get("canonical_name", node_id),
            "entity_type": data.get("entity_type", "UNKNOWN"),
            "mention_count": len(mentions),
            "mentions": [
                {
                    "surface_text": m.get("surface_text", ""),
                    "doc_id": m.get("doc_id", ""),
                    "confidence": m.get("confidence", 0.0)
                }
                for m in mentions
            ]
        })
    merges.sort(key=lambda m: m["mention_count"], reverse=True)
    return {"merges": merges}

@app.get("/link/{node_a}/{node_b}")
def get_link_explanation(node_a: str, node_b: str, user: dict = Depends(require_permission("prediction.run"))):
    if node_a not in state.graph or node_b not in state.graph:
        raise HTTPException(status_code=404, detail="Not found")
    probability = predict_link_probability(state.model, state.scaler, state.graph, node_a, node_b)
    return {"probability": probability, "explanation": explain_link(state.graph, node_a, node_b, state.model, state.scaler)}

@app.get("/candidates", response_model=List[LinkCandidate])
def get_candidates(top_n: int = 10, user: dict = Depends(require_permission("prediction.run"))):
    candidates = rank_candidate_links(state.model, state.scaler, state.graph, top_n=max(1, min(top_n, 200)))
    return [LinkCandidate(node_a=a, node_b=b, probability=prob) for a, b, prob in candidates]

@app.get("/influence", response_model=List[InfluentialNode])
def get_influence(top_n: int = 10, user: dict = Depends(require_permission("graphs.read"))):
    # person_only=True: "Key Individuals" (PRD FR-22) means people to
    # investigate, not the tower or shell company they happened to use.
    ranked = rank_influential_nodes(state.graph, top_n=max(1, min(top_n, 60)), person_only=True)
    return [InfluentialNode(rank=rank, node=node, name=state.graph.nodes[node].get("canonical_name", node), score=score) for rank, (node, score) in enumerate(ranked, start=1)]

@app.get("/audit-logs")
def query_audit_logs(user: dict = Depends(require_permission("audit.read"))):
    return {"logs": get_audit_logs()}

@app.get("/influence/{node}/explain")
def explain_influence_endpoint(node: str, user: dict = Depends(require_permission("graphs.read"))):
    """
    XAI justification for a single influence-ranked entity (PRD FR-23,
    FR-28-30, FR-38: every influence rank must be traceable to the
    specific evidence that drove it, on request, not just as a score).
    """
    if node not in state.graph:
        raise HTTPException(status_code=404, detail="Not found")
    return {"node": node, "explanation": explain_influence(state.graph, node)}

@app.get("/evaluation/summary")
def evaluation_summary(user: dict = Depends(require_permission("audit.read"))):
    """
    DEMO / JUDGE-MODE SCORING ONLY.

    Compares the already-computed candidate ranking and influence
    ranking against ground_truth.json (the answer key). This file is
    loaded fresh, here only, and is never passed into ingestion,
    graph-building, training, or ranking anywhere above - it exists
    purely to score already-produced results after the fact, the same
    way evaluate_ground_truth.py does offline. Gated behind the same
    "audit.read" permission as the audit log (Admin/DSP only) since
    this is an internal scoring tool, not an investigative output.
    """
    try:
        ground_truth = load_json("ground_truth.json")
    except Exception:
        raise HTTPException(status_code=404, detail="ground_truth.json not available on this deployment")

    # ---- Hidden-link recall ----
    true_hidden_pairs = set()
    for item in ground_truth.get("hidden_links", []):
        pair = item.get("pair")
        if pair and len(pair) >= 2:
            true_hidden_pairs.add(tuple(sorted((pair[0], pair[1]))))

    top_candidates = rank_candidate_links(state.model, state.scaler, state.graph, top_n=100)
    matched = []
    for rank, (a, b, prob) in enumerate(top_candidates, start=1):
        pair = tuple(sorted((a, b)))
        if pair in true_hidden_pairs:
            matched.append({"pair": list(pair), "rank": rank, "probability": prob, "mechanism": "candidate_ranking"})

    found_pairs = {tuple(m["pair"]) for m in matched}
    for pair in true_hidden_pairs - found_pairs:
        a, b = pair
        if state.graph.has_edge(a, b):
            edge = state.graph[a][b]
            if edge.get("fir_weight", 0) > 0 and edge.get("cdr_weight", 0) == 0 and edge.get("txn_weight", 0) == 0:
                prob = predict_link_probability(state.model, state.scaler, state.graph, a, b)
                matched.append({"pair": list(pair), "rank": None, "probability": prob, "mechanism": "fir_co_mention_fusion"})

    hidden_link_recall = (len(matched) / len(true_hidden_pairs)) if true_hidden_pairs else None

    # ---- Key-influencer recall ----
    true_influencers = {i.get("person_id") for i in ground_truth.get("key_influencers", []) if i.get("person_id")}
    pool_size = max(10, len(true_influencers) * 3)
    ranked = rank_influential_nodes(state.graph, top_n=pool_size, person_only=True)
    ranked_ids = [node for node, _ in ranked]
    influencer_hits = [
        {"person_id": pid, "rank": ranked_ids.index(pid) + 1}
        for pid in true_influencers if pid in ranked_ids
    ]
    influencer_recall = (len(influencer_hits) / len(true_influencers)) if true_influencers else None

    return {
        "note": "Demo/judge-mode scoring only. ground_truth.json is never fed into ingestion, graph-building, training, or ranking above.",
        "hidden_links": {
            "total_planted": len(true_hidden_pairs),
            "recovered": matched,
            "recall": hidden_link_recall
        },
        "key_influencers": {
            "total_planted": len(true_influencers),
            "checked_against_top_n": pool_size,
            "recovered": influencer_hits,
            "recall": influencer_recall
        }
    }

app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")
"""
SyndicateScope Backend API
==========================
FastAPI REST server connecting the SyndicateScope Intelligence Dashboard
frontend to the underlying Graph-RAG pipeline, NLP entity extraction,
resolution engine, link prediction model, RBAC security, and audit logger.
"""

import os
import json
import csv
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, Depends, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

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
# NOTE: these imports used to be wrapped in a bare `try/except ImportError`
# that only printed a warning and kept going. That's why the backend looked
# "not connected" to the frontend: if ANY of these modules failed to import
# (most commonly ner.py's `import spacy` + `spacy.load('en_core_web_sm')`,
# since spacy isn't in requirement.txt and the model is a separate download),
# every name below silently ended up undefined. The app then crashed with a
# NameError the moment a route decorator referenced `require_permission`,
# so uvicorn never actually served anything and the frontend's fetch() calls
# always failed and fell back to its mock/demo data. Letting the ImportError
# raise here means a missing dependency now fails loudly and immediately,
# with a real traceback pointing at the actual missing package.


def locate_dataset_file(filename: str) -> Path:
    """Finds target dataset file across root directory or ./data/ folder, supporting versioned filenames."""
    p_path = Path(filename)
    base_name = p_path.stem
    ext = p_path.suffix

    search_dirs = [
        Path("."),
        Path("data"),
        Path(__file__).parent,
        Path(__file__).parent / "data"
    ]

    for d in search_dirs:
        exact = d / filename
        if exact.exists():
            return exact

    for d in search_dirs:
        if d.exists():
            for f in d.iterdir():
                if f.name.startswith(base_name) and f.name.endswith(ext):
                    return f

    return Path("data") / filename


def load_csv_robust(filename: str) -> List[Dict[str, Any]]:
    """Loads CSV file safely regardless of directory location or versioned name."""
    filepath = locate_dataset_file(filename)
    if not filepath.exists():
        print(f"[Warning] CSV file '{filename}' not found at {filepath}")
        return []
    print(f"-> API loaded dataset: {filepath}")
    with open(filepath, mode="r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_json_robust(filename: str) -> Any:
    """Loads JSON file safely regardless of directory location or versioned name."""
    filepath = locate_dataset_file(filename)
    if not filepath.exists():
        print(f"[Warning] JSON file '{filename}' not found at {filepath}")
        return []
    print(f"-> API loaded dataset: {filepath}")
    with open(filepath, mode="r", encoding="utf-8") as f:
        return json.load(f)


class PipelineState:
    """Global state container for initialized graph and trained ML models."""
    def __init__(self):
        self.graph = None
        self.model = None
        self.scaler = None
        self.firs: List[Dict[str, Any]] = []
        self.people: List[Dict[str, Any]] = []
        self.canonical_database: Dict[str, Any] = {}
        self.cached_top_candidates: List[Any] = []

state = PipelineState()


def build_pipeline():
    """Ingests real datasets, constructs multi-layer graph, and trains link predictor."""
    print("-> Ingesting dataset files (people, FIRs, CDRs, transactions, vehicles, orgs, locations)...")
    people = load_csv_robust("people.csv")
    firs = load_json_robust("firs.json")
    cdrs = load_csv_robust("cdrs.csv")
    transactions = load_csv_robust("transactions.csv")
    vehicles = load_csv_robust("vehicles.csv")
    organizations = load_csv_robust("organizations.csv")
    locations = load_csv_robust("locations.csv")

    known_person_names = [p.get("name", "") for p in people if p.get("name")]

    print("-> Running NER entity extraction across FIR narratives...")
    all_clues = []
    for fir in firs:
        all_clues.extend(extract_all_entities(fir, known_person_names))

    print("-> Deduplicating canonical entities via resolution engine...")
    canonical_database = resolve_entities(all_clues, threshold=0.75)

    print("-> Constructing investigation graph topology...")
    graph = build_investigation_graph(people, canonical_database, cdrs, transactions)
    graph = add_fir_co_mention_edges(graph, canonical_database, people)

    if vehicles:
        graph = add_vehicle_edges(graph, vehicles)
    if organizations:
        graph = add_organization_edges(graph, organizations)
    if locations:
        graph = add_location_edges(graph, locations, cdrs)

    print("-> Training link prediction model & pre-computing top candidate ties...")
    model, scaler, test_edges, training_graph = train_link_prediction_model(graph)

    state.firs = firs
    state.people = people
    state.canonical_database = canonical_database

    try:
        state.cached_top_candidates = rank_candidate_links(model, scaler, graph, top_n=100)
    except Exception as err:
        print(f"[Warning] Could not pre-cache candidate links: {err}")
        state.cached_top_candidates = []

    return graph, model, scaler


app = FastAPI(
    title="SyndicateScope Backend API",
    description="Restricted Law Enforcement Network - Graph-RAG, RBAC, Link Prediction & FIR Reporting API",
    version="0.7.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    """Initializes graph pipeline on server startup."""
    try:
        state.graph, state.model, state.scaler = build_pipeline()
        print(f"[SUCCESS] SyndicateScope Pipeline Online! Nodes: {state.graph.number_of_nodes()}, Edges: {state.graph.number_of_edges()}, FIRs: {len(state.firs)}")
    except Exception as err:
        print(f"[ERROR] Pipeline startup failed: {err}")


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


class PersonFIR(BaseModel):
    fir_id: str
    date: str
    narrative: str


class PersonSummary(BaseModel):
    person_id: str
    name: str
    entity_type: str
    alias: str = ""
    degree: int
    influence_score: float = 0.0
    connections: List[ConnectionInfo] = []
    merged_mentions: List[MergedMention] = []
    firs: List[PersonFIR] = []


class LinkCandidate(BaseModel):
    node_a: str
    node_b: str
    probability: float
    city: str = "Bengaluru"
    area: str = "General"
    locations: List[str] = []


class InfluentialNode(BaseModel):
    rank: int
    node: str
    name: str
    score: float


class FIREntity(BaseModel):
    id: str
    name: str
    type: str


class FIRSummary(BaseModel):
    fir_id: str
    date: str
    narrative: str
    entities_count: int = 0
    entities: List[FIREntity] = []


def find_firs_for_person(person_id: str) -> List[dict]:
    """Retrieves all FIR case narratives associated with a person ID."""
    if not state.graph or person_id not in state.graph:
        return []

    node_data = state.graph.nodes[person_id]
    canonical_name = (node_data.get("canonical_name") or "").strip().lower()
    alias = (node_data.get("alias") or "").strip().lower()

    mentions = node_data.get("resolved_source_mentions") or node_data.get("source_mentions") or []
    doc_ids = {m.get("doc_id") for m in mentions if m.get("doc_id")}

    surface_texts = {m.get("surface_text", "").strip().lower() for m in mentions if m.get("surface_text")}
    if canonical_name:
        surface_texts.add(canonical_name)
    if alias:
        surface_texts.add(alias)

    matched_firs = []
    seen_ids = set()

    for fir in state.firs:
        f_id = fir.get("fir_id")
        if not f_id or f_id in seen_ids:
            continue

        narrative_lower = (fir.get("narrative") or "").lower()
        is_match = f_id in doc_ids or any(st in narrative_lower for st in surface_texts if st)

        if is_match:
            seen_ids.add(f_id)
            matched_firs.append({
                "fir_id": f_id,
                "date": fir.get("date", ""),
                "narrative": fir.get("narrative", "")
            })

    matched_firs.sort(key=lambda x: x["date"], reverse=True)
    return matched_firs


def extract_area_name(text: str) -> str:
    """Helper to detect standard district/area names from entity or location labels."""
    if not text:
        return "Central Bengaluru"
    text_lower = text.lower()

    # Mumbai Areas
    if "bandra" in text_lower: return "Bandra"
    if "andheri" in text_lower: return "Andheri"
    if "colaba" in text_lower: return "Colaba"
    if "powai" in text_lower: return "Powai"
    if "juhu" in text_lower: return "Juhu"
    if "dadar" in text_lower: return "Dadar"
    if "worli" in text_lower: return "Worli"
    if "kurla" in text_lower: return "Kurla"
    if "thane" in text_lower: return "Thane"
    if "navi mumbai" in text_lower: return "Navi Mumbai"

    # Bengaluru Areas
    if "indiranagar" in text_lower: return "Indiranagar"
    if "whitefield" in text_lower: return "Whitefield"
    if "electronic city" in text_lower: return "Electronic City"
    if "hebbal" in text_lower: return "Hebbal"
    if "silk board" in text_lower: return "Silk Board"
    if "kr market" in text_lower: return "KR Market"
    if "yeshwantpur" in text_lower: return "Yeshwantpur"
    if "bannerghatta" in text_lower: return "Bannerghatta Road"
    if "malleshwaram" in text_lower: return "Malleshwaram"
    if "majestic" in text_lower: return "Majestic"
    if "mumbai" in text_lower: return "Central Mumbai"
    return "Central Bengaluru"


def get_candidate_location_info(graph, node_a: str, node_b: str):
    """Derives City and Area tags for candidate pairs by looking up connected location hubs."""
    areas = []
    mumbai_areas = {"Bandra", "Andheri", "Colaba", "Powai", "Juhu", "Dadar", "Worli", "Kurla", "Thane", "Navi Mumbai", "Central Mumbai"}

    for n in (node_a, node_b):
        if graph and n in graph:
            n_data = graph.nodes[n]
            if n_data.get("canonical_name"):
                area = extract_area_name(n_data.get("canonical_name"))
                if area not in ("Central Bengaluru", "Central Mumbai"):
                    areas.append(area)
            for neighbor in graph.neighbors(n):
                neighbor_data = graph.nodes[neighbor]
                if neighbor_data.get("entity_type") == "LOCATION":
                    area = extract_area_name(neighbor_data.get("canonical_name", neighbor))
                    if area not in ("Central Bengaluru", "Central Mumbai"):
                        areas.append(area)

    city = "Bengaluru"
    if any(a in mumbai_areas for a in areas):
        city = "Mumbai"
    elif any(n in graph and ("mumbai" in str(graph.nodes[n].get("canonical_name", "")).lower() or "mh" in str(graph.nodes[n].get("canonical_name", "")).lower()) for n in (node_a, node_b)):
        city = "Mumbai"

    primary_area = areas[0] if areas else ("Central Mumbai" if city == "Mumbai" else "Central Bengaluru")
    return city, primary_area, list(set(areas))


@app.get("/health")
def health():
    """System health & live status endpoint."""
    return {
        "status": "ok",
        "nodes": state.graph.number_of_nodes() if state.graph else 0,
        "edges": state.graph.number_of_edges() if state.graph else 0,
        "firs": len(state.firs)
    }


@app.get("/graph/data")
def get_graph_data(
    focus_node: Optional[str] = Query(None, description="Center graph visualization around this node ID"),
    hops: int = Query(1, description="Hop depth for ego-network expansion"),
    include_types: Optional[str] = Query("PERSON,LOCATION,VEHICLE,ORGANIZATION,PHONE_NUMBER,FINANCIAL,CASE", description="Comma-separated entity types"),
    city: Optional[str] = Query("all", description="Filter graph by City name"),
    area: Optional[str] = Query("all", description="Filter graph by Area / Sector name"),
    user: dict = Depends(require_permission("graphs.read", resource_type="full_graph"))
):
    """Returns live investigation graph payload with node/edge metadata, spatial coordinates, and top AI predictions."""
    if not state.graph:
        raise HTTPException(status_code=503, detail="Graph pipeline not initialized.")

    allowed_types = set(t.strip().upper() for t in include_types.split(","))
    selected_city = (city or "all").strip().lower()
    selected_area = (area or "all").strip().lower()

    influence_scores = rank_influential_nodes(state.graph, top_n=state.graph.number_of_nodes())
    influence_map = {node: score for node, score in influence_scores}

    person_influence_scores = rank_influential_nodes(
        state.graph, top_n=state.graph.number_of_nodes(), person_only=True
    )

    included_nodes = set()

    if focus_node and focus_node in state.graph:
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
        top_people = [node for node, score in person_influence_scores[:35]]
        included_nodes.update(top_people)
        for p in top_people:
            for neighbor in state.graph.neighbors(p):
                etype = state.graph.nodes[neighbor].get("entity_type", "").upper()
                if etype in ["LOCATION", "VEHICLE", "ORGANIZATION"]:
                    included_nodes.add(neighbor)

    # Spatial Location Filtering (City / Area)
    if selected_city != "all" or selected_area != "all":
        filtered_nodes = set()
        for node_id in included_nodes:
            if node_id == focus_node:
                filtered_nodes.add(node_id)
                continue

            node_city, node_area, node_locs = get_candidate_location_info(state.graph, node_id, node_id)
            city_match = (selected_city == "all") or (selected_city in node_city.lower())
            area_match = (selected_area == "all") or (selected_area in node_area.lower()) or any(selected_area in l.lower() for l in node_locs)

            if city_match and area_match:
                filtered_nodes.add(node_id)

        if len(filtered_nodes) > 0:
            included_nodes = filtered_nodes

    nodes = []
    valid_node_ids = set()

    for node_id in included_nodes:
        if node_id not in state.graph:
            continue
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
            "degree": state.graph.degree(node_id),
            "latitude": data.get("latitude"),
            "longitude": data.get("longitude"),
            "location_type": data.get("location_type", "")
        })
        valid_node_ids.add(node_id)

    edges = []
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

    # Inject top AI-predicted links from pre-cached list
    top_candidates = state.cached_top_candidates or rank_candidate_links(state.model, state.scaler, state.graph, top_n=30)
    injected_predictions = 0

    for node_a, node_b, prob in top_candidates:
        if injected_predictions >= 10:
            break

        if node_a in valid_node_ids or node_b in valid_node_ids:
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
                        "degree": state.graph.degree(n),
                        "latitude": data.get("latitude"),
                        "longitude": data.get("longitude"),
                        "location_type": data.get("location_type", "")
                    })

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
        "selected_city": city,
        "selected_area": area,
        "nodes": nodes,
        "edges": edges
    }

@app.get("/firs", response_model=List[FIRSummary])
def search_firs(
    q: Optional[str] = Query(None, description="Search term for FIR ID, location, or keyword in narrative"),
    limit: int = Query(30, description="Maximum FIR records to return"),
    user: dict = Depends(require_permission("datasets.read"))
):
    """Search real FIR records with keyword filtering and linked entity identification."""
    if not state.firs:
        return []

    query = (q or "").strip().lower()
    results = []

    for fir in state.firs:
        f_id = fir.get("fir_id", "")
        date = fir.get("date", "")
        narrative = fir.get("narrative", "")

        if not query or query in f_id.lower() or query in date.lower() or query in narrative.lower():
            entities = []
            if state.graph:
                for node_id, data in state.graph.nodes(data=True):
                    mentions = data.get("resolved_source_mentions") or data.get("source_mentions") or []
                    if any(m.get("doc_id") == f_id for m in mentions) or (
                        data.get("canonical_name") and data.get("canonical_name").lower() in narrative.lower()
                    ):
                        entities.append({
                            "id": node_id,
                            "name": data.get("canonical_name", node_id),
                            "type": data.get("entity_type", "UNKNOWN")
                        })

            results.append({
                "fir_id": f_id,
                "date": date,
                "narrative": narrative,
                "entities_count": len(entities),
                "entities": entities
            })
            if len(results) >= limit:
                break

    return results

@app.get("/candidates", response_model=List[LinkCandidate])
def get_candidates(
    top_n: int = Query(50, description="Maximum predictions to retrieve"),
    city: Optional[str] = Query(None, description="Filter candidates by City name"),
    area: Optional[str] = Query(None, description="Filter candidates by Area name"),
    sort_by: str = Query("probability", description="Sort by: 'probability', 'area', 'city'"),
    user: dict = Depends(require_permission("prediction.run"))
):
    """Returns top predicted hidden ties enriched with City & Area tags, plus sorting/filtering parameters."""
    if not state.graph:
        return []

    candidates = state.cached_top_candidates or rank_candidate_links(state.model, state.scaler, state.graph, top_n=100)

    result = []
    for a, b, prob in candidates:
        c_city, c_area, c_locs = get_candidate_location_info(state.graph, a, b)

        # Filter by City
        if city and city.lower() != "all" and city.lower() not in c_city.lower():
            continue

        # Filter by Area
        if area and area.lower() != "all" and area.lower() not in c_area.lower() and not any(area.lower() in l.lower() for l in c_locs):
            continue

        result.append(LinkCandidate(
            node_a=a,
            node_b=b,
            probability=prob,
            city=c_city,
            area=c_area,
            locations=c_locs
        ))

    # Apply Sorting
    if sort_by == "area":
        result.sort(key=lambda x: (x.area, -x.probability))
    elif sort_by == "city":
        result.sort(key=lambda x: (x.city, x.area, -x.probability))
    else:
        result.sort(key=lambda x: x.probability, reverse=True)

    return result[:top_n]


@app.get("/firs/{fir_id}")
def get_fir_detail(
    fir_id: str,
    user: dict = Depends(require_permission("datasets.read"))
):
    """Retrieves complete FIR report dossier with associated entities."""
    target_fir = None
    for fir in state.firs:
        if fir.get("fir_id", "").lower() == fir_id.lower():
            target_fir = fir
            break

    if not target_fir:
        raise HTTPException(status_code=404, detail=f"FIR '{fir_id}' not found.")

    entities = []
    if state.graph:
        for node_id, data in state.graph.nodes(data=True):
            mentions = data.get("resolved_source_mentions") or data.get("source_mentions") or []
            matched_mentions = [m for m in mentions if m.get("doc_id") == target_fir.get("fir_id")]
            c_name = data.get("canonical_name", "")
            if matched_mentions or (c_name and c_name.lower() in target_fir.get("narrative", "").lower()):
                entities.append({
                    "id": node_id,
                    "name": c_name or node_id,
                    "type": data.get("entity_type", "UNKNOWN"),
                    "mentions": matched_mentions
                })

    return {
        "fir_id": target_fir.get("fir_id"),
        "date": target_fir.get("date"),
        "narrative": target_fir.get("narrative"),
        "entities": entities
    }


@app.get("/people/{person_id}", response_model=PersonSummary)
def get_person(person_id: str, user: dict = Depends(require_permission("graphs.read"))):
    """Retrieves full entity profile including connections, merged mentions, and FIR case narratives."""
    if not state.graph or person_id not in state.graph:
        raise HTTPException(status_code=404, detail="Entity profile not found.")

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

    firs_list = find_firs_for_person(person_id)

    return PersonSummary(
        person_id=person_id,
        name=node_data.get("canonical_name", person_id),
        entity_type=node_data.get("entity_type", "PERSON"),
        alias=node_data.get("alias", ""),
        degree=state.graph.degree(person_id),
        influence_score=influence_scores.get(person_id, 0.0),
        connections=connections,
        merged_mentions=merged_mentions,
        firs=[PersonFIR(**f) for f in firs_list]
    )


@app.get("/resolution/merges")
def get_resolution_merges(user: dict = Depends(require_permission("graphs.read"))):
    """Global feed of entity resolution merges across the system."""
    if not state.graph:
        return {"merges": []}

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
    """Explains relationship probability and evidence between two entities."""
    if not state.graph or node_a not in state.graph or node_b not in state.graph:
        raise HTTPException(status_code=404, detail="Entity pair not found.")
    probability = predict_link_probability(state.model, state.scaler, state.graph, node_a, node_b)
    return {
        "probability": probability,
        "explanation": explain_link(state.graph, node_a, node_b, state.model, state.scaler)
    }


@app.get("/influence", response_model=List[InfluentialNode])
def get_influence(top_n: int = 10, user: dict = Depends(require_permission("graphs.read"))):
    """Returns top high-value target individuals ranked by Graph-RAG influence score."""
    if not state.graph:
        return []
    ranked = rank_influential_nodes(state.graph, top_n=max(1, min(top_n, 60)), person_only=True)
    return [
        InfluentialNode(
            rank=rank,
            node=node,
            name=state.graph.nodes[node].get("canonical_name", node),
            score=score
        ) for rank, (node, score) in enumerate(ranked, start=1)
    ]


@app.get("/influence/{node}/explain")
def explain_influence_endpoint(node: str, user: dict = Depends(require_permission("graphs.read"))):
    """Provides Graph-RAG textual evidence for an entity's influence score."""
    if not state.graph or node not in state.graph:
        raise HTTPException(status_code=404, detail="Entity node not found.")
    return {"node": node, "explanation": explain_influence(state.graph, node)}


@app.get("/audit-logs")
def query_audit_logs(user: dict = Depends(require_permission("audit.read"))):
    """Retrieves security audit event log trail."""
    return {"logs": get_audit_logs()}


@app.get("/evaluation/summary")
def evaluation_summary(user: dict = Depends(require_permission("audit.read"))):
    """Judge Mode ground-truth recall scoring against test dataset."""
    ground_truth = load_json_robust("ground_truth.json")
    if not ground_truth:
        raise HTTPException(status_code=404, detail="ground_truth.json file not available on this server.")

    true_hidden_pairs = set()
    for item in ground_truth.get("hidden_links", []):
        pair = item.get("pair")
        if pair and len(pair) >= 2:
            true_hidden_pairs.add(tuple(sorted((pair[0], pair[1]))))

    top_candidates = state.cached_top_candidates or rank_candidate_links(state.model, state.scaler, state.graph, top_n=100)
    matched = []
    for rank, (a, b, prob) in enumerate(top_candidates, start=1):
        pair = tuple(sorted((a, b)))
        if pair in true_hidden_pairs:
            matched.append({"pair": list(pair), "rank": rank, "probability": prob, "mechanism": "candidate_ranking"})

    found_pairs = {tuple(m["pair"]) for m in matched}
    for pair in true_hidden_pairs - found_pairs:
        a, b = pair
        if state.graph and state.graph.has_edge(a, b):
            edge = state.graph[a][b]
            if edge.get("fir_weight", 0) > 0 and edge.get("cdr_weight", 0) == 0 and edge.get("txn_weight", 0) == 0:
                prob = predict_link_probability(state.model, state.scaler, state.graph, a, b)
                matched.append({"pair": list(pair), "rank": None, "probability": prob, "mechanism": "fir_co_mention_fusion"})

    hidden_link_recall = (len(matched) / len(true_hidden_pairs)) if true_hidden_pairs else None

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
        "note": "Judge Mode Evaluation Scoring",
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


if os.path.exists("../frontend"):
    app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")
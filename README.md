# SyndicateScope

**AI-assisted criminal network inference — hackathon prototype**
*NCRB Women Safety Division · SIH26189*

SyndicateScope fuses call records, financial transactions, and free-text FIR
narratives into a single investigation graph, then uses a trained link-prediction
model to surface relationships between people that were never directly recorded
anywhere in the source data. Every prediction, and every "key individual"
ranking, comes with a plain-English, evidence-cited explanation — not just a
bare probability score — so an investigator can see exactly what evidence
produced it before acting on it.

This is the **lite / prototype implementation** built for the hackathon demo. It
follows the same three-stage philosophy (fuse evidence → predict hidden links →
explain and gate access) described in the project's architecture document, but
every component is a working, lightweight, from-scratch implementation rather
than the production-scale system described there. See
[**How this differs from the long-form architecture doc**](#how-this-differs-from-the-long-form-architecture-doc)
below for the specific gaps.

---

## What it does

1. **Ingests** synthetic police data — people, FIRs, call detail records (CDRs),
   financial transactions, vehicles, organizations, and locations.
2. **Extracts entities** from unstructured FIR narrative text (people, phone
   numbers, vehicle plates, locations, organizations) using spaCy NER, regex,
   and a known-names gazetteer.
3. **Resolves duplicate identities** — the same person appearing as slightly
   different name spellings/aliases across records gets merged into one
   canonical profile, using string-similarity matching.
4. **Builds a fused graph** (NetworkX) where CDRs, transactions, FIR
   co-mentions, vehicle ownership, org membership, and cell-tower pings all
   become edges between entities.
5. **Trains a link-prediction model** (logistic regression over graph
   features — shared neighbours, Jaccard similarity, shortest path,
   co-location events, Adamic-Adar, resource allocation, FIR co-mention
   weight) to score the probability of a hidden relationship between any two
   people who have never directly interacted.
6. **Ranks influential individuals** using a topology + evidence-diversity
   score (degree/betweenness/PageRank centrality, gated by how many
   independent evidence channels — CDR, transaction, FIR — corroborate that
   person).
7. **Explains every prediction and ranking** in plain language, citing the
   specific shared contacts, FIR IDs, co-location events, or evidence
   channels that produced the score.
8. **Gates all of the above behind role-based access control** (Admin /
   Analyst / Viewer), with every access attempt written to an append-only
   audit log.
9. **Serves everything through a FastAPI backend** and a single-page,
   dark-themed investigation dashboard — a force-directed relationship graph
   (vis-network) plus a real geographic **Map View** (Leaflet + OpenStreetMap)
   showing planted locations at their true Bengaluru coordinates, alongside
   profile panels and candidate/influence feeds.

---

## Architecture

```
CSV / JSON data
      │
      ▼
 ingest.py  ──────────────────────────────────────────────┐
      │                                                    │
      ▼                                                    │
 ner/ner.py            (spaCy + regex + gazetteer)         │
      │                                                    │
      ▼                                                    │
 resolution/resolver.py (difflib similarity → canonical    │
      │                    entity nodes)                   │
      ▼                                                    │
 graph/graph.py         (NetworkX graph: PERSON, LOCATION, │
      │                   ORGANIZATION, VEHICLE nodes;     │
      │                   CDR / TRANSACTION / FIR_CO_MENTION /
      │                   REGISTERED_OWNER / ORG_MEMBERSHIP /
      │                   TOWER_PING edges; LOCATION nodes  │
      │                   carry real lat/long)              │
      ▼                                                    │
 prediction/link_prediction.py                             │
      │   (feature extraction → train/test split →         │
      │    logistic regression → probability scoring)      │
      ▼                                                    │
 analytics/influence.py (centrality × evidence-diversity   │
      │                   scoring)                          │
      ▼                                                    │
 rag/graph_rag.py       (evidence retrieval → plain-English │
      │                   explanation generation)           │
      ▼                                                    │
 security.py            (RBAC + audit logging, wraps every  │
      │                   endpoint below)                   │
      ▼                                                    │
 api.py                 (FastAPI: /graph/data — now incl.   │
                          lat/long — /people/{id},           │
                          /candidates, /influence, /link/…,  │
                          /audit-logs, /evaluation/summary)  │
      │
      ▼
 frontend/index.html    (vis-network relationship graph +
                          Leaflet map view, persona login,
                          candidate feed, influence panel,
                          audit log viewer)
```

---

## Repository layout

```
NexusGuard-lite/
├── Backend/
│   ├── main.py                    # standalone CLI pipeline run + console report
│   ├── api.py                     # FastAPI app — all HTTP endpoints
│   ├── security.py                # RBAC roles, permission checks, audit log
│   ├── ingest.py                  # CSV/JSON loaders
│   ├── evaluate_ground_truth.py   # offline scoring script (uses ground_truth.json)
│   ├── audit_logs.jsonl           # append-only audit log (generated at runtime)
│   ├── ner/ner.py                 # entity extraction from FIR text
│   ├── resolution/resolver.py     # duplicate-identity merging
│   ├── graph/graph.py             # graph construction (all edge types)
│   ├── prediction/link_prediction.py  # features, training, scoring, ranking
│   ├── analytics/influence.py     # influence/centrality scoring
│   ├── rag/graph_rag.py           # evidence-grounded explanation generation
│   └── data/
│       ├── generate_data.py       # synthetic dataset generator (Faker, seed=42)
│       ├── people.csv / cdrs.csv / transactions.csv / vehicles.csv /
│       │   organizations.csv / locations.csv / firs.json
│       ├── ground_truth.json      # answer key — verification only, never fed to the pipeline
│       └── README2.md             # full dataset schema + ground-truth mapping
├── frontend/
│   └── index.html                 # single-page dashboard (Tailwind + vis-network + Leaflet)
├── requirement.txt
└── README.md                      # this file
```

---

## Tech stack

| Layer | Technology |
|---|---|
| Graph engine | NetworkX |
| NLP / NER | spaCy (`en_core_web_sm`) + regex + gazetteer matching |
| Entity resolution | `difflib.SequenceMatcher` string similarity |
| Link prediction model | scikit-learn `LogisticRegression` (`StandardScaler`-normalized graph features) |
| Backend API | FastAPI + Uvicorn |
| Synthetic data | Faker (`en_IN` locale, fixed seed) |
| Frontend — relationship graph | Static HTML/JS, Tailwind CSS (CDN), vis-network (CDN) |
| Frontend — geographic map | Leaflet.js + OpenStreetMap tiles (CDN, no API key) |
| Auth (demo) | Header-based persona selection (`X-User-Id`), no real credential verification |
| Audit trail | Append-only JSON Lines file (`audit_logs.jsonl`) |

---

## Getting started

### 1. Install dependencies

```bash
pip install -r requirement.txt
python -m spacy download en_core_web_sm
```

### 2. (Optional) Regenerate the synthetic dataset

```bash
cd Backend/data
python generate_data.py
```

This is deterministic (seed 42) — the committed CSVs/JSON were produced by this
script and don't need to be regenerated unless you want to change the dataset
size or planted signal.

### 3a. Run the standalone pipeline (console report)

```bash
cd Backend
python main.py
```

Prints: entity-resolution stats, sample graph node, link-prediction feature
vectors, model accuracy/precision@10, top hidden-link candidates, top
influential nodes, and plain-English explanations for both.

### 3b. Run the API + dashboard

```bash
cd Backend
uvicorn api:app --reload
```

Then open `http://127.0.0.1:8000/` — the FastAPI app serves the frontend
directly (`app.mount("/", StaticFiles(directory="../frontend", html=True))`).
On startup, the API rebuilds the full pipeline (ingest → NER → resolve → graph
→ train) once and keeps it in memory (`PipelineState`).

### 4. (Optional) Offline ground-truth scoring

```bash
cd Backend
python evaluate_ground_truth.py
```

Scores the trained model's hidden-link recall and influencer recall against
`data/ground_truth.json`. This file is **never** used anywhere in the
ingestion/training path — only for after-the-fact evaluation, exactly as
`/evaluation/summary` does inside the API.

---

## Dashboard features

- **Relationship graph** (default view) — force-directed vis-network canvas.
  Filter by entity type, focus/expand around a node, click any entity or edge
  for a full evidence breakdown, and inject the top AI-predicted hidden links
  as dashed edges via "Run AI Model".
- **Map View** — toggle button next to the entity-type filters swaps the
  canvas for a real OpenStreetMap view centered on Bengaluru. Every
  `LOCATION` node (cell towers, warehouses, markets, lodges, offices, etc.)
  is plotted as a marker at its **true lat/long**, sized by connection count.
  Clicking a marker opens the same inspector panel used in the graph view.
  Only `LOCATION` nodes are geocoded in this dataset — people, vehicles, and
  organizations still live in the relationship graph only.
- **Intelligence Inspector** — evidence detail for whatever is selected
  (node or edge), plus "Explain Ranking" for influence scores.
- **Targets panel** — top-N key individuals, click to focus the graph on
  them.
- **Merges panel** — every entity-resolution merge, so an investigator can
  audit which raw name/alias mentions were folded into which canonical
  profile.
- **Audit panel** (Admin only) — live view of `audit_logs.jsonl`.
- **Judge Mode panel** (Admin only) — runs `/evaluation/summary` against
  `ground_truth.json` live in the browser.

---

## API reference

All endpoints except `/health` require an `X-User-Id` header
(`u_admin` / `u_analyst` / `u_viewer` — see [RBAC](#rbac--audit-logging)
below) and are individually permission-gated.

| Endpoint | Permission required | Purpose |
|---|---|---|
| `GET /health` | none | Node/edge count sanity check |
| `GET /graph/data` | `graphs.read` | Progressive-disclosure graph payload for the canvas (ego-network around a focus node, or default top-influencer view); `LOCATION` nodes include `latitude`/`longitude`/`location_type` for the Map View |
| `GET /people/{person_id}` | `graphs.read` | Full profile: identity, influence score, every connection with evidence weights, every raw NER mention merged into this node |
| `GET /resolution/merges` | `graphs.read` | Global feed of every entity-resolution merge (which raw mentions became which canonical node) |
| `GET /link/{node_a}/{node_b}` | `prediction.run` | Hidden-link probability + plain-English explanation for a specific pair |
| `GET /candidates?top_n=` | `prediction.run` | Top-N ranked unobserved PERSON–PERSON pairs by predicted probability |
| `GET /influence?top_n=` | `graphs.read` | Top-N ranked key individuals (person-only) |
| `GET /influence/{node}/explain` | `graphs.read` | Evidence-channel explanation for one influence ranking |
| `GET /audit-logs` | `audit.read` | Full audit trail |
| `GET /evaluation/summary` | `audit.read` | Judge/demo-mode scoring against `ground_truth.json` |

---

## RBAC & audit logging

Three mock roles, selected via the frontend's login screen (no password —
persona-selection only, for demo purposes):

| Role | Permissions |
|---|---|
| **Admin** (DSP) | Everything — users, roles, datasets, graph read/write, prediction, audit logs |
| **Analyst** (Investigator) | `datasets.read`, `graphs.read`, `prediction.run` |
| **Viewer** (Beat Officer) | `datasets.read`, `graphs.read` only |

`security.require_permission(...)` is a FastAPI dependency wrapping every
protected route: it resolves the role from the `X-User-Id` header, checks the
permission against `ROLES_PERMISSIONS`, checks a resource-level rule (e.g.
`SEC_`-prefixed resource IDs are Admin-only), and **always** writes a
structured event — user, role, action, resource, allow/deny, request ID — to
`audit_logs.jsonl`, whether access was granted or denied.

---

## The link-prediction model, concretely

`prediction/link_prediction.py` extracts a 7-dimensional feature vector for
any pair of PERSON nodes:

1. Shared-neighbour ratio
2. Jaccard similarity of neighbourhoods
3. Inverse shortest-path length
4. Co-location event ratio (both parties active at the same location within a
   60-minute window, parsed from timestamped edge data)
5. Adamic-Adar index
6. Resource-allocation index
7. FIR co-mention weight (pulled from `graph.graph["fir_co_mentions"]`, which
   is preserved independently of the edge list so it survives the
   train/test edge-removal split)

Training (`create_training_data` → `train_link_prediction_model`):
existing PERSON–PERSON edges are 80/20 split; the held-out 20% is removed from
a copy of the graph; an equal number of random non-edges is sampled as
negatives; features are standardized (`StandardScaler`) and fed to a
class-balanced `LogisticRegression`. `rank_candidate_links` then scores every
remaining unobserved PERSON–PERSON pair and returns the top N.

---

## The influence-ranking model, concretely

`analytics/influence.py` combines:

- **Topology** — 45% degree centrality + 35% betweenness + 20% PageRank
- **Evidence gating** — that topology score is scaled by
  `0.4 + 0.6 × (channels touched / 3)`, so a node that is only busy in one
  channel (e.g. pure CDR noise) is capped at 40% of its raw topology score,
  while a node corroborated across CDR + transaction + FIR keeps 100%
- **FIR evidence** — weighted independently at 35% of the final score (FIR
  co-mention is treated as the hardest signal to fabricate through random
  noise)
- **Multi-source bonus** — a further 10% for touching multiple evidence
  channels

This is specifically designed to catch the dataset's "central only when
combined" planted signal: three key individuals who look unremarkable
(degree 5) through any single evidence channel alone, but jump to a combined
neighbourhood of 15 once CDRs, transactions, and FIR mentions are fused (see
`data/README2.md`).

---

## Dataset

100% synthetic, generated with Faker (`en_IN` locale, seed 42) — nothing
refers to a real person, vehicle, or organization. 60 people (24 planted
network + 36 red herrings), 35 FIRs, 300 CDRs, 150 transactions, 15 vehicles,
5 organizations, 10 locations (each with real Bengaluru lat/long, used by the
Map View). Full schema, noise ratios, and exactly how the three planted
"hidden links" and three planted "key influencers" map to the raw data are
documented in `Backend/data/README2.md`.

---

## How this differs from the long-form architecture doc

The project's *Extended Technical Architecture & Deployment Guide* describes a
larger target system. This prototype implements the same three-tier philosophy
end-to-end but with different, simpler components at every tier. For an honest
demo, know these gaps:

| Architecture doc describes | This prototype actually does |
|---|---|
| Tier 1: transformer cross-encoder (BERT-like) entity resolution, similarity threshold 0.95 | `difflib.SequenceMatcher` string-similarity matching, threshold 0.75–0.80 |
| Tier 2: Temporal Graph Neural Network (GRU-based message passing, `h_v(t)` formula) | Logistic regression over 7 hand-engineered static graph/temporal features (no neural network, no learned embeddings) |
| Tier 3: Graph-RAG feeding an LLM to generate justifications | Template-based plain-English generation directly from retrieved evidence (`rag/graph_rag.py`) — no LLM call in the explanation path |
| Zero-trust architecture, cryptographic identity per request | Header-based persona selection (`X-User-Id`), no password or token verification — RBAC permission logic itself is real and centrally enforced |
| Blockchain-inspired SHA-256 hash-chained, tamper-evident audit log | Plain append-only JSON Lines log (`audit_logs.jsonl`) — complete and structured, but not hash-chained, so a deleted line would not be cryptographically detectable |

None of this is a criticism of the demo — hackathon prototypes are expected to
prove the *concept* with lightweight implementations. But when presenting,
be precise about which tier is "real ML on real graph features" (link
prediction, influence ranking, RBAC/audit logging) versus which is a
simplified stand-in for a heavier component described in the architecture
document (entity resolution, the prediction model's internals, and the
explanation engine).

---

## Known limitations

- `main.py`'s console report calls `evaluate_model(...)` twice — once
  unlabelled ("Accuracy") and once with `top_k=10` ("Precision@10") — both
  calls are functionally identical (`evaluate_model` always computes a
  Precision@k over the top-`top_k` ranked candidates), so the two printed
  numbers are the same metric under two names.
- The RBAC layer has no real authentication — any client can claim any
  `X-User-Id` value.
- The audit log is append-only in practice but not cryptographically
  tamper-evident.
- Entity resolution is O(n) similarity comparisons against every existing
  canonical node per new mention — fine at this dataset's scale (dozens of
  entities), not designed for production volume.
- Map View only geocodes `LOCATION` nodes. People, vehicles, and
  organizations have no coordinates in the dataset and remain graph-only;
  pinning a person's most recent tower ping on the map is a possible future
  extension.
- Map View's OpenStreetMap tiles are served from the public tile server,
  which is fine for a demo but isn't licensed for production-scale traffic.

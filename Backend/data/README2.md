# NexusGuard — Synthetic Demo Dataset

All data is 100% fictional, generated with Python + Faker (`en_IN` locale) using a fixed
seed (42), so re-running `generate_data.py` reproduces the same dataset. Locations use a
Bengaluru lat/long bounding box for plausibility. Nothing here refers to a real person,
vehicle, or organization.

## Files

| File | Rows | Description |
|---|---|---|
| `people.csv` | 60 | Entities. `person_id` starting `N` = planted network (24), `R` = red herring (36, 60%). |
| `vehicles.csv` | 15 | Vehicles linked to an owner `person_id`. |
| `locations.csv` | 10 | Named locations with lat/long (Bengaluru area) and a `type`. |
| `organizations.csv` | 5 orgs, one row per member | Shell companies/fronts, each with 1–3 linked people and a role. |
| `firs.json` | 35 | Free-text police report narratives, 3–5 sentences, mentioning 2–4 entities each. |
| `cdrs.csv` | 300 | Call detail records over Jun 1 – Aug 31, 2026. |
| `transactions.csv` | 150 | Financial transactions over the same window. |
| `ground_truth.json` | — | **Answer key only — exclude from your pipeline's input.** Used to score your system's output afterwards. |

## Schemas

**people.csv**: `person_id, name, age, phone, address, alias` (alias is blank except for 3 people: N07/"Bunty", N12/"Chintu", N16/"Guddu").

**vehicles.csv**: `vehicle_id, plate_number, make_model, owner_person_id`

**locations.csv**: `location_id, name, type, latitude, longitude`

**organizations.csv**: `org_id, name, type, person_id, role` (role is `owner` or `associate`; an org with 3 members appears as 3 rows sharing the same `org_id`).

**firs.json**: list of `{fir_id, date, narrative}`. Narratives deliberately vary sentence structure/phrasing and sometimes use an alias instead of a full name, so entity extraction isn't trivial pattern matching. Entity mentions are **not** pre-labelled in this file — that's what your NER/extraction step is for.

**cdrs.csv**: `caller_id, receiver_id, timestamp, duration_seconds, tower_location` (`tower_location` refers to a `location_id`).

**transactions.csv**: `sender_id, receiver_id, amount, timestamp, mode` (`mode` ∈ cash / UPI / bank_transfer).

**ground_truth.json**: verification-only mapping of the planted signal (see below). Keep this out of your ingestion pipeline; use it only to check precision/recall of what your system surfaces.

## Noise level

- 36 of 60 people (60%) are pure red herrings with no connection to the planted network.
- 185 of 300 CDRs (62%) and 90 of 150 transactions (60%) are between two red-herring people, with no link to the planted network.
- Timestamps are randomly distributed across the 3-month window but engineered signal (an influencer's contact-group calls/transactions, tower pings, chain hops) is clustered into ±4-day bursts around a randomly chosen day, so a time-based view will show believable bursts rather than uniform noise.

## How the ground truth maps to the raw data

### 1. Three "key influencers" (N01, N02, N03)
Each has **three disjoint contact groups of 5 people**, one per channel:

| Influencer | CDR-only contacts | Transaction-only contacts | FIR-mention-only contacts |
|---|---|---|---|
| N01 | N04–N08 | N09–N13 | N14–N18 |
| N02 | N19–N23 | N04–N08 | N09–N13 |
| N03 | N14–N18 | N19–N23 | N04–N08 |

Looked at through any **one** channel alone, each influencer has degree 5 — unremarkable.
Only once you fuse CDRs + transactions + FIR co-mentions into a single graph does each
influencer's combined neighbourhood jump to 15 people, which is where their degree/
betweenness/PageRank should spike. This is the exact "central only when combined" signal
your Tier 2-Extended influence scoring (FR-20/FR-21) is meant to catch.

### 2. Three "hidden links" (never in direct contact)
1. **N24 ↔ N19 — shared financial intermediary.** N24 pays N08, and N08 separately pays
   N19 a few days later (see `transactions.csv`, and the corresponding entry in
   `ground_truth.json → hidden_links[0]`). N24 and N19 never appear in the same CDR or
   transaction row.
2. **N22 ↔ N16 — shared cell tower ping.** Both independently call other people from
   `L01` (Whitefield Junction Tower) within minutes of each other, more than once — a
   physical co-location signal — but never call one another. See `cdrs.csv` rows with
   `tower_location = L01`.
3. **N05 ↔ N15 — FIR co-mention only.** Both are named in the same FIR narrative
   (`FIR-001`… check `ground_truth.json → fir_mention_ground_truth` for the exact ID) but
   have zero CDR or transaction contact anywhere in the dataset.

### 3. Multi-hop financial chains
`ground_truth.json → multi_hop_chains` lists two A→B→C sequences you should be able to
trace through `transactions.csv` by following sender/receiver chains in time order —
one of these chains is the same one underlying hidden link #1 above.

## Suggested verification flow

1. Run your ingestion → NER → entity resolution → graph build **without** touching
   `ground_truth.json`.
2. Compute influence scores and pull your top-N "key individuals" — check whether
   N01/N02/N03 appear.
3. Run your link-prediction / anomaly step and check whether it flags the three pairs
   in `hidden_links`.
4. Only then open `ground_truth.json` to score precision/recall for your demo slide.

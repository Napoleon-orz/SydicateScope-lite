#!/usr/bin/env python3
"""
NexusGuard synthetic demo dataset generator.
Everything is fictional (Faker-generated). Deterministic via fixed seed.
"""

import csv
import json
import random
from datetime import datetime, timedelta

from faker import Faker

SEED = 42
random.seed(SEED)
fake = Faker("en_IN")
Faker.seed(SEED)

OUT = "/home/claude/nexusguard_data"

START_DATE = datetime(2026, 6, 1)
END_DATE = datetime(2026, 8, 31)
TOTAL_DAYS = (END_DATE - START_DATE).days

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rand_timestamp(start=START_DATE, end=END_DATE, cluster=None):
    """Return a random datetime between start/end. If `cluster` (a datetime)
    is given, bias towards a +/- 4 day window around it for realistic bursts."""
    if cluster is not None:
        lo = max(start, cluster - timedelta(days=4))
        hi = min(end, cluster + timedelta(days=4))
        delta = (hi - lo).total_seconds()
        return lo + timedelta(seconds=random.uniform(0, delta))
    delta = (end - start).total_seconds()
    return start + timedelta(seconds=random.uniform(0, delta))


def fmt(dt):
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def indian_plate():
    state = random.choice(["KA", "MH", "DL", "TN", "AP", "TS", "KL"])
    return f"{state}{random.randint(1,60):02d}{random.choice('ABCDEFGHJKLMNPQR')}{random.choice('ABCDEFGHJKLMNPQR')}{random.randint(1000,9999)}"


def phone():
    return f"9{random.randint(100000000,999999999)}"


# ---------------------------------------------------------------------------
# 1. PEOPLE  (60 total: N01-N24 = planted network, R01-R36 = red herring, 60%)
# ---------------------------------------------------------------------------

NETWORK_N = 24
REDHERRING_N = 36  # 60% of 60

people = []
person_ids = []

alias_pool = ["Bunty", "Chintu", "Guddu"]
alias_assignees = []  # filled below once IDs exist

for i in range(1, NETWORK_N + 1):
    pid = f"N{i:02d}"
    person_ids.append(pid)
    people.append({
        "person_id": pid,
        "name": fake.name(),
        "age": random.randint(19, 62),
        "phone": phone(),
        "address": fake.address().replace("\n", ", "),
        "alias": "",
        "group": "network",
    })

for i in range(1, REDHERRING_N + 1):
    pid = f"R{i:02d}"
    person_ids.append(pid)
    people.append({
        "person_id": pid,
        "name": fake.name(),
        "age": random.randint(18, 70),
        "phone": phone(),
        "address": fake.address().replace("\n", ", "),
        "alias": "",
        "group": "red_herring",
    })

# assign 3 aliases to network people (not the 3 key influencers themselves,
# to keep influencer-spotting non-trivial)
alias_targets = ["N07", "N12", "N16"]
for pid, alias in zip(alias_targets, alias_pool):
    for p in people:
        if p["person_id"] == pid:
            p["alias"] = alias

by_id = {p["person_id"]: p for p in people}

with open(f"{OUT}/people.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["person_id", "name", "age", "phone", "address", "alias"])
    w.writeheader()
    for p in people:
        w.writerow({k: p[k] for k in ["person_id", "name", "age", "phone", "address", "alias"]})

# ---------------------------------------------------------------------------
# 2. VEHICLES (~15)
# ---------------------------------------------------------------------------

vehicles = []
makes = ["Maruti Swift", "Hero Splendor", "Bajaj Pulsar", "Toyota Innova", "Honda Activa",
         "Mahindra Bolero", "Tata Ace", "Royal Enfield Classic", "Hyundai i20", "Ford EcoSport"]

vehicle_owner_pool = [f"N{i:02d}" for i in [1, 2, 3, 6, 9, 11, 14, 17, 19, 22]] + \
                     [f"R{i:02d}" for i in [3, 8, 15, 21, 30]]

for i, owner in enumerate(vehicle_owner_pool, start=1):
    vehicles.append({
        "vehicle_id": f"V{i:02d}",
        "plate_number": indian_plate(),
        "make_model": random.choice(makes),
        "owner_person_id": owner,
    })

with open(f"{OUT}/vehicles.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["vehicle_id", "plate_number", "make_model", "owner_person_id"])
    w.writeheader()
    w.writerows(vehicles)

# ---------------------------------------------------------------------------
# 3. LOCATIONS (~10, Bengaluru bounding box)
# ---------------------------------------------------------------------------

loc_names = [
    ("Whitefield Junction Tower", "cell_tower"),
    ("Old Silk Board Warehouse", "warehouse"),
    ("KR Market Wholesale Complex", "market"),
    ("Yeshwantpur Freight Yard", "transport_hub"),
    ("Electronic City Phase 2 Lodge", "lodge"),
    ("Hebbal Flyover CCTV Point", "surveillance_point"),
    ("Majestic Bus Terminus", "transport_hub"),
    ("Indiranagar 100ft Road Office", "office"),
    ("Bannerghatta Road Godown", "warehouse"),
    ("Malleshwaram Residency", "residence"),
]

BLR_LAT_RANGE = (12.85, 13.05)
BLR_LON_RANGE = (77.45, 77.75)

locations = []
for i, (name, ltype) in enumerate(loc_names, start=1):
    locations.append({
        "location_id": f"L{i:02d}",
        "name": name,
        "type": ltype,
        "latitude": round(random.uniform(*BLR_LAT_RANGE), 6),
        "longitude": round(random.uniform(*BLR_LON_RANGE), 6),
    })

with open(f"{OUT}/locations.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["location_id", "name", "type", "latitude", "longitude"])
    w.writeheader()
    w.writerows(locations)

loc_ids = [l["location_id"] for l in locations]
loc_by_id = {l["location_id"]: l for l in locations}

# ---------------------------------------------------------------------------
# 4. ORGANIZATIONS (~5 shell companies)
# ---------------------------------------------------------------------------

org_defs = [
    ("Sundar Freight & Logistics Pvt Ltd", "logistics_shell"),
    ("Vaibhav Agro Traders", "trading_shell"),
    ("Prime Estates Consultancy", "real_estate_shell"),
    ("Om Sai Finance Associates", "informal_finance"),
    ("Nova Digital Solutions LLP", "shell_it_services"),
]

org_people_map = {
    1: [("N02", "owner"), ("N09", "associate")],
    2: [("N03", "owner"), ("N19", "associate"), ("N21", "associate")],
    3: [("N01", "owner"), ("N14", "associate")],
    4: [("N08", "owner"), ("N24", "associate")],
    5: [("N17", "owner"), ("N04", "associate"), ("N05", "associate")],
}

organizations = []
for idx, (name, otype) in enumerate(org_defs, start=1):
    members = org_people_map[idx]
    organizations.append({
        "org_id": f"O{idx}",
        "name": name,
        "type": otype,
        "members": [{"person_id": pid, "role": role} for pid, role in members],
    })

with open(f"{OUT}/organizations.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["org_id", "name", "type", "person_id", "role"])
    for org in organizations:
        for m in org["members"]:
            w.writerow([org["org_id"], org["name"], org["type"], m["person_id"], m["role"]])

print("Static entity files written.")

# ---------------------------------------------------------------------------
# GROUND-TRUTH DESIGN (defined before FIR/CDR/TXN generation so we can
# engineer the right signals into the raw data)
# ---------------------------------------------------------------------------

# Three key influencers. Each has THREE DISJOINT contact groups, one per
# channel (CDR / transaction / FIR-mention). Seen through any single
# channel they look like a peripheral node with degree 5. Only once all
# three channels are fused does their true combined degree (15) and
# betweenness stand out.
INFLUENCERS = {
    "N01": {
        "cdr":  ["N04", "N05", "N06", "N07", "N08"],
        "txn":  ["N09", "N10", "N11", "N12", "N13"],
        "fir":  ["N14", "N15", "N16", "N17", "N18"],
    },
    "N02": {
        "cdr":  ["N19", "N20", "N21", "N22", "N23"],
        "txn":  ["N04", "N05", "N06", "N07", "N08"],
        "fir":  ["N09", "N10", "N11", "N12", "N13"],
    },
    "N03": {
        "cdr":  ["N14", "N15", "N16", "N17", "N18"],
        "txn":  ["N19", "N20", "N21", "N22", "N23"],
        "fir":  ["N04", "N05", "N06", "N07", "N08"],
    },
}

# Hidden links: pairs that NEVER call or transact with each other directly,
# connected only through an indirect signal.
HIDDEN_LINKS = [
    {
        "pair": ["N24", "N19"],
        "mechanism": "shared_financial_intermediary",
        "detail": "N24 pays intermediary N08, who separately pays N19, within a short window. "
                   "N24 and N19 never appear together in any CDR or transaction record.",
        "intermediary": "N08",
    },
    {
        "pair": ["N22", "N16"],
        "mechanism": "shared_cdr_tower_ping",
        "detail": "N22 and N16 each place unrelated calls from the same cell tower "
                   "(Whitefield Junction Tower, L01) within minutes of each other on the same day, "
                   "suggesting physical co-location, but never call one another directly.",
        "tower": "L01",
    },
    {
        "pair": ["N05", "N15"],
        "mechanism": "fir_co_mention_only",
        "detail": "N05 and N15 are named together in the same FIR narrative (seen together at a "
                   "location) but have zero direct CDR or transaction contact anywhere in the dataset.",
    },
]

# Independent multi-hop financial trail (not itself a "hidden link", just a
# traceable A->B->C chain investigators should be able to follow).
MULTIHOP_CHAINS = [
    {"chain": ["N11", "N17", "N21"], "note": "Sequential transfers across a 9-day window."},
    {"chain": ["N24", "N08", "N19"], "note": "Same chain that constitutes the hidden financial-intermediary link above."},
]

# ---------------------------------------------------------------------------
# 5. FIRs / POLICE REPORTS (35 narratives, varied phrasing)
# ---------------------------------------------------------------------------

fir_templates = [
    "Complainant reported that {p1}, believed to frequent {loc}, was seen in the company of "
    "{p2} on the evening of the incident. A {vehicle} matching the description given was "
    "noted parked nearby. Investigators have opened a preliminary inquiry.",

    "During a routine patrol near {loc}, officers noted the presence of a vehicle bearing "
    "registration {vehicle}, reportedly linked to {p1}. Local sources named {p2} as a "
    "frequent visitor to the same address. No arrests have been made at this stage.",

    "A tip-off led investigators to {loc}, where {p1} — known locally as '{alias}' — was "
    "allegedly conducting a meeting with an unidentified associate. {p2} was named as a "
    "possible facilitator based on prior surveillance notes.",

    "The complainant, a shopkeeper near {loc}, stated that two men, one of whom matched the "
    "description of {p1}, arrived in a {vehicle} and were later joined by {p2}. The matter "
    "has been forwarded to the crime branch for further verification.",

    "Surveillance staff flagged a gathering at {loc} attended by {p1} and {p2}; a third "
    "individual, thought to be connected through {p2}, was not conclusively identified. "
    "The case has been logged for follow-up.",

    "{p1} was questioned in connection with an altercation near {loc}. Witnesses separately "
    "placed {p2} at the same location earlier that day, though the two were not seen "
    "interacting directly. Statements have been recorded.",

    "A confidential informant reported that {p1}, associated with the vehicle {vehicle}, has "
    "been operating out of {loc} for the past several weeks, occasionally in the presence of "
    "{p2}. The report has been marked for intelligence follow-up.",

    "Field officers observed {p1} exiting {loc} shortly before midnight; {p2} was seen "
    "entering the same premises roughly an hour later. It is unclear whether the two visits "
    "were connected. Further monitoring has been recommended.",

    "Residents near {loc} lodged a complaint against unidentified persons using the premises "
    "for suspicious late-night activity. CCTV stills reportedly resemble {p1}; a person "
    "answering to the alias '{alias}' was also mentioned by a witness.",

    "{p1} and {p2} were both named in witness statements taken after a disturbance near "
    "{loc}, though investigating officers noted no direct contact between the two was "
    "observed at the scene.",
]

random.shuffle(fir_templates)

# people who have an alias, for template slots that need one
alias_people = {p["person_id"]: p["alias"] for p in people if p["alias"]}

# guarantee the FIR co-mention hidden link (N05 & N15, no direct contact)
firs = []
fir_id = 1


def make_fir(p1, p2=None, need_loc=True, need_vehicle=False, alias_pid=None):
    global fir_id
    tmpl = random.choice(fir_templates)
    loc = random.choice(loc_names)[0] if need_loc else random.choice(loc_names)[0]
    veh = random.choice(vehicles)["plate_number"]
    alias = alias_people.get(alias_pid, random.choice(alias_pool)) if "{alias}" in tmpl else ""
    name1 = by_id[p1]["name"]
    name2 = by_id[p2]["name"] if p2 else fake.name()
    text = tmpl.format(p1=name1, p2=name2, loc=loc, vehicle=veh, alias=alias)
    dt = rand_timestamp()
    firs.append({
        "fir_id": f"FIR-{fir_id:03d}",
        "date": dt.strftime("%Y-%m-%d"),
        "narrative": text,
        # NOTE: mentioned_entities below is investigator/ground-truth metadata,
        # kept ONLY in ground_truth.json — the firs.json file given to the
        # extraction pipeline contains narrative text only, as a real system
        # would have to extract these itself.
    })
    fir_id += 1
    return firs[-1]["fir_id"]


fir_mention_log = []  # (fir_id, [person_ids mentioned]) for ground truth

# 1) The engineered FIR co-mention hidden link
fid = make_fir("N05", "N15", alias_pid="N05")
fir_mention_log.append((fid, ["N05", "N15"]))

# 2) FIR-channel contact groups for the three influencers (5 FIRs each,
# one FIR per contact -> 15 FIRs)
for inf, groups in INFLUENCERS.items():
    for contact in groups["fir"]:
        fid = make_fir(inf, contact, alias_pid=inf)
        fir_mention_log.append((fid, [inf, contact]))

# 3) Remaining FIRs (35 - 1 - 15 = 19) as general noise/coverage: mix of
# network and red-herring people, some with only one identifiable person
# plus vehicle/location, to keep extraction non-trivial.
remaining = 35 - len(firs)
noise_pool = [f"R{i:02d}" for i in range(1, REDHERRING_N + 1)]
for _ in range(remaining):
    p1 = random.choice(person_ids)
    p2 = random.choice([x for x in person_ids if x != p1])
    fid = make_fir(p1, p2, alias_pid=p1)
    fir_mention_log.append((fid, [p1, p2]))

with open(f"{OUT}/firs.json", "w") as f:
    json.dump(firs, f, indent=2)

print(f"{len(firs)} FIRs written.")

# ---------------------------------------------------------------------------
# 6. CALL DETAIL RECORDS (~300, 3-month window, 60% red-herring)
# ---------------------------------------------------------------------------

cdrs = []


def add_cdr(caller, receiver, ts=None, tower=None, cluster=None):
    ts = ts or rand_timestamp(cluster=cluster)
    cdrs.append({
        "caller_id": caller,
        "receiver_id": receiver,
        "timestamp": fmt(ts),
        "duration_seconds": random.randint(15, 1400),
        "tower_location": tower or random.choice(loc_ids),
    })
    return ts


# --- engineered signal: influencer CDR contact groups (5 contacts x 3
# influencers = 15 relationships, ~4 calls each spread over the window) ---
for inf, groups in INFLUENCERS.items():
    for contact in groups["cdr"]:
        cluster_day = rand_timestamp()
        for _ in range(random.randint(3, 5)):
            add_cdr(inf, contact, cluster=cluster_day)

# --- engineered signal: shared-tower hidden link (N22 <-> N16, never
# calling each other, but pinging the same tower minutes apart) ---
shared_day = rand_timestamp()
base_time = shared_day.replace(hour=15, minute=random.randint(0, 30), second=0)
add_cdr("N22", "N23", ts=base_time - timedelta(minutes=random.randint(1, 6)), tower="L01")
add_cdr("N16", "N06", ts=base_time + timedelta(minutes=random.randint(1, 8)), tower="L01")
# a couple more corroborating pings on other days, still same tower, still no direct contact
for _ in range(2):
    d = rand_timestamp()
    add_cdr("N22", random.choice(["N23", "N09"]), ts=d.replace(hour=random.choice([9, 18])), tower="L01")
    add_cdr("N16", random.choice(["N06", "N12"]), ts=d.replace(hour=random.choice([9, 18])) + timedelta(minutes=random.randint(2, 10)), tower="L01")

# --- general (unflagged) chatter among network members, to avoid an
# artificially sparse "clean" network graph ---
network_people = [f"N{i:02d}" for i in range(1, NETWORK_N + 1)]
for _ in range(40):
    a, b = random.sample(network_people, 2)
    add_cdr(a, b)

# --- red herring calls: pad to ~300 total, at least 60% (180) purely
# among red-herring people with zero connection to the planted network ---
target_total = 300
pure_red_herring_target = int(target_total * 0.6)  # 180

while sum(1 for c in cdrs if c["caller_id"].startswith("R") and c["receiver_id"].startswith("R")) < pure_red_herring_target:
    a, b = random.sample([f"R{i:02d}" for i in range(1, REDHERRING_N + 1)], 2)
    add_cdr(a, b)

# top up remainder with a mix (red herring <-> network, more general noise)
while len(cdrs) < target_total:
    a = random.choice(person_ids)
    b = random.choice([x for x in person_ids if x != a])
    add_cdr(a, b)

random.shuffle(cdrs)
with open(f"{OUT}/cdrs.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["caller_id", "receiver_id", "timestamp", "duration_seconds", "tower_location"])
    w.writeheader()
    w.writerows(cdrs)

pure_rh = sum(1 for c in cdrs if c["caller_id"].startswith("R") and c["receiver_id"].startswith("R"))
print(f"{len(cdrs)} CDRs written ({pure_rh} pure red-herring, {pure_rh/len(cdrs):.0%}).")

# ---------------------------------------------------------------------------
# 7. FINANCIAL TRANSACTIONS (~150, 60% red-herring, with multi-hop chains)
# ---------------------------------------------------------------------------

transactions = []
modes = ["cash", "UPI", "bank_transfer"]


def add_txn(sender, receiver, ts=None, amount=None, cluster=None):
    ts = ts or rand_timestamp(cluster=cluster)
    transactions.append({
        "sender_id": sender,
        "receiver_id": receiver,
        "amount": amount or random.choice([2500, 5000, 7500, 10000, 15000, 22000, 30000, 45000, 60000]),
        "timestamp": fmt(ts),
        "mode": random.choice(modes),
    })
    return ts


# --- engineered signal: influencer transaction contact groups ---
for inf, groups in INFLUENCERS.items():
    for contact in groups["txn"]:
        cluster_day = rand_timestamp()
        for _ in range(random.randint(2, 3)):
            add_txn(inf, contact, cluster=cluster_day)

# --- multi-hop chains (A -> B -> C, sequential in time) ---
for chain in MULTIHOP_CHAINS:
    a, b, c = chain["chain"]
    t1 = rand_timestamp()
    t2 = t1 + timedelta(days=random.randint(2, 6))
    add_txn(a, b, ts=t1, amount=random.choice([20000, 35000, 50000]))
    add_txn(b, c, ts=t2, amount=random.choice([18000, 32000, 47000]))

# --- general (unflagged) network transactions ---
for _ in range(20):
    a, b = random.sample(network_people, 2)
    add_txn(a, b)

# --- red herring transactions: pad to ~150 total, >=60% (90) pure red herring ---
target_total_txn = 150
pure_rh_target_txn = int(target_total_txn * 0.6)  # 90

while sum(1 for t in transactions if t["sender_id"].startswith("R") and t["receiver_id"].startswith("R")) < pure_rh_target_txn:
    a, b = random.sample([f"R{i:02d}" for i in range(1, REDHERRING_N + 1)], 2)
    add_txn(a, b)

while len(transactions) < target_total_txn:
    a = random.choice(person_ids)
    b = random.choice([x for x in person_ids if x != a])
    add_txn(a, b)

random.shuffle(transactions)
with open(f"{OUT}/transactions.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["sender_id", "receiver_id", "amount", "timestamp", "mode"])
    w.writeheader()
    w.writerows(transactions)

pure_rh_t = sum(1 for t in transactions if t["sender_id"].startswith("R") and t["receiver_id"].startswith("R"))
print(f"{len(transactions)} transactions written ({pure_rh_t} pure red-herring, {pure_rh_t/len(transactions):.0%}).")

# ---------------------------------------------------------------------------
# 8. GROUND TRUTH (kept separate — NOT to be fed into the extraction/graph
#    pipeline; used only to score the pipeline's output afterwards)
# ---------------------------------------------------------------------------

ground_truth = {
    "note": "Verification file only. Do NOT feed this into the ingestion/graph pipeline — "
            "use it afterwards to check whether your system recovered these signals.",
    "planted_network_person_ids": network_people,
    "red_herring_person_ids": [f"R{i:02d}" for i in range(1, REDHERRING_N + 1)],
    "key_influencers": [
        {
            "person_id": inf,
            "name": by_id[inf]["name"],
            "why_hidden_per_channel": (
                f"In CDRs alone this person only touches {groups['cdr']} (degree 5). "
                f"In transactions alone only {groups['txn']} (degree 5). "
                f"In FIR mentions alone only {groups['fir']} (degree 5). "
                "None of these single-channel views make them stand out."
            ),
            "why_central_combined": (
                "Once CDR + transaction + FIR-mention edges are merged into one graph, "
                f"this person's combined neighbourhood is all 15 people above, giving them "
                "a combined degree/betweenness far above the single-channel view — "
                "this is the signal your fusion + centrality step should surface."
            ),
            "cdr_contacts": groups["cdr"],
            "txn_contacts": groups["txn"],
            "fir_contacts": groups["fir"],
        }
        for inf, groups in INFLUENCERS.items()
    ],
    "hidden_links": HIDDEN_LINKS,
    "multi_hop_chains": MULTIHOP_CHAINS,
    "fir_mention_ground_truth": [
        {"fir_id": fid, "mentioned_person_ids": pids} for fid, pids in fir_mention_log
    ],
}

with open(f"{OUT}/ground_truth.json", "w") as f:
    json.dump(ground_truth, f, indent=2)

print("ground_truth.json written.")


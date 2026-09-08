import networkx as nx
from difflib import SequenceMatcher


def find_existing_person(person_name, people, threshold=0.80):
    """
    Match a resolved PERSON name back to an existing person_id.

    This prevents duplicate identities such as:
        N01 -> Aryan Maharaj
        PERSON_1 -> Aryan Maharaj

    from becoming two separate graph nodes.

    Matches against BOTH the planted network (N..) and red herrings
    (R..) - restricting this to N-only meant any FIR mention of a
    red herring's real name could never map back to their actual
    person_id and always spawned a duplicate PERSON_x node instead.

    Also checks the `alias` column, since some FIRs deliberately use
    an alias ("Bunty", "Chintu", "Guddu") instead of the full name -
    those mentions were never being matched against anything before.
    """

    if not person_name:
        return None

    person_name = str(person_name).strip().lower()

    best_match = None
    best_score = 0.0

    for person in people:
        person_id = person.get("person_id")

        if not person_id:
            continue

        candidate_strings = []

        candidate_name = str(
            person.get("name", "")
        ).strip().lower()

        if candidate_name:
            candidate_strings.append(candidate_name)

        candidate_alias = str(
            person.get("alias", "")
        ).strip().lower()

        if candidate_alias:
            candidate_strings.append(candidate_alias)

        for candidate in candidate_strings:

            score = SequenceMatcher(
                None,
                person_name,
                candidate
            ).ratio()

            if score > best_score:
                best_score = score
                best_match = person_id

    if best_score >= threshold:
        return best_match

    return None


def build_investigation_graph(
    people,
    canonical_nodes,
    cdrs,
    transactions
):
    """
    Build the main investigation graph.

    Node types:
        PERSON
        LOCATION
        ORGANIZATION
        VEHICLE
        etc.

    Edge sources:
        CDR
        TRANSACTION

    FIR co-mentions are stored separately in:
        graph.graph["fir_co_mentions"]

    This is important because a hidden link may NOT yet exist
    as a graph edge, but FIR evidence can still exist for that pair.
    """

    graph = nx.Graph()

    # ---------------------------------------------------------
    # GRAPH-LEVEL EVIDENCE STORAGE
    # ---------------------------------------------------------

    graph.graph["fir_co_mentions"] = {}

    # ---------------------------------------------------------
    # 1. ADD BASE PEOPLE
    # ---------------------------------------------------------

    for person in people:

        person_id = person.get("person_id")

        if not person_id:
            continue

        graph.add_node(
            person_id,
            entity_type="PERSON",
            canonical_name=person.get("name", ""),
            group=person.get("group", "unknown"),
            alias=person.get("alias", "")
        )

    # ---------------------------------------------------------
    # 2. ADD RESOLVED CANONICAL NODES
    # ---------------------------------------------------------

    for canonical_id, entity in canonical_nodes.items():

        entity_type = entity.get(
            "entity_type",
            "UNKNOWN"
        )

        canonical_name = entity.get(
            "canonical_name",
            ""
        )

        # -----------------------------------------------------
        # PERSON IDENTITY MERGING
        # -----------------------------------------------------

        if entity_type == "PERSON":

            existing_person = find_existing_person(
                canonical_name,
                people
            )

            if existing_person:

                # Store resolver evidence on the real Nxx node.
                graph.nodes[existing_person][
                    "resolved_source_mentions"
                ] = entity.get(
                    "source_mentions",
                    []
                )

                continue

        # -----------------------------------------------------
        # ADD OTHER CANONICAL NODES
        # -----------------------------------------------------

        if canonical_id not in graph:

            graph.add_node(
                canonical_id,
                entity_type=entity_type,
                canonical_name=canonical_name,
                alias=entity.get("alias", ""),
                source_mentions=entity.get(
                    "source_mentions",
                    []
                )
            )

    # ---------------------------------------------------------
    # 3. ADD CDR EDGES
    # ---------------------------------------------------------

    for cdr in cdrs:

        caller = cdr.get("caller_id")
        receiver = cdr.get("receiver_id")

        if not caller or not receiver:
            continue

        if caller == receiver:
            continue

        timestamp = cdr.get("timestamp", "")
        duration = cdr.get("duration_seconds", "")
        location = cdr.get("tower_location", "")

        # If an edge already exists, preserve the existing
        # information and increase CDR weight.

        if graph.has_edge(caller, receiver):

            edge_data = graph[caller][receiver]

            edge_data["cdr_weight"] = (
                edge_data.get("cdr_weight", 0.0) + 1.0
            )

            # Keep useful event information.
            if timestamp:
                edge_data.setdefault(
                    "timestamps",
                    []
                ).append(timestamp)

            if location:
                edge_data.setdefault(
                    "locations",
                    []
                ).append(location)

        else:

            graph.add_edge(
                caller,
                receiver,
                cdr_weight=1.0,
                txn_weight=0.0,
                fir_weight=0.0,
                edge_type="CALL",
                timestamps=[timestamp]
                if timestamp else [],
                locations=[location]
                if location else [],
                durations=[duration]
                if duration else []
            )

    # ---------------------------------------------------------
    # 4. ADD TRANSACTION EDGES
    # ---------------------------------------------------------

    for transaction in transactions:

        sender = transaction.get("sender_id")
        receiver = transaction.get("receiver_id")

        if not sender or not receiver:
            continue

        if sender == receiver:
            continue

        timestamp = transaction.get("timestamp", "")
        amount = transaction.get("amount", "")
        mode = transaction.get("mode", "")

        if graph.has_edge(sender, receiver):

            edge_data = graph[sender][receiver]

            edge_data["txn_weight"] = (
                edge_data.get("txn_weight", 0.0) + 1.0
            )

            if timestamp:
                edge_data.setdefault(
                    "transaction_timestamps",
                    []
                ).append(timestamp)

            if amount:
                edge_data.setdefault(
                    "transaction_amounts",
                    []
                ).append(amount)

            if mode:
                edge_data.setdefault(
                    "transaction_modes",
                    []
                ).append(mode)

        else:

            graph.add_edge(
                sender,
                receiver,
                cdr_weight=0.0,
                txn_weight=1.0,
                fir_weight=0.0,
                edge_type="TRANSACTION",
                transaction_timestamps=[
                    timestamp
                ] if timestamp else [],
                transaction_amounts=[
                    amount
                ] if amount else [],
                transaction_modes=[
                    mode
                ] if mode else []
            )

    return graph


def add_fir_co_mention_edges(
    graph,
    canonical_nodes,
    people
):
    """
    Convert FIR PERSON co-mentions into graph evidence.

    IMPORTANT:

    FIR evidence is stored in:

        graph.graph["fir_co_mentions"]

    even when two people do not already have an edge.

    Example:

        ("N05", "N15") -> 1.0

    This allows link prediction to detect a hidden relationship
    using FIR evidence.
    """

    # ---------------------------------------------------------
    # INITIALISE FIR EVIDENCE DICTIONARY
    # ---------------------------------------------------------

    fir_co_mentions = graph.graph.get(
        "fir_co_mentions",
        {}
    )

    # ---------------------------------------------------------
    # COLLECT PERSON MENTIONS BY FIR
    # ---------------------------------------------------------

    fir_people = {}

    for canonical_id, entity in canonical_nodes.items():

        entity_type = entity.get(
            "entity_type",
            ""
        )

        if entity_type != "PERSON":
            continue

        canonical_name = entity.get(
            "canonical_name",
            ""
        )

        # Map resolver identity back to Nxx.
        person_id = find_existing_person(
            canonical_name,
            people
        )

        if not person_id:
            continue

        source_mentions = entity.get(
            "source_mentions",
            []
        )

        for mention in source_mentions:

            doc_id = mention.get("doc_id")

            if not doc_id:
                continue

            fir_people.setdefault(
                doc_id,
                set()
            ).add(person_id)

    # ---------------------------------------------------------
    # CREATE PAIRWISE FIR EVIDENCE
    # ---------------------------------------------------------

    for doc_id, person_ids in fir_people.items():

        person_ids = sorted(person_ids)

        for i in range(len(person_ids)):

            for j in range(i + 1, len(person_ids)):

                node_a = person_ids[i]
                node_b = person_ids[j]

                pair = tuple(
                    sorted(
                        (node_a, node_b)
                    )
                )

                # Increase co-mention count.
                fir_co_mentions[pair] = (
                    fir_co_mentions.get(pair, 0.0)
                    + 1.0
                )

    # ---------------------------------------------------------
    # SAVE FIR EVIDENCE
    # ---------------------------------------------------------

    graph.graph[
        "fir_co_mentions"
    ] = fir_co_mentions

    # ---------------------------------------------------------
    # ADD FIR INFORMATION TO EXISTING EDGES
    # ---------------------------------------------------------

    for pair, weight in fir_co_mentions.items():

        node_a, node_b = pair

        if graph.has_edge(node_a, node_b):

            graph[node_a][node_b][
                "fir_weight"
            ] = weight

        else:

            # IMPORTANT:
            # We DO create an edge for FIR co-mentions.
            #
            # However, the original pairwise dictionary is still
            # preserved so link prediction can use FIR evidence
            # even after an edge is removed during train/test split.

            graph.add_edge(
                node_a,
                node_b,
                cdr_weight=0.0,
                txn_weight=0.0,
                fir_weight=weight,
                edge_type="FIR_CO_MENTION"
            )

    return graph

def add_vehicle_edges(graph, vehicles):
    """
    Wires VEHICLE nodes (loaded from vehicles.csv) to their registered
    owner.

    Before this function existed, vehicles.csv was never loaded by the
    pipeline at all, and the only way a VEHICLE node could appear in
    the graph was if NER happened to spot a plate number inside a FIR
    narrative - meaning most vehicles were either missing entirely or,
    if present via NER, permanently isolated (degree 0, invisible to
    influence ranking, never shown as anyone's neighbour). This uses
    the actual structured ownership data instead.
    """

    for vehicle in vehicles:

        vehicle_id = vehicle.get("vehicle_id")
        owner_id = vehicle.get("owner_person_id")

        if not vehicle_id or not owner_id:
            continue

        if not graph.has_node(vehicle_id):
            graph.add_node(
                vehicle_id,
                entity_type="VEHICLE",
                canonical_name=f"{vehicle.get('plate_number', vehicle_id)} ({vehicle.get('make_model', 'Unknown')})",
                plate_number=vehicle.get("plate_number", ""),
                make_model=vehicle.get("make_model", "")
            )

        if graph.has_node(owner_id):
            graph.add_edge(
                owner_id,
                vehicle_id,
                edge_type="REGISTERED_OWNER",
                cdr_weight=0.0,
                txn_weight=0.0,
                fir_weight=0.0
            )

    return graph


def add_organization_edges(graph, organizations):
    """
    Wires ORGANIZATION nodes (loaded from organizations.csv) to their
    members, keeping each member's role (owner/associate) on the edge.

    Same underlying problem as vehicles: organizations.csv was never
    loaded anywhere in the pipeline, so the shell-company fronts that
    are a real part of this investigation's narrative never appeared
    in the graph at all.
    """

    for row in organizations:

        org_id = row.get("org_id")
        person_id = row.get("person_id")
        role = row.get("role", "associate")

        if not org_id or not person_id:
            continue

        if not graph.has_node(org_id):
            graph.add_node(
                org_id,
                entity_type="ORGANIZATION",
                canonical_name=row.get("name", org_id),
                org_type=row.get("type", "")
            )

        if graph.has_node(person_id):
            graph.add_edge(
                person_id,
                org_id,
                edge_type="ORG_MEMBERSHIP",
                role=role,
                cdr_weight=0.0,
                txn_weight=0.0,
                fir_weight=0.0
            )

    return graph


def add_location_edges(graph, locations, cdrs):
    """
    Wires LOCATION nodes (loaded from locations.csv) to every person
    whose call pinged that tower, aggregating repeat visits into a
    single edge with a `visit_count`.

    Location was previously only stored as a plain string attribute
    on CALL edges (`tower_location`) - never as an actual node/edge -
    so nothing about *where* people had been showed up anywhere in the
    relationship map, even though tower co-location is the exact
    mechanism behind one of this dataset's three planted hidden links
    (N22<->N16).
    """

    for loc in locations:

        location_id = loc.get("location_id")

        if not location_id:
            continue

        if not graph.has_node(location_id):
            graph.add_node(
                location_id,
                entity_type="LOCATION",
                canonical_name=loc.get("name", location_id),
                location_type=loc.get("type", ""),
                latitude=loc.get("latitude", ""),
                longitude=loc.get("longitude", "")
            )

    for cdr in cdrs:

        location_id = cdr.get("tower_location")

        if not location_id or not graph.has_node(location_id):
            continue

        for person_id in (cdr.get("caller_id"), cdr.get("receiver_id")):

            if not person_id or not graph.has_node(person_id):
                continue

            if graph.has_edge(person_id, location_id):
                graph[person_id][location_id]["visit_count"] = (
                    graph[person_id][location_id].get("visit_count", 0.0) + 1.0
                )
            else:
                graph.add_edge(
                    person_id,
                    location_id,
                    edge_type="TOWER_PING",
                    visit_count=1.0,
                    cdr_weight=0.0,
                    txn_weight=0.0,
                    fir_weight=0.0
                )

    return graph

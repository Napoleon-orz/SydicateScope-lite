import random

import networkx as nx
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler




def get_person_nodes(graph):

    return [
        node
        for node, data in graph.nodes(data=True)
        if data.get("entity_type") == "PERSON"
    ]


def get_neighbors(graph, node):


    if node not in graph:
        return set()

    return set(graph.neighbors(node))


def shared_neighbors(graph, node_a, node_b):


    neighbors_a = get_neighbors(
        graph,
        node_a
    )

    neighbors_b = get_neighbors(
        graph,
        node_b
    )

    if not neighbors_a or not neighbors_b:
        return 0.0

    shared = len(
        neighbors_a & neighbors_b
    )

    return shared / max(
        1,
        min(
            len(neighbors_a),
            len(neighbors_b)
        )
    )


def jaccard_similarity(graph, node_a, node_b):


    neighbors_a = get_neighbors(
        graph,
        node_a
    )

    neighbors_b = get_neighbors(
        graph,
        node_b
    )

    union = (
        neighbors_a |
        neighbors_b
    )

    if not union:
        return 0.0

    intersection = (
        neighbors_a &
        neighbors_b
    )

    return len(intersection) / len(union)


def shortest_path_length(graph, node_a, node_b):

    try:

        path_len = nx.shortest_path_length(
            graph,
            node_a,
            node_b
        )

        if path_len == 0:
            return 0.0

        return 1.0 / path_len

    except nx.NetworkXNoPath:

        return 0.0

    except nx.NodeNotFound:

        return 0.0




def _extract_event_data(graph, node):


    events = []

    if node not in graph:
        return events

    for _, _, data in graph.edges(
        node,
        data=True
    ):

        locations = []

        if "locations" in data:

            locations.extend(
                data.get(
                    "locations",
                    []
                )
            )

        elif data.get("location"):

            locations.append(
                data.get("location")
            )

        timestamps = []

        if "timestamps" in data:

            timestamps.extend(
                data.get(
                    "timestamps",
                    []
                )
            )

        elif data.get("timestamp"):

            timestamps.append(
                data.get("timestamp")
            )


        for timestamp, location in zip(timestamps, locations):

            if timestamp and location:

                events.append(
                    (
                        timestamp,
                        location
                    )
                )

    return events


def _parse_timestamp(timestamp):


    if not timestamp:
        return None

    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M"
    ]

    from datetime import datetime

    for fmt in formats:

        try:
            return datetime.strptime(
                str(timestamp),
                fmt
            )

        except ValueError:
            continue

    return None


def co_location_events(
    graph,
    node_a,
    node_b,
    window_minutes=60
):


    events_a = _extract_event_data(
        graph,
        node_a
    )

    events_b = _extract_event_data(
        graph,
        node_b
    )

    if not events_a or not events_b:
        return 0.0

    parsed_a = []

    for timestamp, location in events_a:

        parsed_time = _parse_timestamp(
            timestamp
        )

        if parsed_time is not None:

            parsed_a.append(
                (
                    parsed_time,
                    location
                )
            )

    parsed_b = []

    for timestamp, location in events_b:

        parsed_time = _parse_timestamp(
            timestamp
        )

        if parsed_time is not None:

            parsed_b.append(
                (
                    parsed_time,
                    location
                )
            )

    if not parsed_a or not parsed_b:
        return 0.0

    matches = 0

    for time_a, loc_a in parsed_a:

        for time_b, loc_b in parsed_b:

            if loc_a != loc_b:
                continue

            difference = abs(
                (
                    time_a - time_b
                ).total_seconds()
            )

            if difference <= window_minutes * 60:

                matches += 1

    total_events = max(
        1,
        min(
            len(parsed_a),
            len(parsed_b)
        )
    )

    return matches / total_events




def fir_co_mention_weight(
    graph,
    node_a,
    node_b
):


    fir_co_mentions = graph.graph.get(
        "fir_co_mentions",
        {}
    )

    pair = tuple(
        sorted(
            (node_a, node_b)
        )
    )

    return float(
        fir_co_mentions.get(
            pair,
            0.0
        )
    )


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(
    graph,
    node_a,
    node_b
):


    # --------------------------------------------------------
    # 1. Shared neighbours
    # --------------------------------------------------------

    try:

        shared_nbrs = shared_neighbors(
            graph,
            node_a,
            node_b
        )

    except Exception:

        shared_nbrs = 0.0

    # --------------------------------------------------------
    # 2. Jaccard
    # --------------------------------------------------------

    try:

        jaccard = jaccard_similarity(
            graph,
            node_a,
            node_b
        )

    except Exception:

        jaccard = 0.0

    # --------------------------------------------------------
    # 3. Shortest path
    # --------------------------------------------------------

    try:

        shortest_path = shortest_path_length(
            graph,
            node_a,
            node_b
        )

    except Exception:

        shortest_path = 0.0

    # --------------------------------------------------------
    # 4. Co-location
    # --------------------------------------------------------

    try:

        co_location = co_location_events(
            graph,
            node_a,
            node_b
        )

    except Exception:

        co_location = 0.0

    # --------------------------------------------------------
    # 5. Adamic-Adar
    # --------------------------------------------------------

    try:

        adamic_adar = list(
            nx.adamic_adar_index(
                graph,
                [(node_a, node_b)]
            )
        )[0][2]

    except Exception:

        adamic_adar = 0.0

    # --------------------------------------------------------
    # 6. Resource allocation
    # --------------------------------------------------------

    try:

        resource_alloc = list(
            nx.resource_allocation_index(
                graph,
                [(node_a, node_b)]
            )
        )[0][2]

    except Exception:

        resource_alloc = 0.0

    # --------------------------------------------------------
    # 7. FIR co-mention
    # --------------------------------------------------------

    try:

        fir_weight = fir_co_mention_weight(
            graph,
            node_a,
            node_b
        )

    except Exception:

        fir_weight = 0.0

    return [
        shared_nbrs,
        jaccard,
        shortest_path,
        co_location,
        adamic_adar,
        resource_alloc,
        fir_weight
    ]




def generate_negative_pairs(
    graph,
    number_of_pairs
):


    nodes = get_person_nodes(graph)

    if len(nodes) < 2:
        return []

    negative_pairs = set()

    max_possible_pairs = (
        len(nodes) *
        (len(nodes) - 1)
    ) // 2

    existing_person_edges = 0

    for node_a, node_b in graph.edges():

        if (
            node_a in nodes
            and
            node_b in nodes
        ):
            existing_person_edges += 1

    max_possible_negative_pairs = (
        max_possible_pairs -
        existing_person_edges
    )

    target = min(
        number_of_pairs,
        max_possible_negative_pairs
    )

    while len(negative_pairs) < target:

        node_a, node_b = random.sample(
            nodes,
            2
        )

        if graph.has_edge(
            node_a,
            node_b
        ):
            continue

        pair = tuple(
            sorted(
                (node_a, node_b)
            )
        )

        negative_pairs.add(pair)

    return list(negative_pairs)


# ============================================================
# TRAINING DATA
# ============================================================

def create_training_data(graph):


    # --------------------------------------------------------
    # ONLY PERSON-PERSON EDGES
    # --------------------------------------------------------

    all_edges = [
        tuple(
            sorted(
                (node_a, node_b)
            )
        )
        for node_a, node_b in graph.edges()
        if (
            graph.nodes[node_a].get(
                "entity_type"
            ) == "PERSON"
            and
            graph.nodes[node_b].get(
                "entity_type"
            ) == "PERSON"
        )
    ]

    if len(all_edges) < 5:

        raise ValueError(
            "Not enough PERSON-PERSON edges "
            "to train the link prediction model."
        )

    # --------------------------------------------------------
    # HOLDOUT SPLIT
    # --------------------------------------------------------

    train_edges, test_edges = train_test_split(
        all_edges,
        test_size=0.20,
        random_state=42
    )

    # --------------------------------------------------------
    # REMOVE TEST EDGES FROM TRAINING GRAPH
    # --------------------------------------------------------

    training_graph = graph.copy()

    training_graph.remove_edges_from(
        test_edges
    )


    negative_pairs = generate_negative_pairs(
        training_graph,
        len(train_edges)
    )

    X = []
    Y = []

    for node_a, node_b in train_edges:

        X.append(
            extract_features(
                training_graph,
                node_a,
                node_b
            )
        )

        Y.append(1)

    # --------------------------------------------------------
    # NEGATIVE EXAMPLES
    # --------------------------------------------------------

    for node_a, node_b in negative_pairs:

        X.append(
            extract_features(
                training_graph,
                node_a,
                node_b
            )
        )

        Y.append(0)

    return (
        X,
        Y,
        test_edges,
        training_graph
    )




def train_link_prediction_model(graph):

    (
        X,
        Y,
        test_edges,
        training_graph
    ) = create_training_data(
        graph
    )

    X = np.asarray(
        X,
        dtype=float
    )

    Y = np.asarray(
        Y,
        dtype=int
    )



    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X
    )



    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    )

    model.fit(
        X_scaled,
        Y
    )

    return (
        model,
        scaler,
        test_edges,
        training_graph
    )




def predict_link_probability(
    model,
    scaler,
    graph,
    node_a,
    node_b
):


    # Safety check.
    if (
        node_a not in graph
        or
        node_b not in graph
    ):
        return 0.0

    features = extract_features(
        graph,
        node_a,
        node_b
    )

    features_scaled = scaler.transform(
        [features]
    )

    probability = model.predict_proba(
        features_scaled
    )[0][1]

    return float(
        probability
    )


# ============================================================
# CANDIDATE RANKING
# ============================================================

def rank_candidate_links(
    model,
    scaler,
    graph,
    top_n=10
):
    """
    Rank unobserved PERSON-PERSON pairs.

    Existing edges are excluded.
    """

    candidates = []

    nodes = get_person_nodes(
        graph
    )

    for i in range(
        len(nodes)
    ):

        for j in range(
            i + 1,
            len(nodes)
        ):

            node_a = nodes[i]
            node_b = nodes[j]

            # Existing relationship.
            if graph.has_edge(
                node_a,
                node_b
            ):
                continue

            probability = predict_link_probability(
                model,
                scaler,
                graph,
                node_a,
                node_b
            )

            candidates.append(
                (
                    node_a,
                    node_b,
                    probability
                )
            )

    candidates.sort(
        key=lambda x: x[2],
        reverse=True
    )

    return candidates[:top_n]


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model,
    scaler,
    graph,
    test_edges,
    top_k=10
):


    candidates = rank_candidate_links(
        model,
        scaler,
        graph,
        top_n=top_k
    )

    test_edge_set = {
        tuple(
            sorted(edge)
        )
        for edge in test_edges
    }

    hits = 0

    for node_a, node_b, probability in candidates:

        pair = tuple(
            sorted(
                (node_a, node_b)
            )
        )

        if pair in test_edge_set:

            hits += 1

    if top_k == 0:
        return 0.0

    return hits / top_k
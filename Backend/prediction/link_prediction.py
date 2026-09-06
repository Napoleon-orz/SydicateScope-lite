import random
import json
import networkx as nx
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

with open("data/firs.json", "r", encoding="utf-8") as f:
    RAW_FIRS = json.load(f)


def extract_edge_features(graph, node_a, node_b):
    """
    Extracts topological and multi-channel features for a pair of nodes.
    """
    features = []

    # 1. Common Neighbors & Jaccard Coefficient
    try:
        preds_common = len(list(nx.common_neighbors(graph, node_a, node_b)))
    except Exception:
        preds_common = 0

    try:
        jaccard = list(nx.jaccard_coefficient(graph, [(node_a, node_b)]))[0][2]
    except Exception:
        jaccard = 0.0

    # 2. Path-Based & Intermediary Features (Crucial for N24 - N19 via N08)
    try:
        adamic_adar = list(nx.adamic_adar_index(graph, [(node_a, node_b)]))[0][2]
    except Exception:
        adamic_adar = 0.0

    try:
        resource_alloc = list(nx.resource_allocation_index(graph, [(node_a, node_b)]))[0][2]
    except Exception:
        resource_alloc = 0.0

    try:
        pref_attach = list(nx.preferential_attachment(graph, [(node_a, node_b)]))[0][2]
    except Exception:
        pref_attach = 0.0

    # 3. Channel-Specific Edge Weights (CDR, Transactions, FIR co-mentions)
    edge_data = graph.get_edge_data(node_a, node_b, default={})
    cdr_weight = edge_data.get('cdr_weight', 0.0)
    txn_weight = edge_data.get('txn_weight', 0.0)
    fir_weight = edge_data.get('fir_weight', 0.0)  # Boosts co-mentions like N05 - N15

    return [
        preds_common,
        jaccard,
        adamic_adar,
        resource_alloc,
        pref_attach,
        cdr_weight,
        txn_weight,
        fir_weight
    ]

def get_neighbors(graph, node):
    return set(graph.neighbors(node))


def shared_neighbors(graph, node_a, node_b):
    neighbors_a = get_neighbors(graph, node_a)
    neighbors_b = get_neighbors(graph, node_b)

    shared = len(neighbors_a & neighbors_b)

    return shared / max(1, min(len(neighbors_a), len(neighbors_b)))

def jaccard_similarity(graph, node_a, node_b):
    neighbors_a = get_neighbors(graph, node_a)
    neighbors_b = get_neighbors(graph, node_b)

    union = neighbors_a | neighbors_b

    if len(union) == 0:
        return 0.0

    intersection = neighbors_a & neighbors_b

    return len(intersection) / len(union)


def shortest_path_length(graph, node_a, node_b):
    try:
        path_len = nx.shortest_path_length(graph, node_a, node_b)
        # Convert to reachability: path of 1 -> 1.0, path of 2 -> 0.5, etc.
        return 1.0 / (path_len + 1)
    except nx.NetworkXNoPath:
        return 0.0  # Disconnected nodes get 0 reachability


def co_location_events(graph, node_a, node_b, window_minutes=60):

    events_a = []
    events_b = []

    for _, _, data in graph.edges(node_a, data=True):
        if "timestamp" in data and "location" in data:
            events_a.append((
                datetime.strptime(data["timestamp"], "%Y-%m-%d %H:%M:%S"),
                data["location"]
            ))

    for _, _, data in graph.edges(node_b, data=True):
        if "timestamp" in data and "location" in data:
            events_b.append((
                datetime.strptime(data["timestamp"], "%Y-%m-%d %H:%M:%S"),
                data["location"]
            ))

    matches = 0
    for time_a, loc_a in events_a:
        for time_b, loc_b in events_b:
            if loc_a == loc_b:
                difference = abs((time_a - time_b).total_seconds())
                if difference <= window_minutes * 60:
                    matches += 1

    total_events = max(1, min(len(events_a), len(events_b)))
    return matches / total_events

def shared_fir_mention(graph, node_a, node_b):
    name_a = graph.nodes[node_a].get("canonical_name", "")
    name_b = graph.nodes[node_b].get("canonical_name", "")
    if not name_a or not name_b:
        return 0.0

    count = 0
    for fir in RAW_FIRS:
        narrative = fir.get("narrative", "")
        if name_a in narrative and name_b in narrative:
            count += 1
    return float(count)


def extract_features(graph, node_a, node_b):
    # 1. Direct & Shared Neighbors
    try:
        shared_nbrs = shared_neighbors(graph, node_a, node_b)
    except Exception:
        shared_nbrs = 0.0

    # 2. Jaccard Similarity
    try:
        jaccard = jaccard_similarity(graph, node_a, node_b)
    except Exception:
        jaccard = 0.0

    # 3. Shortest Path Length
    try:
        shortest_path = shortest_path_length(graph, node_a, node_b)
    except Exception:
        shortest_path = 0.0

    # 4. Co-Location Events
    try:
        co_location = co_location_events(graph, node_a, node_b)
    except Exception:
        co_location = 0.0

    # 5. Adamic-Adar Index (Captures shared intermediaries like N08 for N19-N24)
    try:
        adamic_adar = list(nx.adamic_adar_index(graph, [(node_a, node_b)]))[0][2]
    except Exception:
        adamic_adar = 0.0

    # 6. Resource Allocation Index
    try:
        resource_alloc = list(nx.resource_allocation_index(graph, [(node_a, node_b)]))[0][2]
    except Exception:
        resource_alloc = 0.0

    # 7. FIR Co-Mention / Text Signal
    try:
        fir_weight = shared_fir_mention(graph, node_a, node_b)
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


def generate_negative_pairs(graph, number_of_pairs):
    nodes = list(graph.nodes())
    negative_pairs = set()

    while len(negative_pairs) < number_of_pairs:
        node_a, node_b = random.sample(nodes, 2)

        if not graph.has_edge(node_a, node_b):
            pair = tuple(sorted((node_a, node_b)))
            negative_pairs.add(pair)

    return list(negative_pairs)


def create_training_data(graph):
    all_edges = [
        tuple(sorted(edge))
        for edge in graph.edges()
    ]

    train_edges, test_edges = train_test_split(
        all_edges,
        test_size=0.2,
        random_state=42
    )

    training_graph = graph.copy()
    training_graph.remove_edges_from(test_edges)

    negative_pairs = generate_negative_pairs(
        training_graph,
        len(train_edges)
    )

    X = []
    Y = []

    for node_a, node_b in train_edges:
        X.append(
            extract_features(training_graph, node_a, node_b)
        )
        Y.append(1)

    for node_a, node_b in negative_pairs:
        X.append(
            extract_features(training_graph, node_a, node_b)
        )
        Y.append(0)

    return X, Y, test_edges, training_graph


def train_link_prediction_model(graph):
    X, Y, test_edges, training_graph = create_training_data(graph)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_scaled, Y)

    # Return both the model and the scaler so we can scale predictions too!
    return model, scaler, test_edges, training_graph


def predict_link_probability(model, scaler, graph, node_a, node_b):
    features = extract_features(graph, node_a, node_b)
    features_scaled = scaler.transform([features])
    probability = model.predict_proba(features_scaled)[0][1]
    return probability


def evaluate_model(model, scaler, graph, test_edges, top_k=10):
    # Generate top K candidate links from the graph
    candidates = rank_candidate_links(model, scaler, graph, top_n=top_k)

    # Convert test edges to a set of sorted tuples for clean O(1) lookup
    test_edge_set = {tuple(sorted(edge)) for edge in test_edges}

    hits = 0
    for node_a, node_b, prob in candidates:
        pair = tuple(sorted((node_a, node_b)))
        if pair in test_edge_set:
            hits += 1

    precision_at_k = hits / top_k
    return precision_at_k



def rank_candidate_links(model, scaler, graph, top_n=10):
    candidates = []

    nodes = [
        node
        for node, data in graph.nodes(data=True)
        if data.get("entity_type") == "PERSON"
    ]

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            node_a = nodes[i]
            node_b = nodes[j]

            if graph.has_edge(node_a, node_b):
                continue

            probability = predict_link_probability(
                model,
                scaler,
                graph,
                node_a,
                node_b
            )

            candidates.append(
                (node_a, node_b, probability)
            )

    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates[:top_n]
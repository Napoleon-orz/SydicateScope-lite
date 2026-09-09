import networkx as nx

from prediction.link_prediction import (
    get_neighbors,
    fir_co_mention_weight,
    predict_link_probability,
)


def _name(graph, node):

    if node not in graph:
        return node
    return graph.nodes[node].get("canonical_name", node) or node


def _shared_fir_docs(graph, node_a, node_b):

    docs_a = set()
    docs_b = set()

    for mentions_key in ("resolved_source_mentions", "source_mentions"):
        for m in graph.nodes.get(node_a, {}).get(mentions_key, []) or []:
            if m.get("doc_id"):
                docs_a.add(m["doc_id"])
        for m in graph.nodes.get(node_b, {}).get(mentions_key, []) or []:
            if m.get("doc_id"):
                docs_b.add(m["doc_id"])

    return sorted(docs_a & docs_b)


def _direct_evidence_summary(graph, node_a, node_b):
    """Describe an EXISTING edge between two people in plain language."""
    edge = graph[node_a][node_b]

    parts = []

    calls = edge.get("cdr_weight", 0)
    if calls > 0:
        parts.append(f"{int(calls)} call(s) between them")

    txns = edge.get("txn_weight", 0)
    if txns > 0:
        amounts = edge.get("transaction_amounts", [])
        if amounts:
            parts.append(f"{int(txns)} transaction(s) (amounts: {', '.join(amounts)})")
        else:
            parts.append(f"{int(txns)} transaction(s)")

    fir = edge.get("fir_weight", 0)
    shared_docs = _shared_fir_docs(graph, node_a, node_b)
    if fir > 0:
        if shared_docs:
            parts.append(f"co-mentioned together in FIR report(s) {', '.join(shared_docs)}")
        else:
            parts.append("co-mentioned together in at least one FIR report")

    return parts


def _co_location_detail(graph, node_a, node_b, window_minutes=60):

    from datetime import datetime

    def events(node):
        out = []
        if node not in graph:
            return out
        for _, _, data in graph.edges(node, data=True):
            locs = data.get("locations", [])
            times = data.get("timestamps", [])
            for ts, loc in zip(times, locs):
                if ts and loc:
                    out.append((ts, loc))
        return out

    def parse(ts):
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
            try:
                return datetime.strptime(str(ts), fmt)
            except ValueError:
                continue
        return None

    events_a = [(parse(t), loc) for t, loc in events(node_a)]
    events_b = [(parse(t), loc) for t, loc in events(node_b)]

    matches = []
    for time_a, loc_a in events_a:
        if time_a is None:
            continue
        for time_b, loc_b in events_b:
            if time_b is None or loc_a != loc_b:
                continue
            diff = abs((time_a - time_b).total_seconds())
            if diff <= window_minutes * 60:
                matches.append((loc_a, time_a, time_b))

    return matches


# ============================================================
# LINK EXPLANATION
# ============================================================

def explain_link(graph, node_a, node_b, model=None, scaler=None):


    if node_a not in graph or node_b not in graph:
        return f"{node_a} or {node_b} does not exist in the investigation graph."

    name_a = _name(graph, node_a)
    name_b = _name(graph, node_b)

    # ---- Case 1: they already have a direct edge -------------------
    if graph.has_edge(node_a, node_b):
        evidence = _direct_evidence_summary(graph, node_a, node_b)
        if not evidence:
            return f"{name_a} and {name_b} have a recorded relationship in the graph, but no evidence details are attached."
        return (
            f"{name_a} and {name_b} are DIRECTLY connected: "
            + "; ".join(evidence) + "."
        )

    # ---- Case 2: no direct edge - retrieve indirect evidence --------
    neighbors_a = get_neighbors(graph, node_a)
    neighbors_b = get_neighbors(graph, node_b)
    shared = neighbors_a & neighbors_b
    shared_names = [_name(graph, n) for n in list(shared)[:5]]

    try:
        hops = nx.shortest_path_length(graph, node_a, node_b)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        hops = None

    shared_fir_docs = _shared_fir_docs(graph, node_a, node_b)
    fir_weight = fir_co_mention_weight(graph, node_a, node_b)

    colocations = _co_location_detail(graph, node_a, node_b)

    fragments = []

    if shared:
        extra = f" (e.g. {', '.join(shared_names)})" if shared_names else ""
        fragments.append(f"share {len(shared)} contact(s){extra}")

    if shared_fir_docs:
        fragments.append(f"appeared together in {len(shared_fir_docs)} FIR report(s) ({', '.join(shared_fir_docs)})")
    elif fir_weight > 0:
        fragments.append("show FIR co-mention evidence")

    if colocations:
        loc, ta, tb = colocations[0]
        fragments.append(
            f"were both active at location {loc} within "
            f"{abs((ta - tb).total_seconds()) / 60:.0f} minute(s) of each other "
            f"({len(colocations)} such instance(s) total)"
        )

    if hops is not None:
        fragments.append(f"are {hops} hop(s) apart in the fused graph")

    if not fragments:
        return f"{name_a} and {name_b} have never called, transacted, or been co-mentioned - no meaningful evidence links them."

    explanation = f"{name_a} and {name_b} have never called or transacted directly, but they " + "; ".join(fragments) + "."

    if model is not None and scaler is not None:
        prob = predict_link_probability(model, scaler, graph, node_a, node_b)
        explanation += f" The link-prediction model scores this pair at {prob:.2f} probability of a hidden relationship."

    return explanation



def explain_influence(graph, node):
    """
    Explain WHY a person's influence score is high, in terms of the
    actual evidence channels they appear in - not just raw degree.
    """

    if node not in graph:
        return f"{node} does not exist in the investigation graph."

    name = _name(graph, node)

    cdr_contacts, txn_contacts, fir_contacts = [], [], []

    for other, data in graph[node].items():
        if data.get("cdr_weight", 0) > 0:
            cdr_contacts.append(other)
        if data.get("txn_weight", 0) > 0:
            txn_contacts.append(other)
        if data.get("fir_weight", 0) > 0:
            fir_contacts.append(other)

    fir_docs = set()
    for mentions_key in ("resolved_source_mentions", "source_mentions"):
        for m in graph.nodes[node].get(mentions_key, []) or []:
            if m.get("doc_id"):
                fir_docs.add(m["doc_id"])

    channels = []
    if cdr_contacts:
        channels.append(f"calls with {len(cdr_contacts)} people")
    if txn_contacts:
        channels.append(f"transactions with {len(txn_contacts)} people")
    if fir_contacts:
        detail = f" (FIR(s): {', '.join(sorted(fir_docs))})" if fir_docs else ""
        channels.append(f"FIR co-mentions with {len(fir_contacts)} people{detail}")

    n_channels = sum(bool(x) for x in (cdr_contacts, txn_contacts, fir_contacts))

    if n_channels == 0:
        return f"{name} has no recorded evidence in the graph."

    total_degree = graph.degree(node)

    explanation = (
        f"{name} has {total_degree} total connection(s) in the fused graph, "
        f"spanning {n_channels} of 3 evidence channels: " + "; ".join(channels) + ". "
    )

    if n_channels == 3:
        explanation += (
            "Because this person is corroborated across ALL THREE independent "
            "sources (calls, transactions, and FIR mentions) rather than just "
            "having a high count in one noisy channel, they rank as genuinely "
            "investigation-relevant, not just structurally busy."
        )
    else:
        explanation += (
            f"They only appear in {n_channels} of 3 evidence channel(s), which is "
            "weaker corroboration than someone appearing in all three."
        )

    return explanation
import networkx as nx


def build_investigation_graph(people: list, canonical_nodes: dict, cdrs: list, transactions: list) -> nx.Graph:
    """
    Builds a unified NetworkX graph starting with people from people.csv,
    merged canonical entities, CDR calls, and financial transactions.
    """
    # 1. Initialize an empty graph container in memory
    g = nx.Graph()

    # 2. Add all base people from people.csv as core nodes
    for person in people:
        pid = person.get("person_id")
        g.add_node(
            pid,
            entity_type="PERSON",
            canonical_name=person.get("name"),
            group=person.get("group", "unknown"),
            alias=person.get("alias", "")
        )

    # 3. Add any additional resolved canonical nodes (like vehicles/phones) if not already present
    for cid, node_data in canonical_nodes.items():
        if not g.has_node(cid):
            g.add_node(
                cid,
                entity_type=node_data["entity_type"],
                canonical_name=node_data["canonical_name"],
                source_count=len(node_data["source_mentions"])
            )

    # 4. Add Edges for Phone Call Records (CDRs)
    for cdr in cdrs:
        caller = cdr.get("caller_id")
        receiver = cdr.get("receiver_id")

        # If both nodes exist in our graph, connect them with an edge
        if g.has_node(caller) and g.has_node(receiver):
            g.add_edge(
                caller,
                receiver,
                edge_type="CALL",
                timestamp=cdr.get("timestamp"),
                duration=cdr.get("duration_seconds"),
                location=cdr.get("tower_location")
            )

    # 5. Add Edges for Financial Transactions
    for txn in transactions:
        sender = txn.get("sender_id")
        receiver = txn.get("receiver_id")

        if g.has_node(sender) and g.has_node(receiver):
            g.add_edge(
                sender,
                receiver,
                edge_type="TRANSACTION",
                timestamp=txn.get("timestamp"),
                amount=txn.get("amount"),
                mode=txn.get("mode")
            )

    print(f"Graph Built Successfully! Total Nodes: {g.number_of_nodes()}, Total Edges: {g.number_of_edges()}")
    return g

def add_fir_co_mention_edges(graph, firs_data):
    """
    Adds weighted edges between entities co-mentioned in FIR narratives.
    """
    for fir in firs_data:
        mentioned = fir.get("mentioned_person_ids", [])
        for i in range(len(mentioned)):
            for j in range(i + 1, len(mentioned)):
                u, v = mentioned[i], mentioned[j]
                if graph.has_edge(u, v):
                    graph[u][v]['fir_weight'] = graph[u][v].get('fir_weight', 0.0) + 2.0
                else:
                    graph.add_edge(u, v, fir_weight=2.0, cdr_weight=0.0, txn_weight=0.0)
    return graph
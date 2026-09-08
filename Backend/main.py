from ingest import load_json, load_csv
from ner.ner import extract_all_entities
from resolution.resolver import resolve_entities
from graph.graph import (
    build_investigation_graph,
    add_fir_co_mention_edges
)
from prediction.link_prediction import (
    get_neighbors,
    shared_neighbors,
    jaccard_similarity,
    shortest_path_length,
    co_location_events,
    extract_features,
    create_training_data,
    train_link_prediction_model,
    predict_link_probability,
    evaluate_model,
    rank_candidate_links
)
from analytics.influence import rank_influential_nodes
from rag.graph_rag import explain_link, explain_influence

def main():
    print("--- SyndicateScope: Full End-to-End Pipeline ---")

    # 1. Ingest all datasets (including people.csv)
    people = load_csv("people.csv")
    firs = load_json("firs.json")
    cdrs = load_csv("cdrs.csv")
    transactions = load_csv("transactions.csv")
    print(f"Loaded {len(people)} People, {len(firs)} FIRs, {len(cdrs)} CDRs, and {len(transactions)} Transactions.")

    # 2. NER Extraction
    known_person_names = [p.get("name", "") for p in people if p.get("name")]

    all_extracted_clues = []
    for fir in firs:
        clues = extract_all_entities(fir, known_person_names)
        all_extracted_clues.extend(clues)

    # 3. Entity Resolution (Deduplication)
    canonical_database = resolve_entities(all_extracted_clues, threshold=0.75)

    # 4. Build the NetworkX Graph
    print("\nBuilding NetworkX Graph Backend...")
    investigation_graph = build_investigation_graph(people, canonical_database, cdrs, transactions)

    investigation_graph = add_fir_co_mention_edges(
        investigation_graph,
        canonical_database,
        people
    )

    # 5. Print quick sanity check
    print("\nSample Graph Node Check:")
    sample_node = list(investigation_graph.nodes(data=True))[0]
    print(sample_node)

    print("\n--- LINK PREDICTION FEATURE TEST ---")

    node_a = "N01"
    node_b = "N24"

    print("Node A:", node_a)
    print("Node B:", node_b)

    print("Neighbours of N01:", get_neighbors(investigation_graph, node_a))
    print("Neighbours of N24:", get_neighbors(investigation_graph, node_b))

    print(
        "Shared neighbours:",
        shared_neighbors(investigation_graph, node_a, node_b)
    )

    print(
        "Jaccard similarity:",
        jaccard_similarity(investigation_graph, node_a, node_b)
    )

    print(
        "Shortest path:",
        shortest_path_length(
            investigation_graph,
            node_a,
            node_b
        )
    )

    print(
        "Co-location events:",
        co_location_events(
            investigation_graph,
            node_a,
            node_b
        )
    )

    print(
        "Feature vector:",
        extract_features(
            investigation_graph,
            node_a,
            node_b
        )
    )

    person_person_edges = sum(
        1 for a, b in investigation_graph.edges()
        if investigation_graph.nodes[a].get("entity_type") == "PERSON"
        and investigation_graph.nodes[b].get("entity_type") == "PERSON"
    )
    print(f"PERSON-PERSON edges: {person_person_edges}")

    print("\n--- LINK PREDICTION MODEL ---")

    model, scaler, test_edges, training_graph = train_link_prediction_model(
        investigation_graph
    )

    print("Hidden test edges:", len(test_edges))

    accuracy = evaluate_model(
        model,
        scaler,
        training_graph,
        test_edges
    )

    precision_k = evaluate_model(
        model,
        scaler,
        training_graph,
        test_edges,
        top_k=10
    )

    print("\n--- MODEL EVALUATION ---")
    print("Test edges:", len(test_edges))
    print("Accuracy:", accuracy)
    print("Precision@10 (Ranking Retrieval Power):", precision_k)

    feature_names = [
        "Shared Neighbours",
        "Jaccard Similarity",
        "Shortest Path",
        "Co-Location Events",
        "Adamic-Adar Index",
        "Resource Allocation",
        "FIR Co-Mention Weight"
    ]


    print("\n--- TOP HIDDEN LINK CANDIDATES ---")

    top_candidates = rank_candidate_links(
        model,
        scaler,
        investigation_graph,
        top_n=10
    )

    for rank, (node_a, node_b, probability) in enumerate(
            top_candidates,
            start=1
    ):
        print(
            f"{rank}. {node_a} - {node_b}: "
            f"{probability:.4f}"
        )

    for name, coefficient in zip(feature_names, model.coef_[0]):
        print(f"{name}: {coefficient:.4f}")


    print("Intercept:", model.intercept_[0])

    print("N01 vs N19:", extract_features(investigation_graph, "N01", "N19"))
    print("N05 vs N15:", extract_features(investigation_graph, "N05", "N15"))


    print("\n--- CHECKING KNOWN HIDDEN LINKS DIRECTLY ---")
    known_hidden_links = [("N24", "N19"), ("N22", "N16"), ("N05", "N15")]

    for node_a, node_b in known_hidden_links:
        probability = predict_link_probability(
            model,
            scaler,
            investigation_graph,
            node_a,
            node_b
        )
        print(f"{node_a} - {node_b}: {probability:.4f}")
    top_influential = rank_influential_nodes(
        investigation_graph,
        top_n=10
    )

    print("\n--- TOP INFLUENTIAL NODES ---")

    for rank, (node, score) in enumerate(top_influential, start=1):
        name = investigation_graph.nodes[node].get(
            "canonical_name",
            node
        )

        print(
            f"{rank}. {node} - {name}: {score:.4f}"
        )

    print("\n--- GRAPH-RAG EXPLANATIONS ---")

    for node_a, node_b in known_hidden_links:
        print(explain_link(investigation_graph, node_a, node_b, model, scaler))

    for node, score in top_influential[:3]:
        print(explain_influence(investigation_graph, node))


if __name__ == "__main__":
    main()
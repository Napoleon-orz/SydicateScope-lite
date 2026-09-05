from ingest import load_json, load_csv
from ner.ner import extract_all_entities
from resolution.resolver import resolve_entities
from graph.graph import build_investigation_graph
from prediction.link_prediction import (
    get_neighbors,
    shared_neighbors,
    jaccard_similarity
)

def main():
    print("--- SyndicateScope: Full End-to-End Pipeline ---")

    # 1. Ingest all datasets (including people.csv)
    people = load_csv("people.csv")
    firs = load_json("firs.json")
    cdrs = load_csv("cdrs.csv")
    transactions = load_csv("transactions.csv")
    print(f"Loaded {len(people)} People, {len(firs)} FIRs, {len(cdrs)} CDRs, and {len(transactions)} Transactions.")

    # 2. NER Extraction
    all_extracted_clues = []
    for fir in firs:
        clues = extract_all_entities(fir)
        all_extracted_clues.extend(clues)

    # 3. Entity Resolution (Deduplication)
    canonical_database = resolve_entities(all_extracted_clues, threshold=0.75)

    # 4. Build the NetworkX Graph
    print("\nBuilding NetworkX Graph Backend...")
    investigation_graph = build_investigation_graph(people, canonical_database, cdrs, transactions)

    # 5. Print quick sanity check
    print("\nSample Graph Node Check:")
    sample_node = list(investigation_graph.nodes(data=True))[0]
    print(sample_node)


if __name__ == "__main__":
    main()


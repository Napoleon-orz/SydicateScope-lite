import os
import sys
import json
import csv
import networkx as nx
from sklearn.metrics import roc_auc_score

# Ensure Python can find local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ner.ner import extract_all_entities
from resolution.resolver import resolve_entities
from graph.graph import build_investigation_graph
from prediction.link_prediction import train_link_prediction_model, rank_candidate_links

def load_csv_direct(filename):
    path = filename if os.path.exists(filename) else os.path.join("data", filename)
    with open(path, mode="r", encoding="utf-8") as file:
        return list(csv.DictReader(file))

def load_json_direct(filename):
    path = filename if os.path.exists(filename) else os.path.join("data", filename)
    with open(path, mode="r", encoding="utf-8") as file:
        return json.load(file)

def main():
    print("--- SyndicateScope: Comprehensive Model & Ground Truth Evaluation ---")

    # 1. Load data and build graph
    people = load_csv_direct("people.csv")
    firs = load_json_direct("firs.json")
    cdrs = load_csv_direct("cdrs.csv")
    transactions = load_csv_direct("transactions.csv")
    ground_truth = load_json_direct("ground_truth.json")

    all_extracted_clues = []
    for fir in firs:
        clues = extract_all_entities(fir)
        all_extracted_clues.extend(clues)
    canonical_database = resolve_entities(all_extracted_clues, threshold=0.75)
    investigation_graph = build_investigation_graph(people, canonical_database, cdrs, transactions)

    # 2. Train model and get test split outputs
    model, scaler, test_edges, training_graph = train_link_prediction_model(investigation_graph)

    # --- IMPLEMENTATION 1: Overall Model Predictive Accuracy (ROC-AUC) ---
    # If train_link_prediction_model returns X_test / y_test or if you evaluate via test_edges:
    # Let's compute classifier confidence metrics:
    print("\n--- MODEL CLASSIFICATION PERFORMANCE ---")
    # Note: Ensure your training function exposes X_test, y_test or test probability evaluations if needed,
    # or evaluate directly using test_edges structure.

    # 3. Get extended top candidates (checking top 100 to measure true Recall across all 3 planted links)
    top_candidates = rank_candidate_links(model, scaler, investigation_graph, top_n=100)

    # Load all 3 true hidden pairs from ground_truth.json
    true_hidden_pairs = set()
    for item in ground_truth.get("hidden_links", []):
        if isinstance(item, dict):
            pair = item.get("pair")
            if pair and isinstance(pair, (list, tuple)) and len(pair) >= 2:
                true_hidden_pairs.add(tuple(sorted((pair[0], pair[1]))))

    print(f"\nLoaded {len(true_hidden_pairs)} planted links: {true_hidden_pairs}")
    print("\n--- EVALUATING WIDER RETRIEVAL POOL (TOP 100) ---")

    matched_pairs = set()
    for rank, (node_a, node_b, prob) in enumerate(top_candidates, start=1):
        pair = tuple(sorted((node_a, node_b)))
        if pair in true_hidden_pairs:
            matched_pairs.add(pair)
            print(f"-> FOUND PLANTED LINK {pair} at Rank {rank} with Probability {prob:.4f}")

    recall_score = len(matched_pairs) / len(true_hidden_pairs) if true_hidden_pairs else 0
    print(f"\nGround Truth Recall (Out of {len(true_hidden_pairs)} hidden links): {recall_score * 100:.1f}% ({len(matched_pairs)}/{len(true_hidden_pairs)} recovered)")

if __name__ == "__main__":
    main()
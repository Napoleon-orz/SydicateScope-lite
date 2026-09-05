from ingest import load_json

from ner.ner import extract_all_entities

def main():
    print("--- SyndicateScope: Powering on the Factory ---")

    firs= load_json("firs.json")
    print(f"loaded {len(firs)} FIR documents")

    all_extracted_clues = []
    for fir in firs:
        clues = extract_all_entities(fir)
        all_extracted_clues.extend(clues)

    print(f"Successfully extracted {len(all_extracted_clues)} total clues across all FIRs!\n")

    counts = {}
    for clue in all_extracted_clues:
        etype = clue["entity_type"]
        counts[etype] = counts.get(etype, 0) + 1

    print("Clues Breakdown:")
    for etype, count in counts.items():
        print(f"  - {etype}: {count}")


if __name__ == "__main__":
    main()
import re
import spacy

nlp = spacy.load('en_core_web_sm')

PHONE_REGEX = re.compile(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b|\b[6-9]\d{9}\b')
VEHICLE_REGEX = re.compile(r'\b[A-Z]{2}[-\s]?\d{2}[-\s]?[A-Z]{1,2}[-\s]?\d{4}\b', re.IGNORECASE)

SPACY_TYPE_MAP ={
    "PERSON": "PERSON",
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "ORG": "ORGANIZATION",
}

def extract_regex_entities(text, doc_id):
    clues =[]

    for match in PHONE_REGEX.finditer(text):
        clues.append({
        "doc_id": doc_id,
        "entity_type": "PHONE_NUMBER",
        "surface_text": match.group(),
        "start_char": match.start(),
        "end_char": match.end(),
        "confidence": 1.0
        })

    for match in VEHICLE_REGEX.finditer(text):
        clues.append({
                "doc_id": doc_id,
                "entity_type": "VEHICLE",
                "surface_text": match.group().upper(),
                "start_char": match.start(),
                "end_char": match.end(),
                "confidence": 0.95
            })

    return clues

def extract_spacy_entities(text, doc_id):
    doc = nlp(text)
    clues = []

    for ent in doc.ents:
        if ent.label_ in SPACY_TYPE_MAP:
            clues.append({
                "doc_id": doc_id,
                "entity_type": SPACY_TYPE_MAP[ent.label_],
                "surface_text": ent.text.strip(),
                "start_char": ent.start_char,
                "end_char": ent.end_char,
                "confidence": 0.85
            })

    return clues


def extract_all_entities(fir_record):
    """Combines Regex and spaCy clues into one clean list for a single FIR report."""
    doc_id = fir_record.get("fir_id", "UNKNOWN_FIR")
    text = fir_record.get("text",
           fir_record.get("description",
           fir_record.get("details",
           fir_record.get("narrative", ""))))

    if not text:
        longest_string = ""
        for value in fir_record.values():
            if isinstance(value, str) and len(value) > len(longest_string):
                longest_string = value
        text = longest_string
    # Run both engines
    regex_clues = extract_regex_entities(text, doc_id)
    spacy_clues = extract_spacy_entities(text, doc_id)

    # Combine and deduplicate overlapping character spans
    all_clues = regex_clues + spacy_clues
    unique_clues = []
    seen_spans = set()

    for clue in all_clues:
        span_key = (clue["start_char"], clue["end_char"])
        if span_key not in seen_spans:
            seen_spans.add(span_key)
            unique_clues.append(clue)

    return unique_clues








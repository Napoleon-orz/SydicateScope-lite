import re
import difflib
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

def extract_known_person_mentions(text, doc_id, known_person_names):


    clues = []

    if not known_person_names:
        return clues

    for name in known_person_names:

        name = name.strip()

        if not name:
            continue

        pattern = re.compile(
            r'\b' + re.escape(name) + r'\b',
            re.IGNORECASE
        )

        for match in pattern.finditer(text):
            clues.append({
                "doc_id": doc_id,
                "entity_type": "PERSON",
                "surface_text": match.group(),
                "start_char": match.start(),
                "end_char": match.end(),
                "confidence": 0.99
            })

    return clues


def correct_person_misclassification(clues, known_person_names, threshold=0.85):


    if not known_person_names:
        return clues

    for clue in clues:

        if clue["entity_type"] == "PERSON":
            continue

        if clue["entity_type"] not in ("ORGANIZATION", "LOCATION"):
            continue

        surface = clue["surface_text"].strip().lower()

        for known_name in known_person_names:

            score = difflib.SequenceMatcher(
                None,
                surface,
                known_name.strip().lower()
            ).ratio()

            if score >= threshold:
                clue["entity_type"] = "PERSON"
                break

    return clues


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


def extract_all_entities(fir_record, known_person_names=None):
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
    # Run both engines, plus a direct scan for known person names
    regex_clues = extract_regex_entities(text, doc_id)
    spacy_clues = extract_spacy_entities(text, doc_id)
    gazetteer_clues = extract_known_person_mentions(
        text,
        doc_id,
        known_person_names
    )


    all_clues = regex_clues + spacy_clues + gazetteer_clues
    unique_clues = []
    seen_spans = set()

    for clue in all_clues:
        span_key = (clue["start_char"], clue["end_char"])
        if span_key not in seen_spans:
            seen_spans.add(span_key)
            unique_clues.append(clue)

    unique_clues = correct_person_misclassification(
        unique_clues,
        known_person_names
    )

    return unique_clues
import json
import csv
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

def load_json(filename):
    file_path = DATA_DIR / filename

    with open(file_path, mode="r", encoding="utf-8") as file_drawer:
        return json.load(file_drawer)

def load_csv(filename):
    file_path = DATA_DIR / filename

    with open(file_path, mode="r", encoding="utf-8") as file_drawer:
        # DictReader automatically uses the first row as the dictionary keys
        reader = csv.DictReader(file_drawer)
        return list(reader)

import json
from pathlib import Path

db_path = Path("db")

for file in db_path.glob("*.json"):
    with open(file, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            if isinstance(data, list) and len(data) > 0:
                headers = list(data[0].keys())
            elif isinstance(data, dict):
                headers = list(data.keys())
            else:
                headers = []
            print(f"{file.stem}: {headers}")
        except Exception as e:
            print(f"Error reading {file.name}: {e}")

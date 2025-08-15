from pathlib import Path
import json

def test_heart_schema_fields():
    path = Path(__file__).resolve().parents[1] / "data" / "schemas" / "heart.json"
    schema = json.loads(path.read_text())
    names = [f["name"] for f in schema["features"]]
    assert len(names) == len(set(names))
    assert "age" in names and "chol" in names and "oldpeak" in names

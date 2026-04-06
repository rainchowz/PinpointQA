import json
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: Path) -> Any:
    """Load a JSON file with UTF-8 encoding."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_text(path: Path) -> str:
    """Load a UTF-8 text file."""
    with path.open("r", encoding="utf-8") as f:
        return f.read()


def dump_json(obj: Any, path: Path) -> None:
    """Write JSON with readable formatting."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_jsonl(record: Dict[str, Any], path: Path) -> None:
    """Append one JSON record to a JSONL file."""
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file if it exists."""
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def ensure_dir(path: Path) -> None:
    """Create a directory recursively if needed."""
    path.mkdir(parents=True, exist_ok=True)

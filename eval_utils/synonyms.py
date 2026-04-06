from pathlib import Path
from typing import Dict, Optional

from .io_utils import load_json
from .normalize import _basic_normalize_label


def load_synonyms(path: Optional[Path]) -> Dict[str, str]:
    """Load an optional user synonym map and return custom aliases only."""
    synonyms: Dict[str, str] = {}
    if path is None:
        return synonyms
    user_map = load_json(path)
    if not isinstance(user_map, dict):
        raise ValueError(f"Synonym file must be a JSON object: {path}")
    for k, v in user_map.items():
        synonyms[_basic_normalize_label(k)] = _basic_normalize_label(v)
    return synonyms

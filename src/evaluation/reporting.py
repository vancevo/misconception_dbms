"""Result serialization to JSON files in results/."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


def save_results(
    results: dict[str, Any],
    filename: str,
    results_dir: str | Path = "results",
) -> Path:
    """Serialize a results dict to a JSON file in results/.

    Args:
        results: Dictionary of metric results to serialize.
        filename: Output filename (without .json extension if not provided).
        results_dir: Directory to write results into (default: "results/").

    Returns:
        Path to the written JSON file.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if not filename.endswith(".json"):
        filename = filename + ".json"

    output_path = results_dir / filename

    payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "results": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_serializer)

    return output_path


def _json_serializer(obj: Any) -> Any:
    """Handle non-serializable types (numpy scalars, etc.)."""
    import numpy as np

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def load_results(filepath: str | Path) -> dict[str, Any]:
    """Load a previously saved results JSON file.

    Args:
        filepath: Path to the JSON file.

    Returns:
        The 'results' dict from the file.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("results", payload)

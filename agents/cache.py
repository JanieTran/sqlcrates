import json
from pathlib import Path

from config import logger, settings
from models.schemas import CollectionInsight, DatasetProfile

CACHE_PATH = Path(settings.output_dir) / settings.cache_file


def _current_markers(data_dir: str = "data") -> dict[str, float] | None:
    """Build ``{filename: mtime}`` for CSVs + their metadata files in *data_dir*."""
    root = Path(data_dir)
    if not root.is_dir():
        return None
    csv_files = sorted(root.glob("*.csv"))
    if not csv_files:
        return None

    markers: dict[str, float] = {}
    for csv_path in csv_files:
        markers[csv_path.name] = csv_path.stat().st_mtime
        md_path = csv_path.with_suffix(".md")
        if md_path.exists():
            markers[md_path.name] = md_path.stat().st_mtime
        else:
            txt_path = csv_path.with_suffix(".txt")
            if txt_path.exists():
                markers[txt_path.name] = txt_path.stat().st_mtime
    return markers


def save_run(
    profiles: list[DatasetProfile],
    insight: CollectionInsight,
    data_dir: str = "data",
) -> None:
    """Persist profiles + insight keyed by current file timestamps."""
    markers = _current_markers(data_dir)
    data = {
        "markers": markers,
        "profiles": [p.model_dump() for p in profiles],
        "insight": insight.model_dump(),
    }
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    logger.info("Run cache saved (%d profiles)", len(profiles))


def load_run(data_dir: str = "data") -> tuple[list[DatasetProfile], CollectionInsight] | None:
    """Load cached profiles + insight if still valid, otherwise ``None``."""
    if not CACHE_PATH.exists():
        return None

    data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))

    # Invalidate if any CSV/metadata has changed or been added/removed
    current = _current_markers(data_dir)
    if current is None or data.get("markers") != current:
        logger.info("Cache invalidated — data directory changed")
        return None

    profiles = [DatasetProfile.model_validate(p) for p in data["profiles"]]
    insight = CollectionInsight.model_validate(data["insight"])
    logger.info("Loaded %d profiles from cache", len(profiles))
    return profiles, insight

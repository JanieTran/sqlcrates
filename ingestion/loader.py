import sys
from pathlib import Path
import pandas as pd
from config import conn, logger
from ingestion.profiler import profile
from agents.domain_agent import infer_domain
from models.schemas import DatasetProfile


def _table_name(path: Path) -> str:
    return path.stem.replace("-", "_").replace(" ", "_").lower()


def load_csv(path: str | Path, table_name: str | None = None) -> pd.DataFrame:
    """Read a CSV file and register it in DuckDB as *table_name*."""
    path = Path(path)
    if not path.exists():
        logger.error("File not found: %s", path)
        raise FileNotFoundError(f"CSV file not found: {path}")

    if table_name is None:
        table_name = _table_name(path)

    logger.info("Loading CSV: %s", path)
    df = pd.read_csv(path, skipinitialspace=True, low_memory=False, nrows=100_000)
    logger.info("Loaded %d rows x %d columns (max 100k)", *df.shape)

    conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM df")
    logger.info("Registered %d rows in DuckDB as table `%s`", len(df), table_name)

    return df


def read_metadata(path: str | Path | None) -> str:
    """Read an optional metadata file (.md or .txt). Returns empty string when absent."""
    if path is None:
        return ""

    p = Path(path)
    if not p.exists():
        logger.warning("Metadata file not found (ignoring): %s", p)
        return ""

    logger.info("Reading metadata: %s", p)
    return p.read_text(encoding="utf-8").strip()


def discover_datasets(data_dir: str = "data") -> list[DatasetProfile]:
    """Scan *data_dir* for .csv files, load, profile, infer domain — return up to 3 DatasetProfiles."""
    root = Path(data_dir)
    if not root.is_dir():
        logger.error("Data directory not found: %s", root)
        return []

    csv_files = sorted(root.glob("*.csv"))
    if not csv_files:
        logger.warning("No CSV files found in %s", root)
        return []

    if len(csv_files) > 3:
        logger.error("Found %d CSV files in %s (max 3) — aborting", len(csv_files), root)
        sys.exit(1)

    profiles: list[DatasetProfile] = []
    for csv_path in csv_files:
        name = _table_name(csv_path)

        md_path = csv_path.with_suffix(".md")
        if not md_path.exists():
            md_path = csv_path.with_suffix(".txt")
            if not md_path.exists():
                md_path = None

        logger.info("Processing dataset: %s", name)
        df = load_csv(csv_path, table_name=name)
        meta = read_metadata(md_path)
        prof = profile(df)
        prof.name = name
        prof = infer_domain(prof, meta)
        profiles.append(prof)

    logger.info(
        "Discovered %d dataset(s): %s",
        len(profiles),
        ", ".join(p.name for p in profiles),
    )
    return profiles

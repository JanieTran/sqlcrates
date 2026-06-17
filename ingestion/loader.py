import sys
from pathlib import Path

import pandas as pd

from config import conn, logger
from ingestion.profiler import profile
from agents.domain_agent import infer_dataset_context
from models.schemas import DatasetProfile


def _table_name(path: Path) -> str:
    """Derive a safe DuckDB table name from a CSV filename."""
    return path.stem.replace("-", "_").replace(" ", "_").lower()


def load_csv(path: str | Path, table_name: str | None = None) -> pd.DataFrame:
    """Read a CSV file, return a DataFrame, and register it as a DuckDB table.

    Parameters
    ----------
    path
        Path to the CSV file on disk.
    table_name
        Name for the DuckDB table. When *None*, derived automatically from the
        CSV filename via ``_table_name()``.

    Returns
    -------
    pd.DataFrame
        The full contents of the CSV (capped at 100 000 rows).
    """
    path = Path(path)
    if not path.exists():
        logger.error("File not found: %s", path)
        raise FileNotFoundError(f"CSV file not found: {path}")

    if table_name is None:
        table_name = _table_name(path)

    logger.info("Loading CSV: %s", path)
    df = pd.read_csv(path, skipinitialspace=True, low_memory=False, nrows=100_000)
    logger.info("Loaded %d rows x %d columns (max 100k)", *df.shape)

    # Register the DataFrame as a DuckDB table so subsequent SQL tools can
    # query it.  Using CREATE OR REPLACE makes re-runs idempotent.
    conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM df")
    logger.info("Registered %d rows in DuckDB as table `%s`", len(df), table_name)

    return df


def read_metadata(path: str | Path | None) -> str:
    """Read an optional metadata file (``.md`` or ``.txt``).

    Returns the file contents stripped of leading/trailing whitespace, or an
    empty string when the path is *None* or the file does not exist.
    """
    if path is None:
        return ""

    p = Path(path)
    if not p.exists():
        logger.warning("Metadata file not found (ignoring): %s", p)
        return ""

    logger.info("Reading metadata: %s", p)
    return p.read_text(encoding="utf-8").strip()


def discover_datasets(data_dir: str = "data") -> list[DatasetProfile]:
    """Scan *data_dir*, load and profile every CSV, and return fully-enriched profiles.

    1. Lists all ``.csv`` files inside *data_dir* (alphabetical order).
    2. Caps at 3 datasets — exits with an error if more are found.
    3. For each CSV:
       a. Derives a DuckDB table name from the filename.
       b. Picks up a matching ``.md`` (preferred) or ``.txt`` metadata file.
       c. Loads the CSV into DuckDB + a Pandas DataFrame.
       d. Runs the heuristic profiler to infer column roles and statistics.
       e. Calls the LLM domain agent to populate per-dataset context
          (grain, temporal coverage, description).
    4. Returns the list of enriched ``DatasetProfile`` objects.

    Returns
    -------
    list[DatasetProfile]
        One entry per CSV, in alphabetical order.
    """
    root = Path(data_dir)
    if not root.is_dir():
        logger.error("Data directory not found: %s", root)
        return []

    csv_files = sorted(root.glob("*.csv"))
    if not csv_files:
        logger.warning("No CSV files found in %s", root)
        return []

    # Keep profiling + LLM calls tractable by limiting to 3 datasets
    if len(csv_files) > 3:
        logger.error("Found %d CSV files in %s (max 3) — aborting", len(csv_files), root)
        sys.exit(1)

    profiles: list[DatasetProfile] = []
    for csv_path in csv_files:
        # Derive a table name safe for DuckDB (lowercase, underscores)
        name = _table_name(csv_path)

        # Locate a matching metadata file — try .md first, fall back to .txt
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
        prof = infer_dataset_context(prof, meta)
        profiles.append(prof)

    logger.info(
        "Discovered %d dataset(s): %s",
        len(profiles),
        ", ".join(p.name for p in profiles),
    )
    return profiles

from pathlib import Path
from config import conn, logger
import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
    """Read a CSV file into a DataFrame and register it in DuckDB as ``data``."""
    path = Path(path)
    if not path.exists():
        logger.error("File not found: %s", path)
        raise FileNotFoundError(f"CSV file not found: {path}")

    logger.info("Loading CSV: %s", path)
    df = pd.read_csv(path, skipinitialspace=True, low_memory=False, nrows=100_000)
    logger.info("Loaded %d rows x %d columns (max 100k)", *df.shape)

    conn.execute("CREATE OR REPLACE TABLE data AS SELECT * FROM df")
    logger.info(
        "Registered %d rows in DuckDB as table `data`",
        len(df),
    )

    return df


def read_metadata(path: str | None) -> str:
    """Read an optional metadata text file. Returns empty string when absent."""
    if path is None:
        return ""

    p = Path(path)
    if not p.exists():
        logger.warning("Metadata file not found (ignoring): %s", p)
        return ""

    logger.info("Reading metadata: %s", p)
    return p.read_text(encoding="utf-8").strip()

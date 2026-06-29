import math
import pandas as pd
from config import logger
from models.schemas import *

# Suffixes used to hint at a column being an identifier
_ID_SUFFIXES = ("_id", "_uuid", "_key", "_code", "_uid")
_DATE_KW = {"date", "time", "timestamp", "year", "month", "day"}
_BOOL_WORDS = {"true", "false", "yes", "no", "t", "f", "y", "n"}


def _infer_role(
    name: str,
    dtype: str,
    col: pd.Series,
    row_count: int,
    cardinality: int,
    avg_word_count: float | None,
) -> ColumnRole:
    """Heuristically assign a ColumnRole to a single column."""
    name_lower = name.lower()

    if "datetime" in dtype or "timedelta" in dtype:
        return ColumnRole.DATETIME
    if dtype == "bool":
        return ColumnRole.BOOLEAN

    is_id_name = any(name_lower.endswith(sfx) for sfx in _ID_SUFFIXES) or name_lower in ("id", "uuid", "key", "code", "uid")

    if dtype.startswith(("int", "float")):
        if cardinality <= 2:
            unique_vals = set(col.dropna().unique())
            if unique_vals <= {0, 1, 0.0, 1.0}:
                return ColumnRole.BOOLEAN
        if cardinality == 1:
            return ColumnRole.CATEGORICAL
        if cardinality == row_count:
            return ColumnRole.IDENTIFIER if is_id_name else ColumnRole.CONTINUOUS
        if row_count > 100 and cardinality / row_count < 0.01:
            return ColumnRole.CATEGORICAL
        return ColumnRole.CONTINUOUS

    if dtype in ("object", "str"):
        ratio = cardinality / row_count if row_count else 0

        if cardinality <= 2 and row_count > 0:
            vals = {str(v).lower() for v in col.dropna().unique()}
            if vals <= _BOOL_WORDS | {"0", "1"}:
                return ColumnRole.BOOLEAN

        # Short labels (≤3 words on average) are always categorical
        if avg_word_count is not None and avg_word_count <= 3:
            return ColumnRole.CATEGORICAL if not is_id_name else ColumnRole.IDENTIFIER

        # Long descriptive text
        if avg_word_count is not None and avg_word_count > 8:
            return ColumnRole.FREE_TEXT

        if cardinality == row_count:
            return ColumnRole.IDENTIFIER if is_id_name else ColumnRole.FREE_TEXT

        if ratio < 0.1:
            return ColumnRole.CATEGORICAL

        return ColumnRole.FREE_TEXT

    return ColumnRole.CATEGORICAL


def _profile_column(name: str, col: pd.Series, row_count: int) -> ColumnProfile:
    """Build a ColumnProfile for a single column."""
    # Compute null rate and cardinality
    null_count = col.isna().sum()
    null_rate = null_count / row_count if row_count else 0.0
    cardinality = int(col.nunique())

    # Extract column dtype as string (e.g. "int64", "object", etc.)
    dtype = str(col.dtype)

    # Compute average word count for string columns
    avg_word_count: float | None = None
    if dtype in ("object", "str"):
        word_counts = col.dropna().astype(str).str.split().str.len()
        if not word_counts.empty:
            avg_word_count = float(word_counts.mean())

    role = _infer_role(name, dtype, col, row_count, cardinality, avg_word_count)

    # For continuous columns, compute numeric stats
    numeric_stats: NumericStats | None = None
    if role == ColumnRole.CONTINUOUS:
        s = col.dropna()
        if not s.empty:
            desc = s.describe()
            numeric_stats = NumericStats(
                mean=float(desc.get("mean", 0)),
                median=float(s.median()),
                min=float(desc.get("min", 0)),
                max=float(desc.get("max", 0)),
                std=float(s.std()) if math.isfinite(s.std()) else None,
                p25=float(desc.get("25%", 0)),
                p75=float(desc.get("75%", 0)),
                skew=float(s.skew()) if math.isfinite(s.skew()) else None,
            )

    # For categorical or boolean columns, compute top values
    categorical_stats: CategoricalStats | None = None
    if role in (ColumnRole.CATEGORICAL, ColumnRole.BOOLEAN):
        vc = col.dropna().value_counts().head(10)
        categorical_stats = CategoricalStats(
            top_values=dict(zip(vc.index.astype(str), vc.values.tolist())),
            unique_count=cardinality,
        )

    return ColumnProfile(
        name=name,
        dtype=dtype,
        role=role,
        null_rate=null_rate,
        cardinality=cardinality,
        numeric_stats=numeric_stats,
        categorical_stats=categorical_stats,
    )


def _compute_correlations(df: pd.DataFrame) -> list[CorrelationPair]:
    """Pearson correlation for all numeric column pairs, |r|-sorted descending."""
    # Select only numeric dtypes — correlation is undefined for text/bool.
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.shape[1] < 2:
        return []

    # pandas .corr() returns a square matrix of Pearson r values.  We walk
    # the upper triangle (i < j) to avoid duplicate pairs and self-pairs.
    corr = numeric_df.corr()
    pairs: list[CorrelationPair] = []
    for i, c1 in enumerate(corr.columns):
        for c2 in corr.columns[i + 1 :]:
            r = corr.loc[c1, c2]
            if not math.isnan(r):
                pairs.append(CorrelationPair(col1=c1, col2=c2, r=r))

    # Most interesting (strongest relationship) first
    pairs.sort(key=lambda p: abs(p.r), reverse=True)
    return pairs[:20]


def profile(df: pd.DataFrame) -> DatasetProfile:
    """Profile a DataFrame and return a DatasetProfile with column-level info."""
    row_count = len(df)
    if row_count == 0:
        logger.warning("DataFrame is empty — returning empty profile")
        return DatasetProfile(row_count=0, column_count=0, columns=[])

    logger.info("Profiling %d rows x %d columns", *df.shape)

    columns = [_profile_column(name, df[name], row_count) for name in df.columns]

    # Small random sample so the LLM can see raw data
    sample_size = min(5, row_count)
    sample_rows = df.sample(n=sample_size, random_state=42).to_dict(orient="records")

    profile = DatasetProfile(
        row_count=row_count,
        column_count=len(df.columns),
        columns=columns,
        sample_rows=sample_rows,
    )

    # Log a quick summary of detected column roles for user visibility
    role_counts: dict[str, int] = {}
    for c in columns:
        role_counts[c.role.value] = role_counts.get(c.role.value, 0) + 1
    logger.info("Detected roles: %s", role_counts)

    # Cross-column correlations 
    correlations = _compute_correlations(df)
    if correlations:
        profile.correlations = correlations
        logger.info(
            "Top correlations: %s",
            ", ".join(f"{p.col1}↔{p.col2}(r={p.r:.2f})" for p in correlations[:5]),
        )

    return profile

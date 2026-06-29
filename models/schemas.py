from __future__ import annotations
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class CollectionInsight(BaseModel):
    """High-level overview of the combined dataset collection.

    Populated by the domain agent after all datasets have been profiled.
    """
    domain: str
    description: str
    domain_knowledge: list[str] = Field(default_factory=list)
    seed_questions: list[str] = Field(default_factory=list)
    exploration_ideas: list[str] = Field(default_factory=list)


class ColumnRole(str, Enum):
    """Semantic role assigned to a column by the profiler.

    Determines how the column is treated in downstream analysis and what
    statistics are computed for it.
    """
    IDENTIFIER = "identifier"    # unique row key (e.g. id, uuid)
    CATEGORICAL = "categorical"  # discrete label with few unique values
    CONTINUOUS = "continuous"    # numeric measurement on a continuous scale
    DATETIME = "datetime"        # timestamp or date
    BOOLEAN = "boolean"          # true/false or 0/1
    FREE_TEXT = "free_text"      # unstructured text (e.g. description, comment)


class NumericStats(BaseModel):
    """Summary statistics for a continuous numeric column."""
    mean: float | None = None
    median: float | None = None
    min: float | None = None       # minimum value
    max: float | None = None       # maximum value
    std: float | None = None       # standard deviation
    p25: float | None = None       # 25th percentile / first quartile
    p75: float | None = None       # 75th percentile / third quartile
    skew: float | None = None      # skewness (asymmetry of distribution)


class CategoricalStats(BaseModel):
    """Frequency distribution for a categorical column."""
    top_values: dict[str, int] = Field(default_factory=dict)  # value → count for most frequent values
    unique_count: int = 0                                     # number of distinct values


class CorrelationPair(BaseModel):
    col1: str
    col2: str
    r: float


class ColumnProfile(BaseModel):
    """All profiled information for a single column."""
    name: str                                    # column name as it appears in the CSV header
    dtype: str                                   # pandas dtype string (e.g. int64, object, datetime64[ns])
    role: ColumnRole                             # semantic role assigned by the profiler (identifier, categorical, …)
    null_rate: float                             # fraction of rows where this column is null (0.0 – 1.0)
    cardinality: int                             # number of distinct non-null values
    numeric_stats: NumericStats | None = None    # populated when role is CONTINUOUS
    categorical_stats: CategoricalStats | None = None  # populated when role is CATEGORICAL (or BOOLEAN)

    def compact(self) -> str:
        """Render a token-efficient one-line summary for LLM context windows."""
        def _fmt(v: float | None) -> str:
            return f"{v:.4f}" if v is not None else "N/A"

        parts = [f"{self.name} ({self.dtype}, {self.role.value})"]
        parts.append(f"nulls={self.null_rate:.1%}  cardinality={self.cardinality}")
        if self.numeric_stats:
            s = self.numeric_stats
            parts.append(
                f"range=[{_fmt(s.min)}, {_fmt(s.max)}]  mean={_fmt(s.mean)}  "
                f"med={_fmt(s.median)}  sd={_fmt(s.std)}  "
                f"p25={_fmt(s.p25)}  p75={_fmt(s.p75)}  skew={_fmt(s.skew)}"
            )
        if self.categorical_stats:
            top = dict(list(self.categorical_stats.top_values.items())[:5])
            parts.append(f"top={top}")
        return " - ".join(parts)


class DatasetProfile(BaseModel):
    """Per-dataset profile with column-level stats and contextual fields.

    Populated in two stages:
    1. Profiler fills row_count, column_count, columns, sample_rows.
    2. Domain agent fills grain, temporal_coverage, description.

    The combined cross-dataset view lives in ``CollectionInsight``.
    """
    name: str = ""                                       # dataset name (stem of the CSV file)
    row_count: int                                      # total rows in the CSV
    column_count: int                                   # total columns
    columns: list[ColumnProfile]                        # one entry per column, in CSV order
    sample_rows: list[dict[str, Any]] = Field(default_factory=list)  # small random sample (≤5 rows)
    grain: str = ""                                      # what one row represents, e.g. "one product per row"
    temporal_coverage: str | None = None                 # date range if datetime columns exist, e.g. "2020-01 – 2024-12"
    description: str = ""                                # brief text explaining what the dataset is about
    correlations: list[CorrelationPair] = Field(default_factory=list)

    def compact(self) -> str:
        """Render the entire profile as a compact, LLM-friendly text block."""
        lines = [f"Rows: {self.row_count}  Columns: {self.column_count}"]
        for col in self.columns:
            lines.append(col.compact())
        if self.sample_rows:
            lines.append("Sample rows:")
            for i, row in enumerate(self.sample_rows[:3]):
                lines.append(f"  [{i}] {row}")
        if self.correlations:
            top = self.correlations[:3]
            parts = [f"{p.col1} ↔ {p.col2} (r={p.r:.3f})" for p in top]
            lines.append("Top correlations: " + ", ".join(parts))
        return "\n".join(lines)


class InsightCard(BaseModel):
    """A single finding from the exploration or QA agent."""
    question: str                                           # the analytical question being answered
    sql_query: str | None = None                            # the SQL query that was executed
    result_summary: str                                     # plain-language summary of what the data says
    interpretation: str                                     # what this finding means in the dataset's context
    follow_ups: list[str] = Field(default_factory=list)     # new questions this insight suggests

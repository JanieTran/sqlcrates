import json
from config import logger
from agents.llm_client import call_llm
from models.responses import SqlGenerationResponse
from models.schemas import CollectionInsight, DatasetProfile


def _build_schema_context(profiles: list[DatasetProfile]) -> str:
    """Render a compact table + column reference for the LLM prompt."""
    parts = []
    for p in profiles:
        parts.append(f"table `{p.name}` ({p.row_count} rows, {p.column_count} columns)")
        for col in p.columns:
            parts.append(f"  {col.name}  {col.dtype}  ({col.role.value})")
    return "\n".join(parts)


_SYSTEM_PROMPT = (
    "You are a SQL expert. Given a natural language question and the dataset "
    "schemas below, write valid DuckDB SQL queries to answer the question.\n\n"
    "Rules:\n"
    "- Use only the tables and columns listed below.\n"
    "- Output concrete, runnable SQL — not pseudocode.\n"
    "- If fully answering the question requires pulling different data from "
    "different angles (e.g. a summary count + a detail list, or stats for "
    "different groups that can't fit in one query), write one query per angle "
    "as separate list elements. Each element must be a single valid query.\n"
    "- Do NOT split one query into identical duplicates. Each query should "
    "serve a distinct purpose.\n"
    "- Query results must be returned as rows — use LIMIT where appropriate.\n"
    "- Use standard SQL that DuckDB supports.\n"
    "- When joining tables, use the table-qualified column names.\n"
    "- Respond with valid JSON only.\n"
)


def generate_sql(
    question: str,
    profiles: list[DatasetProfile],
    insight: CollectionInsight | None = None,
) -> SqlGenerationResponse:
    """Convert a natural language *question* into one or more DuckDB SQL queries.

    Parameters
    ----------
    question
        Natural language question about the dataset(s).
    profiles
        Profiled dataset descriptions (table name, columns, types, roles).
    insight
        Optional collection-level context (domain, domain knowledge).

    Returns
    -------
    SqlGenerationResponse
        A brief explanation and a list of SQL queries to execute.
    """
    schema_context = _build_schema_context(profiles)
    parts = [f"Question: {question}\n\nAvailable schemas:\n{schema_context}"]

    if insight:
        extra = ""
        if insight.domain:
            extra += f"Domain: {insight.domain}\n"
        if insight.description:
            extra += f"Description: {insight.description}\n"
        if insight.domain_knowledge:
            extra += "Domain knowledge:\n"
            for item in insight.domain_knowledge:
                extra += f"  - {item}\n"
        if extra:
            parts.append(extra)

    user_prompt = "\n\n".join(parts)

    logger.info("Generating SQL...")
    result = call_llm(
        system_prompt=(
            _SYSTEM_PROMPT
            + "\n\n"
            + f"JSON Schema:\n{json.dumps(SqlGenerationResponse.model_json_schema(), indent=2)}"
        ),
        user_prompt=user_prompt,
        response_model=SqlGenerationResponse,
    )

    logger.info("Generated %d query(ies)", len(result["queries"]))

    return SqlGenerationResponse(
        explanation=result["explanation"],
        queries=result["queries"],
    )

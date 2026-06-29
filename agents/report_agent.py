from pathlib import Path

from agents.llm_client import call_llm
from config import logger, settings
from models.responses import InterpretationResponse
from models.schemas import CollectionInsight, DatasetProfile, InsightCard


_SYSTEM_PROMPT = (
    "You are a data journalist. Given a dataset collection profile below, "
    "write a well-structured markdown summary for a technical audience.\n\n"
    "Cover:\n"
    "- What the collection is about — overall domain and scope\n"
    "- For each dataset: what it contains, its row count, key columns worth "
    "noting, and any interesting correlations between variables\n"
    "- Notable patterns, caveats, and suggested next steps for analysis\n\n"
    "Use headings, bullet points, and tables where appropriate. "
    "Do NOT wrap the output in a code block — write raw markdown only."
)


def write_summary(profiles: list[DatasetProfile], insight: CollectionInsight) -> Path:
    """Generate a narrative markdown summary via LLM and write to ``settings.summary_file``."""
    out_dir = Path(settings.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / settings.summary_file

    # Build the user prompt from profiles + insight
    parts = [
        f"# {insight.domain}\n"
        f"**Collection description:** {insight.description}\n"
    ]
    if insight.domain_knowledge:
        parts.append("**Domain knowledge:**")
        parts.extend(f"- {item}" for item in insight.domain_knowledge)
    if insight.exploration_ideas:
        parts.append("\n**Exploration ideas:**")
        parts.extend(f"- {idea}" for idea in insight.exploration_ideas)

    for prof in profiles:
        parts.append(f"\n\n## {prof.name}\n{prof.compact()}")
        if prof.correlations:
            corr_strs = [f"- {p.col1} ↔ {p.col2} (r={p.r:.3f})" for p in prof.correlations]
            parts.append("**Correlations:**\n" + "\n".join(corr_strs))

    user_prompt = "\n".join(parts)

    logger.info("Generating summary via LLM...")
    result = call_llm(
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )

    out_path.write_text(str(result), encoding="utf-8")
    logger.info("Summary written to %s", out_path.resolve())
    return out_path


_SYSTEM_PROMPT_INTERPRET = (
    "You are a data analyst. Given a question, the SQL approach used to "
    "answer it, the query results, and dataset context, produce a concise "
    "structured interpretation.\n\n"
    "Be specific — reference actual values from the query results. "
    "Do NOT mention SQL or queries in your output; write for a data-literate "
    "reader who has not seen the raw queries.\n\n"
    "Respond in JSON with exactly these keys:\n"
    '- "result_summary": plain-language summary of what the query results show\n'
    '- "interpretation": what this finding means in the dataset\'s context\n'
    '- "follow_ups": a list of 2-3 follow-up question strings\n'
    "Return ONLY valid JSON, no other text."
)


def interpret_results(
    question: str,
    query_explanation: str,
    results: list[dict],
    profiles: list[DatasetProfile],
    insight: CollectionInsight,
) -> InsightCard:
    """Interpret query results and return a structured insight card."""
    parts = [
        f"# Question\n{question}",
        f"# SQL approach\n{query_explanation}",
        "# Query results",
    ]
    for i, r in enumerate(results, start=1):
        sample_rows = r["rows"][:15]
        rows_text = "\n".join(
            str(row) for row in sample_rows
        ) if sample_rows else "(empty)"
        parts.append(
            f"### Query {i} ({r['row_count']} rows returned)\n"
            f"Columns: {r['columns']}\n{rows_text}"
        )
        if r["error"]:
            parts.append(f"Error: {r['error']}")

    parts.append(f"# Dataset profiles\n{''.join(p.compact() for p in profiles)}")

    parts.append(
        f"# Collection context\n"
        f"Domain: {insight.domain}\n"
        f"Description: {insight.description}"
    )

    user_prompt = "\n\n".join(parts)
    logger.info("Interpreting query results via LLM...")

    try:
        resp = call_llm(
            system_prompt=_SYSTEM_PROMPT_INTERPRET,
            user_prompt=user_prompt,
            response_model=InterpretationResponse,
        )
    except Exception:
        logger.exception("Interpretation LLM call failed, returning partial card")
        return InsightCard(
            question=question,
            result_summary="(interpretation failed)",
            interpretation="",
            follow_ups=[],
        )

    return InsightCard(
        question=question,
        result_summary=resp["result_summary"],
        interpretation=resp["interpretation"],
        follow_ups=resp.get("follow_ups", []),
    )

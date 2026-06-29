from pathlib import Path

from agents.llm_client import call_llm
from config import logger
from models.schemas import CollectionInsight, DatasetProfile


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
    """Generate ``output/summary.md`` by asking the LLM to write a narrative summary."""
    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.md"

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

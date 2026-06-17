import json
from config import logger
from agents.llm_client import call_llm
from models.responses import CollectionResponse, DatasetContextResponse
from models.schemas import CollectionInsight, DatasetProfile


_DATASET_SYSTEM_PROMPT = (
    "You are a data analyst. Given a dataset profile below, infer the following "
    "about the dataset and respond with valid JSON only.\n\n"
    f"JSON Schema:\n{json.dumps(DatasetContextResponse.model_json_schema(), indent=2)}"
)


def infer_dataset_context(profile: DatasetProfile, metadata: str = "") -> DatasetProfile:
    """Fill per-dataset contextual fields (grain, temporal_coverage, description)."""
    user_prompt = f"Dataset profile:\n{profile.compact()}\n"
    if metadata:
        user_prompt += f"\nUser-provided metadata:\n{metadata}\n"

    logger.info("Inferring dataset context via LLM...")
    result = call_llm(
        system_prompt=_DATASET_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_model=DatasetContextResponse,
    )

    profile.grain = result["grain"]
    profile.temporal_coverage = result.get("temporal_coverage")
    profile.description = result["description"]

    logger.info("Grain: %s", profile.grain)
    return profile


_COLLECTION_SYSTEM_PROMPT = (
    "You are a data analyst. Below are profiles for multiple datasets in a "
    "collection. Infer the combined domain, a brief description, seed questions "
    "that can be answered using these datasets, and ideas for cross-dataset "
    "exploration. Respond with valid JSON only.\n\n"
    f"JSON Schema:\n{json.dumps(CollectionResponse.model_json_schema(), indent=2)}"
)


def infer_collection_insight(profiles: list[DatasetProfile]) -> CollectionInsight:
    """Produce a CollectionInsight spanning all datasets via a single LLM call."""
    parts = []
    for p in profiles:
        parts.append(f"=== {p.name} ===\n{p.compact()}\n")
    user_prompt = "\n".join(parts)

    logger.info("Inferring collection insight via LLM (%d datasets)...", len(profiles))
    result = call_llm(
        system_prompt=_COLLECTION_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_model=CollectionResponse,
    )

    insight = CollectionInsight(
        domain=result["domain"],
        description=result["description"],
        domain_knowledge=result["domain_knowledge"],
        seed_questions=result["seed_questions"],
        exploration_ideas=result["exploration_ideas"],
    )

    logger.info(
        "Collection domain: %s  Knowledge: %d items  Questions: %d  Ideas: %d",
        insight.domain,
        len(insight.domain_knowledge),
        len(insight.seed_questions),
        len(insight.exploration_ideas),
    )
    return insight

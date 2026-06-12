import json
from pydantic import BaseModel, Field
from config import logger
from llm_client import call_llm
from models.schemas import DatasetProfile


class _DomainResponse(BaseModel):
    domain: str = Field(description="Industry or field the dataset belongs to")
    grain: str = Field(description="What a single row represents")
    temporal_coverage: str | None = Field(
        default=None,
        description="Date range covered if datetime columns exist, e.g. '2020-01 – 2024-12'",
    )
    description: str = Field(
        description="Brief 2-3 sentence explanation of what the dataset is about"
    )
    seed_questions: list[str] = Field(
        description="3-4 analytical questions worth exploring"
    )


_SYSTEM_PROMPT = (
    "You are a data analyst. Given a dataset profile below, infer the following "
    "about the dataset and respond with valid JSON only.\n\n"
    f"JSON Schema:\n{json.dumps(_DomainResponse.model_json_schema(), indent=2)}"
)


def infer_domain(profile: DatasetProfile, metadata: str = "") -> DatasetProfile:
    """Fill domain-specific fields in a DatasetProfile via a single LLM call."""
    user_prompt = f"Dataset profile:\n{profile.compact()}\n"
    if metadata:
        user_prompt += f"\nUser-provided metadata:\n{metadata}\n"

    logger.info("Inferring domain via LLM...")
    result = call_llm(
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_model=_DomainResponse,
    )

    profile.domain = result["domain"]
    profile.grain = result["grain"]
    profile.temporal_coverage = result.get("temporal_coverage")
    profile.description = result["description"]
    profile.seed_questions = result["seed_questions"]

    logger.info(
        "Domain: %s  Grain: %s  Questions: %d",
        profile.domain,
        profile.grain,
        len(profile.seed_questions),
    )

    return profile

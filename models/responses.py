from pydantic import BaseModel, Field


class DatasetContextResponse(BaseModel):
    grain: str = Field(description="What a single row represents")
    temporal_coverage: str | None = Field(
        default=None,
        description="Date range covered if datetime columns exist, e.g. '2020-01 – 2024-12'",
    )
    description: str = Field(
        description="Brief 2-3 sentence explanation of what the dataset is about"
    )


class CollectionResponse(BaseModel):
    domain: str = Field(description="Overall domain or industry spanning all datasets")
    description: str = Field(
        description="Brief 2-3 sentence explanation of what the combined dataset collection represents"
    )
    domain_knowledge: list[str] = Field(
        description="Useful context explaining specific terms and jargon found in the datasets"
    )
    seed_questions: list[str] = Field(
        description="3-4 analytical questions answerable across one or more of the datasets"
    )
    exploration_ideas: list[str] = Field(
        description="2-3 ideas for cross-dataset exploration or joining"
    )

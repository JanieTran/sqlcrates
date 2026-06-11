import os
import logging
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("sqlcrates")
_handler = logging.StreamHandler()
_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s")
)
logger.addHandler(_handler)
logger.setLevel(logging.INFO)


@dataclass
class Settings:
    api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    model: str = os.getenv("MODEL", "openrouter/free")
    max_insights: int = int(os.getenv("MAX_INSIGHTS", "10"))
    max_follow_up_depth: int = int(os.getenv("MAX_FOLLOW_UP_DEPTH", "2"))


settings = Settings()

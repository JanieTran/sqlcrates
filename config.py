import os
import logging
from dataclasses import dataclass, field

import duckdb
from dotenv import load_dotenv

# Load environment variables from .env (optional, won't error if file is missing)
load_dotenv()

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
# ANSI colour codes — pastel palette
_RESET = "\033[0m"
_YELLOW = "\033[38;2;255;229;153m"    # pastel yellow 
_WHITE = "\033[38;2;229;229;229m"    # pastel white
_CYAN = "\033[38;2;173;216;230m"     # pastel cyan 
_PINK = "\033[38;2;255;182;193m"     # pastel pink

class _ColouredFormatter(logging.Formatter):
    def format(self, record):
        timestamp = self.formatTime(record)
        level = record.levelname
        name = record.name
        lineno = record.lineno
        msg = record.getMessage()
        return (
            f"{_CYAN}{timestamp}{_RESET} [{_WHITE}{level}{_RESET}] "
            f"{_YELLOW}{name}{_RESET}:{_PINK}{lineno}{_RESET}: {msg}"
        )

logger = logging.getLogger("sqlcrates")
_handler = logging.StreamHandler()
_handler.setFormatter(
    _ColouredFormatter("%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s")
)
logger.addHandler(_handler)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Settings — read from environment variables
# ---------------------------------------------------------------------------
@dataclass
class Settings:
    # OpenRouter API key (required for LLM calls)
    api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    # Model identifier passed in the OpenRouter request body
    model: str = os.getenv("MODEL", "openrouter/free")
    # Maximum number of insight cards to generate per exploration run
    max_insights: int = int(os.getenv("MAX_INSIGHTS", "10"))
    # Maximum depth of follow-up questions per thread
    max_follow_up_depth: int = int(os.getenv("MAX_FOLLOW_UP_DEPTH", "2"))


settings = Settings()

# ---------------------------------------------------------------------------
# DuckDB — single in-memory connection shared across all modules
# ---------------------------------------------------------------------------
conn = duckdb.connect()

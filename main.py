#!/usr/bin/env python3
from __future__ import annotations

import argparse

from config import logger
from ingestion.loader import discover_datasets
from agents.domain_agent import infer_collection_insight
from models.schemas import CollectionInsight, DatasetProfile


def cmd_explore(profiles: list[DatasetProfile], insight: CollectionInsight):
    logger.info("---DATASET PROFILES")
    for prof in profiles:
        logger.info(
            "Dataset: %s\n"
            "Grain: %s\n"
            "Description: %s\n"
            "Temporal coverage: %s",
            prof.name,
            prof.grain,
            prof.description,
            prof.temporal_coverage,
        )
    logger.info(
        "---COLLECTION SUMMARY\n"
        "- Domain: %s\n"
        "- Description: %s\n"
        "- Domain knowledge:\n\t%s\n"
        "- Seed questions:\n\t%s\n"
        "- Exploration ideas:\n\t%s",
        insight.domain,
        insight.description,
        "\n\t".join(insight.domain_knowledge),
        "\n\t".join(insight.seed_questions),
        "\n\t".join(insight.exploration_ideas),
    )


def cmd_chat(prof: DatasetProfile, insight: CollectionInsight):
    logger.warning("Chat mode not yet implemented")


def main(args):
    datasets = discover_datasets()
    if not datasets:
        return

    if args.command == "explore":
        insight = infer_collection_insight(datasets)
        cmd_explore(datasets, insight)
    elif args.command == "chat":
        insight = infer_collection_insight(datasets)
        cmd_chat(datasets[0], insight)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="sqlcrates",
        description="AI-powered dataset exploration tool",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("explore", help="Run full autonomous exploration on all datasets in data/")
    sub.add_parser("chat", help="Interactive Q&A session with the first dataset in data/")

    args = parser.parse_args()
    main(args)

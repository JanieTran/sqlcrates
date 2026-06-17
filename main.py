#!/usr/bin/env python3
from __future__ import annotations

import argparse

from config import logger
from ingestion.loader import discover_datasets
from models.schemas import DatasetProfile


def cmd_explore(profiles: list[DatasetProfile]):
    for prof in profiles:
        logger.info(
            "Domain inference completed\n"
            "Dataset: %s\n"
            "Domain: %s\n"
            "Grain: %s\n"
            "Description: %s\n"
            "Seed questions:\n\t%s",
            prof.name,
            prof.domain,
            prof.grain,
            prof.description,
            "\n\t".join(prof.seed_questions),
        )


def cmd_chat(prof: DatasetProfile):
    logger.warning("Chat mode not yet implemented")


def main(args):
    datasets = discover_datasets()
    if not datasets:
        return

    if args.command == "explore":
        cmd_explore(datasets)
    elif args.command == "chat":
        cmd_chat(datasets[0])


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

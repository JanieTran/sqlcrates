#!/usr/bin/env python3
from __future__ import annotations

import argparse

from config import logger
from agents.cache import load_run, save_run
from agents.domain_agent import infer_collection_insight
from agents.report_agent import interpret_results, write_summary
from agents.sql_agent import generate_sql
from ingestion.loader import discover_datasets, register_tables
from tools.sql_executor import execute_sql
from models.schemas import CollectionInsight, DatasetProfile, InsightCard


def cmd_explore(profiles: list[DatasetProfile], insight: CollectionInsight):
    if not insight.seed_questions:
        write_summary(profiles, insight)
        return

    cards: list[InsightCard] = []

    for idx, q in enumerate(insight.seed_questions, start=1):
        logger.info("--- Question %d/%d", idx, len(insight.seed_questions))
        logger.info("Question: %s", q)

        sql_resp = generate_sql(q, profiles, insight)
        logger.info("Explanation: %s", sql_resp.explanation)

        all_results: list[dict] = []
        for i, query in enumerate(sql_resp.queries, start=1):
            logger.info("Executing query %d/%d", i, len(sql_resp.queries))
            result = execute_sql(query)
            if result["error"]:
                logger.error("Query %d failed: %s", i, result["error"])
            else:
                logger.info("Query %d returned %d rows", i, result["row_count"])
            all_results.append(result)

        card = interpret_results(q, sql_resp.explanation, all_results, profiles, insight)
        cards.append(card)

    write_summary(profiles, insight, cards=cards)


def cmd_chat(prof: DatasetProfile, insight: CollectionInsight):
    logger.warning("Chat mode not yet implemented")


def main(args):
    cached = load_run()
    if cached:
        datasets, insight = cached
        if not register_tables():
            return
        logger.info("Using cached profiles + insight")
    else:
        datasets = discover_datasets()
        if not datasets:
            return
        insight = infer_collection_insight(datasets)
        save_run(datasets, insight)

    if args.command == "explore":
        cmd_explore(datasets, insight)
    elif args.command == "chat":
        pass


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

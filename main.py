#!/usr/bin/env python3
from __future__ import annotations

import argparse

from config import logger
from ingestion.loader import discover_datasets
from agents.domain_agent import infer_collection_insight
from agents.report_agent import write_summary
from agents.sql_agent import generate_sql
from tools.sql_executor import display_query_results, execute_sql
from models.schemas import CollectionInsight, DatasetProfile


def cmd_explore(profiles: list[DatasetProfile], insight: CollectionInsight):
    write_summary(profiles, insight)

    if not insight.seed_questions:
        return

    logger.info("---TEST SQL GENERATION WITH FIRST SEED QUESTION")
    q = insight.seed_questions[0]

    logger.info("Question: %s", q)
    sql_resp = generate_sql(q, profiles, insight)
    logger.info("Explanation: %s", sql_resp.explanation)

    for i, query in enumerate(sql_resp.queries):
        logger.info("Executing query %d/%d", i + 1, len(sql_resp.queries))
        result = execute_sql(query)
        if result["error"]:
            logger.error("Query %d failed: %s", i + 1, result["error"])
        else:
            logger.info(
                "Query %d returned %d rows",
                i + 1,
                result["row_count"]
            )
            display_query_results(result["columns"], result["rows"])


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

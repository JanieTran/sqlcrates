#!/usr/bin/env python3
import argparse
from config import logger
from dataset.loader import load_csv, read_metadata
from dataset.profiler import profile
from agents.domain_agent import infer_domain


def cmd_explore(prof, meta):
    prof = infer_domain(prof, meta)
    logger.info(
        "Domain inference completed\n" \
        "Domain: %s \nGrain: %s \nDescription: %s\nSeed questions:\n\t%s",
        prof.domain,
        prof.grain,
        prof.description,
        '\n\t'.join(prof.seed_questions),
    )


def cmd_ask(prof, meta, question):
    logger.warning("QA agent not yet implemented")
    print(f"Question: {question}\n(not implemented)")


def cmd_chat(prof, meta):
    logger.warning("Chat mode not yet implemented")
    print("Chat mode not yet implemented")


def main(args):
    df = load_csv(args.csv)
    meta = read_metadata(args.metadata)
    prof = profile(df)

    if args.command == "explore":
        cmd_explore(prof, meta)
    elif args.command == "ask":
        cmd_ask(prof, meta, args.question)
    elif args.command == "chat":
        cmd_chat(prof, meta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="sqlcrates",
        description="AI-powered dataset exploration tool",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("explore", help="Run full autonomous exploration")
    p.add_argument("csv", help="Path to CSV file")
    p.add_argument("--metadata", "-m", help="Path to metadata .txt file")

    p = sub.add_parser("ask", help="Answer a single question about the dataset")
    p.add_argument("csv", help="Path to CSV file")
    p.add_argument("question", help="Natural language question")
    p.add_argument("--metadata", "-m", help="Path to metadata .txt file")

    p = sub.add_parser("chat", help="Interactive Q&A session")
    p.add_argument("csv", help="Path to CSV file")
    p.add_argument("--metadata", "-m", help="Path to metadata .txt file")

    args = parser.parse_args()
    main(args)

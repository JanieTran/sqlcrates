#!/usr/bin/env python3
import argparse

from config import logger
from dataset.loader import load_csv, read_metadata
from dataset.profiler import profile


def cmd_explore():
    pass


def cmd_ask():
    pass


def cmd_chat():
    pass


def main(args):
    df = load_csv(args.csv)
    meta = read_metadata(args.metadata)
    prof = profile(df)
    logger.info(prof)
    
    if args.command == "explore":
        cmd_explore()
    elif args.command == "ask":
        cmd_ask()
    elif args.command == "chat":
        cmd_chat()


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

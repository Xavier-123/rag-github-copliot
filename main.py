"""
Main entry point for the Agentic RAG + Self-Improving RAG system.

Provides a command-line interface with two modes:
- ``chat``      – interactive multi-turn conversation loop
- ``ingest``    – add documents to the knowledge base from a file

Usage
-----
    python main.py chat
    python main.py ingest --file path/to/documents.jsonl
    python main.py --help
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from config.settings import settings
from pipeline.rag_pipeline import RAGPipeline
from utils.helpers import get_logger


logger = get_logger("main")


def run_chat(pipeline: RAGPipeline, session_id: str = "cli-session") -> None:
    """
    Run an interactive chat loop in the terminal.

    The session maintains conversation history across turns so the model
    can resolve co-references and follow-up questions.

    Type ``/quit`` or ``/exit`` to exit.  Type ``/status`` to view the
    system optimizer status.

    Args:
        pipeline:   Initialised :class:`RAGPipeline` instance.
        session_id: Session identifier (default: ``"cli-session"``).
    """
    print("\n" + "=" * 60)
    print(" Agentic RAG System  (type /quit to exit, /status for info)")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting. Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit"):
            print("Goodbye!")
            break

        if user_input.lower() == "/status":
            status = pipeline.get_optimizer_status()
            print(f"[System Status] {json.dumps(status, indent=2)}")
            continue

        response = pipeline.chat(user_input, session_id=session_id)

        print(f"\nAssistant: {response.answer}")
        if response.sources:
            print(f"Sources: {', '.join(response.sources)}")
        if response.evaluation:
            scores = response.evaluation
            print(
                f"[Eval] overall={scores.get('overall', 0):.2f} "
                f"faithfulness={scores.get('faithfulness', 0):.2f} "
                f"relevance={scores.get('answer_relevance', 0):.2f}"
            )
        print(f"[confidence={response.confidence:.2f}, elapsed={response.elapsed_ms:.0f}ms]\n")


def run_ingest(pipeline: RAGPipeline, file_path: str) -> None:
    """
    Ingest documents from a JSONL file into the knowledge base.

    Each line of the file should be a JSON object with at minimum a
    ``"content"`` field.  Optional fields: ``"source"``, ``"metadata"``.

    Args:
        pipeline:  Initialised :class:`RAGPipeline` instance.
        file_path: Path to the JSONL document file.
    """
    path = Path(file_path)
    if not path.exists():
        logger.error("File not found: %s", file_path)
        sys.exit(1)

    documents = []
    with path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
                documents.append(doc)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping invalid JSON on line %d: %s", i, exc)

    if not documents:
        logger.warning("No valid documents found in %s", file_path)
        return

    count = pipeline.add_documents(documents)
    print(f"Ingested {count} document chunks from {len(documents)} source documents.")


def main() -> None:
    """Parse CLI arguments and run the appropriate command."""
    parser = argparse.ArgumentParser(
        description="Agentic RAG + Self-Improving RAG system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command")

    # chat command
    chat_parser = sub.add_parser("chat", help="Start an interactive chat session")
    chat_parser.add_argument(
        "--session", default="cli-session", help="Session ID (default: cli-session)"
    )

    # ingest command
    ingest_parser = sub.add_parser("ingest", help="Ingest documents into the knowledge base")
    ingest_parser.add_argument(
        "--file", required=True, help="Path to JSONL document file"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    logger.info("Initialising RAGPipeline …")
    pipeline = RAGPipeline()

    if args.command == "chat":
        run_chat(pipeline, session_id=args.session)
    elif args.command == "ingest":
        run_ingest(pipeline, file_path=args.file)


if __name__ == "__main__":
    main()

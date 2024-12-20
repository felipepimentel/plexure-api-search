"""Index command."""

import logging
import sys
from pathlib import Path
from typing import List

import click
from rich.console import Console

from ...indexing import indexer

logger = logging.getLogger(__name__)
console = Console()

@click.command()
@click.option(
    "--clear",
    is_flag=True,
    help="Clear existing index before indexing",
)
def index(clear: bool) -> None:
    """Index API contracts."""
    try:
        # Clear index if requested
        if clear:
            logger.info("Clearing existing index")
            indexer.clear()

        # Find API files
        api_dir = Path("assets/apis")
        if not api_dir.exists():
            console.print(f"[red]Error:[/red] API directory not found: {api_dir}")
            sys.exit(1)

        api_files = list(api_dir.glob("*.yaml"))
        if not api_files:
            console.print(f"[red]Error:[/red] No API files found in {api_dir}")
            sys.exit(1)

        logger.info(f"Found {len(api_files)} API files")

        # Index each file
        for file_path in api_files:
            try:
                indexer.index_contract(str(file_path))
            except Exception as e:
                logger.error(f"Failed to index {file_path}: {e}")
                continue

        logger.info("Indexing completed successfully")

    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        console.print(f"[red]Error:[/red] {str(e)}")
        sys.exit(1)
    finally:
        # Clean up
        indexer.cleanup()

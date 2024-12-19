"""Index command."""

import logging
import os
from pathlib import Path
from typing import Optional

import click

from ...indexing import api_indexer
from ...utils.file import find_api_files

logger = logging.getLogger(__name__)

@click.command()
@click.option(
    "--directory",
    "-d",
    help="Directory containing API contracts",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
)
@click.option(
    "--clear",
    "-c",
    is_flag=True,
    help="Clear existing index before indexing",
)
def index(directory: Optional[str] = None, clear: bool = False) -> None:
    """Index API contracts."""
    try:
        # Initialize indexer
        api_indexer.initialize()

        # Clear index if requested
        if clear:
            logger.info("Clearing existing index")
            api_indexer.clear()

        # Find API files
        files = find_api_files(directory)
        if not files:
            logger.warning("No API files found")
            return

        logger.info(f"Found {len(files)} API files")
        for file in files:
            logger.debug(f"Found API file: {file}")

        # Index contracts
        api_indexer.index_contracts(files)
        logger.info("Indexing completed successfully")

    except Exception as e:
        logger.error(f"Failed to index: {e}")
        raise click.ClickException(str(e))

    finally:
        api_indexer.cleanup()

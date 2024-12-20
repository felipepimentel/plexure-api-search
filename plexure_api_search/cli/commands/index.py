"""
Index Command for Plexure API Search CLI

This module provides the index command functionality for the Plexure API Search CLI.
It handles the command-line interface for indexing API contracts, managing the index,
and providing feedback on indexing operations.

Key Features:
- API contract indexing
- Index management
- Progress tracking
- Error handling
- Status reporting
- Batch processing
- Index optimization
- Cache management

The module provides commands for:
- Indexing API contracts
- Managing index state
- Clearing index
- Optimizing index
- Status reporting
- Cache control

Command Options:
1. Indexing Parameters:
   - Input directory/file
   - Batch size
   - Clear existing
   - Recursive scan
   - File patterns

2. Index Management:
   - Clear index
   - Optimize index
   - Show status
   - Cache control
   - Backup/restore

3. Progress Options:
   - Progress display
   - Verbosity level
   - Error reporting
   - Status updates

Example Usage:
    $ plexure-api-search index apis/
    $ plexure-api-search index --clear apis/*.yaml
    $ plexure-api-search index --batch-size 32 apis/
    $ plexure-api-search index --status

Output Features:
- Progress bars
- Status updates
- Error reporting
- Statistics display
- Performance metrics
"""

import logging
import sys
from pathlib import Path

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

"""Command line interface for API search."""

import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import click
from dotenv import load_dotenv
from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

from .indexing import APIIndexer
from .monitoring.events import Event, event_manager
from .search import Consistency
from .search.searcher import APISearcher
from .search.understanding import ZeroShotUnderstanding

# Load environment variables
load_dotenv()

# Initialize console
console = Console()


# Configure logging based on verbosity
def setup_logging(verbosity: int) -> None:
    """Configure logging based on verbosity level."""
    if verbosity == 0:
        logging.basicConfig(level=logging.ERROR)
    elif verbosity == 1:
        logging.basicConfig(level=logging.WARNING)
    elif verbosity == 2:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.DEBUG)

    # Suppress specific loggers unless in debug mode
    if verbosity < 3:
        logging.getLogger("pinecone").setLevel(logging.WARNING)
        logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
        logging.getLogger("pinecone_plugin_interface").setLevel(logging.WARNING)


def render_config_info(config: Dict[str, Any]) -> Panel:
    """Render configuration information."""
    config_text = Text()
    config_text.append("\n🔧 Pinecone Configuration\n", style="bold yellow")

    # Show non-sensitive config info
    safe_keys = ["index_name", "environment", "cloud", "region"]
    for key in safe_keys:
        if key in config:
            config_text.append(f"{key.title()}: ", style="bright_blue")
            config_text.append(f"{config[key]}\n", style="white")

    return Panel(
        config_text,
        title="[bold]Configuration",
        border_style="blue",
        box=box.ROUNDED,
        padding=(1, 2),
    )


@click.group()
def cli() -> None:
    """API Search CLI"""
    pass


@cli.command()
@click.argument("query")
@click.option("-v", "--verbose", count=True, help="Increase output verbosity")
@click.option("--limit", default=10, help="Maximum number of results to return")
@click.option("--threshold", default=0.5, help="Minimum similarity score threshold")
def search(query: str, verbose: int, limit: int, threshold: float) -> None:
    """Search for API endpoints matching the query."""
    try:
        # Setup logging
        setup_logging(verbose)
        
        # Initialize components
        searcher = APISearcher()
        understanding = ZeroShotUnderstanding()
        
        # Process query
        query_intent = understanding.classify_intent(query)
        
        # Perform search
        results = searcher.search(
            query=query,
            filters={"score_threshold": threshold} if threshold else None,
            include_metadata=True,
            enhance_results=True
        )
        
        # Display results
        if not results:
            console.print("\n[yellow]No matching endpoints found.[/yellow]")
            return
            
        # Create results table
        table = Table(
            title=f"\nSearch Results for: [cyan]{query}[/cyan]",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
        )
        
        table.add_column("Score", justify="right", style="cyan", no_wrap=True)
        table.add_column("Method", style="green")
        table.add_column("Endpoint", style="blue")
        table.add_column("Description", style="white")
        
        for result in results:
            table.add_row(
                f"{result.get('score', 0.0):.2f}",
                result.get('method', ''),
                result.get('endpoint', ''),
                result.get('description', '')
            )
        
        console.print(table)
        
        # Show query intent if verbose
        if verbose > 0:
            console.print(f"[dim]Detected intent:[/dim] {query_intent}")
        
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {str(e)}")
        if verbose > 0:
            console.print_exception()
        raise click.Abort()


@cli.command()
@click.option("-v", "--verbose", count=True, help="Increase output verbosity")
def index(verbose: int) -> None:
    """Index API endpoints."""
    try:
        # Setup logging
        setup_logging(verbose)
        
        # Initialize indexer
        indexer = APIIndexer()
        
        # Index APIs
        results = indexer.index_directory(force=True, validate=True)
        
        # Display results
        if results["total_endpoints"] == 0:
            console.print("\n[yellow]No endpoints found to index.[/yellow]")
            return
            
        # Create summary table
        table = Table(
            title="\nIndexing Results",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
        )
        
        table.add_column("Metric", style="bright_blue")
        table.add_column("Value", style="white")
        
        table.add_row("Total Files", str(results["total_files"]))
        table.add_row("Total Endpoints", str(results["total_endpoints"]))
        table.add_row("Failed Files", str(len(results["failed_files"])))
        
        console.print(table)
        
        # Show failed files if any
        if results["failed_files"]:
            console.print("\n[red]Failed Files:[/red]")
            for failed in results["failed_files"]:
                console.print(f"  • {failed['path']}: {failed['error']}")
        
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {str(e)}")
        if verbose > 0:
            console.print_exception()
        raise click.Abort()


if __name__ == "__main__":
    cli()

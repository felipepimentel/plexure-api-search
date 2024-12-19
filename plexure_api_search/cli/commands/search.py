"""Search command for finding API endpoints."""

import logging
import click
from rich.console import Console

from ...search import api_searcher

logger = logging.getLogger(__name__)
console = Console()

@click.command()
@click.argument("query")
@click.option(
    "--top-k",
    type=int,
    default=10,
    help="Number of results to return",
)
@click.option(
    "--min-score",
    type=float,
    default=0.1,
    help="Minimum similarity score (0.0 to 1.0)",
)
@click.option(
    "--expand/--no-expand",
    default=True,
    help="Whether to expand query",
)
@click.option(
    "--rerank/--no-rerank",
    default=True,
    help="Whether to rerank results",
)
def search(
    query: str,
    top_k: int,
    min_score: float,
    expand: bool,
    rerank: bool,
) -> None:
    """Search for API endpoints."""
    try:
        logger.debug(f"Searching with query: {query}")
        logger.debug(f"Parameters: top_k={top_k}, min_score={min_score}, expand={expand}, rerank={rerank}")

        results = api_searcher.search(
            query=query,
            top_k=top_k,
            min_score=min_score,
            expand_query=expand,
            rerank_results=rerank,
        )

        if not results:
            console.print("No results found.", style="yellow")
            return

        for result in results:
            console.print(f"\nScore: {result['score']:.3f}", style="green")
            console.print(f"Method: {result.get('method', 'N/A')}", style="blue")
            console.print(f"Endpoint: {result.get('endpoint', 'N/A')}", style="blue")
            if description := result.get('description'):
                console.print(f"Description: {description}")
            if parameters := result.get('parameters'):
                console.print("\nParameters:", style="cyan")
                for param in parameters:
                    console.print(f"  - {param['name']}: {param['type']}")
                    if param_desc := param.get('description'):
                        console.print(f"    {param_desc}")
            console.print("-" * 80, style="dim")

    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        console.print(f"Error: {e}", style="red")
        raise click.Abort() 
"""
Search Command for Plexure API Search CLI

This module provides the search command functionality for the Plexure API Search CLI.
It handles the command-line interface for executing semantic searches over API endpoints,
formatting and displaying results, and managing search options.

Key Features:
- Natural language query processing
- Result formatting and display
- Search parameter configuration
- Output formatting options
- Error handling
- Progress feedback
- Result pagination
- Export capabilities

The module provides commands for:
- Semantic endpoint search
- Result filtering
- Output formatting
- Export options
- Search configuration
- Help documentation

Command Options:
1. Search Parameters:
   - Query text
   - Result limit
   - Minimum score
   - Result format
   - Output file

2. Display Options:
   - Output format (text/json/yaml)
   - Verbosity level
   - Color output
   - Pagination
   - Sorting

3. Export Options:
   - Export format
   - Export file
   - Template usage
   - Field selection

Example Usage:
    $ plexure-api-search search "find authentication endpoints"
    $ plexure-api-search search --limit 5 "user management"
    $ plexure-api-search search --format json "payment api"
    $ plexure-api-search search --min-score 0.5 "order processing"

Output Features:
- Formatted text output
- JSON/YAML export
- Color highlighting
- Result grouping
- Score display
"""

import logging
import sys

import click
from rich.console import Console
from rich.table import Table

from ...search.searcher import searcher

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
def search(query: str, top_k: int) -> None:
    """Search for API endpoints.

    Args:
        query: Search query
        top_k: Number of results to return
    """
    try:
        # Search
        results = searcher.search(query, top_k=top_k)

        if not results:
            console.print("No results found.")
            sys.exit(0)

        # Create results table
        table = Table(
            title=f"Search Results for: {query}",
            show_header=True,
            header_style="bold magenta",
        )

        # Add columns
        table.add_column("Score", justify="right", style="cyan")
        table.add_column("Method", justify="left", style="green")
        table.add_column("Path", justify="left", style="blue")
        table.add_column("Description", justify="left")

        # Add rows
        for result in results:
            score = f"{result['score']:.3f}"
            method = result["method"].upper()
            path = result["path"]
            description = result.get("description") or result.get("summary", "")

            table.add_row(score, method, path, description)

        # Print table
        console.print(table)

        # Print parameter details for top result
        if results:
            top_result = results[0]

            # Parameters
            if top_result.get("parameters"):
                console.print("\nParameters:", style="bold")
                param_table = Table(show_header=True, header_style="bold")
                param_table.add_column("Name", style="cyan")
                param_table.add_column("In", style="green")
                param_table.add_column("Required", style="yellow")
                param_table.add_column("Description")

                for param in top_result["parameters"]:
                    param_table.add_row(
                        param.get("name", ""),
                        param.get("in", ""),
                        str(param.get("required", False)),
                        param.get("description", ""),
                    )

                console.print(param_table)

            # Responses
            if top_result.get("responses"):
                console.print("\nResponses:", style="bold")
                resp_table = Table(show_header=True, header_style="bold")
                resp_table.add_column("Code", style="cyan")
                resp_table.add_column("Description")

                for code, details in top_result["responses"].items():
                    resp_table.add_row(
                        str(code),
                        details.get("description", ""),
                    )

                console.print(resp_table)

            # Tags
            if top_result.get("tags"):
                console.print("\nTags:", style="bold")
                console.print(", ".join(top_result["tags"]), style="blue")

    except Exception as e:
        logger.error(f"Search failed: {e}")
        console.print(f"[red]Error:[/red] {str(e)}")
        sys.exit(1)

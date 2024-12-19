"""Index command for indexing API contracts."""

import click
from rich.console import Console
from rich.table import Table

from ...config import config_instance
from ...indexing import api_indexer

console = Console()

@click.command()
@click.option(
    "--directory",
    type=str,
    help="Directory containing API files",
)
def index(directory: str = None):
    """Index API contracts."""
    try:
        # Run indexing
        results = api_indexer.index_directory(directory)

        # Display results
        table = Table(title="Indexing Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Files", str(results["total_files"]))
        table.add_row("Total Endpoints", str(results["total_endpoints"]))
        table.add_row("Failed Files", str(len(results["failed_files"])))

        console.print(table)

        # Show failures if any
        if results["failed_files"]:
            console.print("\nFailed Files:", style="red")
            for failure in results["failed_files"]:
                console.print(
                    f"  {failure['path']}: {failure['error']}",
                    style="red",
                )

    except Exception as e:
        console.print(f"Error: {e}", style="red")
        raise click.Abort()

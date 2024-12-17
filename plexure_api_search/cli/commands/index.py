"""Index command implementation."""

import click
from rich.console import Console
from rich.table import Table
from rich import box

from ...indexing import APIIndexer
from ..ui.logging import setup_logging

console = Console()

@click.command()
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
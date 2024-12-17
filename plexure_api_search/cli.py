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
from rich.logging import RichHandler
from rich.columns import Columns

from .indexing import APIIndexer
from .monitoring.events import Event, event_manager, EventType
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


@cli.command()
@click.option("-v", "--verbose", count=True, help="Increase output verbosity")
@click.option("--interval", default=2, help="Monitoring interval in seconds")
def monitor(verbose: int, interval: int) -> None:
    """Monitor system metrics and performance with live updates."""
    try:
        # Setup logging with rich handler
        logging.basicConfig(
            level=logging.DEBUG if verbose > 1 else logging.INFO,
            format="%(message)s",
            handlers=[RichHandler(rich_tracebacks=True, markup=True)]
        )
        
        # Initialize event manager monitoring
        event_manager.start_monitoring()
        
        def make_metrics_panel() -> Panel:
            """Create metrics panel with current stats."""
            events = event_manager.get_recent_events()
            
            metrics_table = Table(
                show_header=True,
                header_style="bold magenta",
                box=box.ROUNDED,
                title="System Metrics",
                padding=(0, 1),
                expand=True
            )
            
            metrics_table.add_column("Metric", style="cyan")
            metrics_table.add_column("Value", style="green", justify="right")
            
            # Add core metrics
            metrics = {
                "Total Events": len(events),
                "Search Events": len([e for e in events if e.type == EventType.SEARCH_STARTED]),
                "Error Events": len([e for e in events if not e.success]),
                "Cache Hit Rate": f"{event_manager.get_cache_hit_rate():.2%}",
                "Avg Search Time": f"{event_manager.get_average_search_time():.3f}s",
                "Active Models": event_manager.get_active_model_count(),
            }
            
            for metric, value in metrics.items():
                metrics_table.add_row(metric, str(value))
                
            return Panel(metrics_table, border_style="blue", expand=True)
            
        def make_events_panel() -> Panel:
            """Create events panel with recent activity."""
            events = event_manager.get_recent_events(minutes=2)  # Last 2 minutes
            
            if not events:
                return Panel(
                    "[dim]No recent events[/dim]",
                    title="Recent Events",
                    border_style="yellow",
                    expand=True
                )
            
            events_table = Table(
                show_header=True,
                header_style="bold yellow",
                box=box.ROUNDED,
                title="Recent Events",
                padding=(0, 1),
                expand=True
            )
            
            events_table.add_column("Time", style="cyan", width=8)
            events_table.add_column("Type", style="magenta")
            events_table.add_column("Component", style="blue")
            events_table.add_column("Status", style="green", width=6)
            events_table.add_column("Description", style="white")
            
            for event in reversed(events[-10:]):  # Show last 10 events
                status = "✓" if event.success else "✗"
                status_style = "green" if event.success else "red"
                
                events_table.add_row(
                    event.timestamp.strftime("%H:%M:%S"),
                    event.type.name.replace("_", " ").title(),
                    event.component,
                    f"[{status_style}]{status}[/{status_style}]",
                    str(event.description or "")
                )
            
            return Panel(events_table, border_style="yellow", expand=True)
            
        def make_error_panel() -> Panel:
            """Create error panel showing recent errors."""
            events = [e for e in event_manager.get_recent_events() if not e.success]
            
            if not events:
                return Panel(
                    "[dim]No errors[/dim]",
                    title="Recent Errors",
                    border_style="red",
                    expand=True
                )
            
            error_table = Table(
                show_header=True,
                header_style="bold red",
                box=box.ROUNDED,
                title="Recent Errors",
                padding=(0, 1),
                expand=True
            )
            
            error_table.add_column("Time", style="cyan", width=8)
            error_table.add_column("Component", style="blue")
            error_table.add_column("Error", style="red")
            
            for event in reversed(events[-5:]):  # Show last 5 errors
                error_table.add_row(
                    event.timestamp.strftime("%H:%M:%S"),
                    event.component,
                    str(event.error or "Unknown error")
                )
            
            return Panel(error_table, border_style="red", expand=True)
        
        # Create layout
        layout = Layout()
        
        # Vertical layout
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="metrics", size=10),
            Layout(name="events", ratio=2),
            Layout(name="errors", size=10),
            Layout(name="footer", size=3)
        )
        
        # Start live display
        with Live(layout, refresh_per_second=1/interval, screen=True) as live:
            try:
                while True:
                    # Update header
                    layout["header"].update(
                        Panel(
                            f"[bold green]System Monitor[/bold green] - Refresh: {interval}s - Press Ctrl+C to exit",
                            style="bold white on blue",
                            expand=True
                        )
                    )
                    
                    # Update main panels
                    layout["metrics"].update(make_metrics_panel())
                    layout["events"].update(make_events_panel())
                    layout["errors"].update(make_error_panel())
                    
                    # Update footer with timestamp
                    layout["footer"].update(
                        Panel(
                            f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                            style="bold white on blue",
                            expand=True
                        )
                    )
                    
                    # Emit monitoring update event
                    event_manager.emit(Event(
                        type=EventType.MONITORING_UPDATED,
                        timestamp=datetime.now(),
                        component="monitoring",
                        description="Updated monitoring display"
                    ))
                    
                    time.sleep(interval)
                    
            except KeyboardInterrupt:
                event_manager.stop_monitoring()
                console.print("\n[yellow]Monitoring stopped.[/yellow]")
                
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {str(e)}")
        if verbose > 0:
            console.print_exception()
        raise click.Abort()


if __name__ == "__main__":
    cli()

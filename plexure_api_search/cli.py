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
        
        # Emit search started event
        start_time = time.time()
        event_manager.emit(Event(
            type=EventType.SEARCH_STARTED,
            timestamp=datetime.now(),
            component="search",
            description=f"Starting search for query: {query}"
        ))
        
        try:
            # Process query
            query_intent = understanding.classify_intent(query)
            event_manager.emit(Event(
                type=EventType.SEARCH_QUERY_PROCESSED,
                timestamp=datetime.now(),
                component="search",
                description=f"Query intent: {query_intent}",
                metadata={"intent": query_intent}
            ))
            
            # Perform search
            results = searcher.search(
                query=query,
                filters={"score_threshold": threshold} if threshold else None,
                include_metadata=True,
                enhance_results=True
            )
            
            # Calculate search duration
            duration_ms = (time.time() - start_time) * 1000
            
            if results:
                event_manager.emit(Event(
                    type=EventType.SEARCH_RESULTS_FOUND,
                    timestamp=datetime.now(),
                    component="search",
                    description=f"Found {len(results)} results",
                    duration_ms=duration_ms,
                    metadata={
                        "result_count": len(results),
                        "top_score": max(r.get("score", 0) for r in results)
                    }
                ))
            else:
                event_manager.emit(Event(
                    type=EventType.SEARCH_COMPLETED,
                    timestamp=datetime.now(),
                    component="search",
                    description="No matching endpoints found",
                    duration_ms=duration_ms,
                    metadata={"result_count": 0}
                ))
                console.print("\n[yellow]No matching endpoints found.[/yellow]")
                return
            
            # Display results
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
            
            # Emit final success event
            event_manager.emit(Event(
                type=EventType.SEARCH_COMPLETED,
                timestamp=datetime.now(),
                component="search",
                description="Search completed successfully",
                duration_ms=duration_ms,
                success=True,
                metadata={
                    "query": query,
                    "result_count": len(results),
                    "duration_ms": duration_ms,
                    "intent": query_intent
                }
            ))
            
        except Exception as e:
            # Emit error event
            event_manager.emit(Event(
                type=EventType.SEARCH_FAILED,
                timestamp=datetime.now(),
                component="search",
                description=f"Search failed: {str(e)}",
                success=False,
                error=str(e),
                metadata={"query": query}
            ))
            raise
        
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
        last_update = datetime.now()
        
        def make_header() -> Panel:
            """Create enhanced header with system information."""
            now = datetime.now()
            uptime = (now - event_manager._monitoring_start_time).total_seconds()
            
            # Create header grid
            grid = Table.grid(padding=(0, 1), expand=True)
            grid.add_column(ratio=1, justify="left")
            grid.add_column(ratio=1, justify="center")
            grid.add_column(ratio=1, justify="right")
            
            # Left section - System Info
            left_text = Text()
            left_text.append("🔍 ", style="bold green")
            left_text.append("Plexure API Search", style="bold green")
            
            # Center section - Monitoring Info
            center_text = Text()
            center_text.append("⚡ ", style="bold yellow")
            center_text.append(f"Refresh: {interval}s", style="bold yellow")
            center_text.append(" | ", style="dim")
            center_text.append("Uptime: ", style="dim")
            center_text.append(f"{int(uptime)}s", style="bold white")
            
            # Right section - Time Info
            right_text = Text()
            right_text.append("🕒 ", style="bold blue")
            right_text.append(now.strftime("%H:%M:%S"), style="bold blue")
            right_text.append(" | ", style="dim")
            right_text.append(now.strftime("%Y-%m-%d"), style="white")
            
            # Add sections to grid
            grid.add_row(left_text, center_text, right_text)
            
            # Create status bar
            status_grid = Table.grid(padding=(0, 1), expand=True)
            status_grid.add_column(ratio=1, justify="left")
            status_grid.add_column(ratio=1, justify="right")
            
            # Status info
            status_text = Text()
            status_text.append("Status: ", style="dim")
            status_text.append("[green]Active[/green]")
            status_text.append(" | ", style="dim")
            status_text.append("Press ", style="dim")
            status_text.append("[bold]Ctrl+C[/bold]", style="white")
            status_text.append(" to exit", style="dim")
            
            # System info
            system_text = Text()
            system_text.append("Memory: ", style="dim")
            system_text.append("[cyan]Active[/cyan]")
            system_text.append(" | ", style="dim")
            system_text.append("CPU: ", style="dim")
            system_text.append("[cyan]Normal[/cyan]")
            
            status_grid.add_row(status_text, system_text)
            
            # Combine grids in a group
            header_content = Group(
                grid,
                Text(),  # Spacer
                status_grid
            )
            
            return Panel(
                header_content,
                border_style="blue",
                title="[bold white]System Monitor[/bold white]",
                padding=(0, 1),
                expand=True
            )
        
        def make_metrics_panel() -> Panel:
            """Create metrics panel with current stats."""
            events = [e for e in event_manager.get_recent_events() 
                     if e.type != EventType.MONITORING_UPDATED]  # Filter out monitoring updates
            
            metrics_table = Table(
                show_header=True,
                header_style="bold magenta",
                box=box.ROUNDED,
                padding=(0, 1),
                expand=True
            )
            
            metrics_table.add_column("📊 Metric", style="cyan", no_wrap=True)
            metrics_table.add_column("Value", style="green", justify="right", no_wrap=True)
            
            # Add core metrics with icons
            metrics = {
                "📈 Total Events": len(events),
                "🔍 Search Events": len([e for e in events if e.type == EventType.SEARCH_STARTED]),
                "❌ Error Events": len([e for e in events if not e.success]),
                "💾 Cache Hit Rate": f"{event_manager.get_cache_hit_rate():.2%}",
                "⚡ Avg Search Time": f"{event_manager.get_average_search_time():.3f}s",
                "🤖 Active Models": event_manager.get_active_model_count(),
            }
            
            for metric, value in metrics.items():
                metrics_table.add_row(metric, str(value))
                
            return Panel(
                metrics_table,
                border_style="blue",
                title="[bold white]System Metrics[/bold white]",
                subtitle="[dim]Real-time system statistics[/dim]",
                padding=(0, 1),
                expand=True
            )
            
        def make_events_panel() -> Panel:
            """Create events panel with recent activity."""
            # Filter out monitoring events
            events = [e for e in event_manager.get_recent_events(minutes=2)
                     if e.type != EventType.MONITORING_UPDATED and 
                        e.type != EventType.MONITORING_STARTED and
                        e.type != EventType.MONITORING_STOPPED]
            
            if not events:
                return Panel(
                    "[dim]No recent events[/dim]",
                    title="[bold white]Recent Events[/bold white]",
                    subtitle="[dim]Waiting for activity...[/dim]",
                    border_style="yellow",
                    padding=(0, 1),
                    expand=True
                )
            
            events_table = Table(
                show_header=True,
                header_style="bold yellow",
                box=box.ROUNDED,
                padding=(0, 1),
                expand=True,
                show_lines=True  # Adiciona linhas entre as rows para melhor legibilidade
            )
            
            events_table.add_column("⏰ Time", style="cyan", width=8, no_wrap=True)
            events_table.add_column(
                "📋 Type",
                style="magenta",
                width=25,  # Aumentado para acomodar tipos mais longos
                overflow="fold"  # Quebra o texto em várias linhas se necessário
            )
            events_table.add_column(
                "🔧 Component",
                style="blue",
                width=12,
                no_wrap=True
            )
            events_table.add_column("✨", style="green", width=2, justify="center")
            events_table.add_column(
                "📝 Description",
                style="white",
                width=50,  # Largura fixa para descrição
                overflow="fold"  # Quebra o texto em várias linhas se necessário
            )
            
            for event in reversed(events[-10:]):  # Show last 10 events
                status = "✓" if event.success else "✗"
                status_style = "green" if event.success else "red"
                
                # Formata o tipo do evento para melhor legibilidade
                event_type = event.type.name.replace("_", " ").title()
                
                events_table.add_row(
                    event.timestamp.strftime("%H:%M:%S"),
                    event_type,
                    event.component,
                    f"[{status_style}]{status}[/{status_style}]",
                    str(event.description or "")
                )
            
            return Panel(
                events_table,
                border_style="yellow",
                title="[bold white]Recent Events[/bold white]",
                subtitle=f"[dim]Last {len(events)} events[/dim]",
                padding=(0, 1),
                expand=True
            )
            
        def make_error_panel() -> Panel:
            """Create error panel showing recent errors."""
            # Filter out monitoring events
            events = [e for e in event_manager.get_recent_events() 
                     if not e.success and e.type != EventType.MONITORING_UPDATED]
            
            if not events:
                return Panel(
                    "[dim]No errors[/dim]",
                    title="[bold white]Recent Errors[/bold white]",
                    subtitle="[dim]System running smoothly[/dim]",
                    border_style="red",
                    padding=(0, 1),
                    expand=True
                )
            
            error_table = Table(
                show_header=True,
                header_style="bold red",
                box=box.ROUNDED,
                padding=(0, 1),
                expand=True
            )
            
            error_table.add_column("⏰ Time", style="cyan", width=8, no_wrap=True)
            error_table.add_column("🔧 Component", style="blue", width=15, no_wrap=True)
            error_table.add_column("❌ Error", style="red")
            
            for event in reversed(events[-5:]):  # Show last 5 errors
                error_table.add_row(
                    event.timestamp.strftime("%H:%M:%S"),
                    event.component,
                    str(event.error or "Unknown error")
                )
            
            return Panel(
                error_table,
                border_style="red",
                title="[bold white]Recent Errors[/bold white]",
                subtitle=f"[dim]Last {len(events)} errors[/dim]",
                padding=(0, 1),
                expand=True
            )
        
        # Create layout
        layout = Layout()
        
        # Vertical layout
        layout.split_column(
            Layout(name="header", size=4),  # Ajustado para 4 linhas
            Layout(name="metrics", size=8),
            Layout(name="events", ratio=2),
            Layout(name="errors", size=8)
        )
        
        # Start live display
        with Live(layout, refresh_per_second=1/interval, screen=True) as live:
            try:
                while True:
                    # Update all panels
                    layout["header"].update(make_header())
                    layout["metrics"].update(make_metrics_panel())
                    layout["events"].update(make_events_panel())
                    layout["errors"].update(make_error_panel())
                    
                    # Emit monitoring update event (but don't show in events panel)
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

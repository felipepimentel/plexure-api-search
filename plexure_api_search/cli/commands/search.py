"""Search command implementation."""

import logging
import time
from datetime import datetime
from typing import Any, Dict

import click
from rich.console import Console
from rich.table import Table
from rich import box

from ...monitoring.events import Event, EventType, event_manager
from ...search.searcher import APISearcher
from ...search.understanding import ZeroShotUnderstanding
from ..ui.logging import setup_logging

console = Console()

@click.command()
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
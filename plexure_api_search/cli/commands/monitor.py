"""Monitor command implementation."""

import os
import time
from datetime import datetime

import click
from rich.console import Console
from rich.live import Live

from ...monitoring.events import Event, EventType, event_manager
from ..ui.logging import setup_logging
from ..ui.monitor import (
    MonitorUI,
    make_header,
    make_metrics_panel,
    make_events_panel,
    make_error_panel,
    create_monitor_layout
)
from ..ui.hot_reload import start_hot_reload

console = Console()

@click.command()
@click.option("-v", "--verbose", count=True, help="Increase output verbosity")
@click.option("--interval", default=2, help="Monitoring interval in seconds")
@click.option("--hot-reload/--no-hot-reload", default=False, help="Enable hot reload during development")
def monitor(verbose: int, interval: int, hot_reload: bool) -> None:
    """Monitor system metrics and performance with live updates."""
    try:
        # Setup logging
        setup_logging(verbose)
        
        # Initialize event manager monitoring
        event_manager.start_monitoring()
        
        # Create layout
        layout = create_monitor_layout()
        
        # Start hot reload if enabled
        observer = None
        if hot_reload:
            base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            observer = start_hot_reload(
                base_path=base_path,
                patterns=["*.py"],
                callback=lambda: console.print("[green]Hot reload: Modules reloaded[/green]")
            )
            console.print("[yellow]Hot reload enabled. Edit files to see changes.[/yellow]")
        
        # Start live display
        with Live(layout, refresh_per_second=1/interval, screen=True) as live:
            try:
                # Initialize monitor UI with event subscription
                monitor_ui = MonitorUI(layout, live, interval)
                
                # Initial update
                monitor_ui.update_all()
                
                while True:
                    # Sleep for interval
                    time.sleep(interval)
                    
                    # Emit monitoring update event
                    event_manager.emit(Event(
                        type=EventType.MONITORING_UPDATED,
                        timestamp=datetime.now(),
                        component="monitoring",
                        description="Updated monitoring display"
                    ))
                    
            except KeyboardInterrupt:
                event_manager.stop_monitoring()
                if observer:
                    observer.stop()
                    observer.join()
                console.print("\n[yellow]Monitoring stopped.[/yellow]")
                
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {str(e)}")
        if verbose > 0:
            console.print_exception()
        raise click.Abort() 
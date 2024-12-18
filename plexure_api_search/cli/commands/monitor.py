"""Monitor command implementation."""

import logging
import os
import signal
import sys
import time
from typing import Optional

import click
from rich.console import Console
from rich.live import Live

from plexure_api_search.cli.ui.hot_reload import start_hot_reload
from plexure_api_search.cli.ui.logging import setup_logging
from plexure_api_search.cli.ui.monitor import MonitorUI, create_monitor_layout
from plexure_api_search.monitoring.events import Event, subscriber

console = Console()
logger = logging.getLogger(__name__)


def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    logger.info("Received signal %s, shutting down...", signum)
    sys.exit(0)


def handle_event(event: Event) -> None:
    """Handle events in log-only mode."""
    process_name = event.metadata.get("process_name", "N/A")
    pid = event.metadata.get("pid", "N/A")
    description = event.description or "N/A"
    duration = f" ({event.duration_ms:.2f}ms)" if event.duration_ms else ""
    error = f" [ERROR: {event.error}]" if event.error else ""

    logger.info(
        "[%s] %s (PID: %s) - %s%s%s",
        event.type.name,
        process_name,
        pid,
        description,
        duration,
        error,
    )


@click.command()
@click.option("-v", "--verbose", count=True, help="Increase output verbosity")
@click.option(
    "--refresh-interval",
    "-r",
    default=2.0,
    help="Refresh interval in seconds (min: 0.1, max: 60.0)",
    type=float,
)
@click.option(
    "--hot-reload",
    is_flag=True,
    help="Enable hot reload for development",
)
@click.option(
    "--process-filter",
    "-p",
    help="Filter events by process name (e.g., 'python' or 'plexure-api')",
    type=str,
)
@click.option(
    "--no-tui",
    is_flag=True,
    help="Disable TUI and show only logs",
)
def monitor(
    verbose: int,
    refresh_interval: float,
    hot_reload: bool,
    process_filter: Optional[str] = None,
    no_tui: bool = False,
) -> None:
    """Monitor external process metrics and performance with live updates."""
    try:
        # Validate refresh interval
        if not 0.1 <= refresh_interval <= 60.0:
            console.print(
                "[red]Error:[/red] Refresh interval must be between 0.1 and 60.0 seconds"
            )
            sys.exit(1)

        # Setup logging with monitor flag
        setup_logging(verbose, is_monitor=True)
        logger.debug(
            f"Starting external process monitor with refresh interval: {refresh_interval}s"
        )

        # Setup signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Start subscriber first (must bind before publishers connect)
        subscriber.start()
        logger.info("Event subscriber started and waiting for publishers...")

        if no_tui:
            # Log-only mode
            try:
                subscriber.subscribe(handle_event)
                logger.info("Log-only mode enabled. Press Ctrl+C to stop.")

                # Keep main thread alive
                while True:
                    time.sleep(0.1)

            except KeyboardInterrupt:
                logger.info("Monitor stopped by user.")
            finally:
                subscriber.unsubscribe(handle_event)
                subscriber.stop()
                return

        # Create layout first
        layout = create_monitor_layout()
        logger.debug("Monitor layout created")

        # Calculate optimal refresh rate for live display
        live_refresh_rate = max(10, int(1 / (refresh_interval / 4)))  # At least 10 Hz

        # Create live display with proper refresh rate
        live = Live(
            layout,
            console=console,
            screen=True,
            refresh_per_second=live_refresh_rate,
            transient=True,  # Prevent screen artifacts
        )

        # Start hot reload if enabled
        observer = None
        if hot_reload:
            try:
                base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                observer = start_hot_reload(
                    base_path=base_path,
                    patterns=["*.py"],
                    callback=lambda: logger.info("Files changed, restart required"),
                )
                console.print(
                    "[yellow]Hot reload enabled. Edit files to see changes.[/yellow]"
                )
                logger.debug("Hot reload observer started")
            except Exception as e:
                logger.error("Failed to start hot reload: %s", str(e))
                console.print(
                    "[red]Failed to start hot reload. Continuing without it.[/red]"
                )

        # Create and start monitor UI
        monitor_ui = None
        try:
            with live:
                # Initialize monitor UI
                monitor_ui = MonitorUI(layout, live, refresh_interval)
                logger.debug("MonitorUI initialized")

                # Start monitoring loop
                monitor_ui.start()  # Blocks until Ctrl+C or error

        except KeyboardInterrupt:
            logger.info("Monitor stopped by user.")
        except Exception as e:
            logger.error("Error in live display: %s", str(e))
            console.print(f"[red]Error in live display:[/red] {str(e)}")
            if verbose > 1:
                console.print_exception()
        finally:
            # Cleanup resources
            if monitor_ui:
                monitor_ui.cleanup()
            if observer:
                observer.stop()
                observer.join()
            subscriber.stop()
            console.print("\n[yellow]Monitor stopped.[/yellow]")

    except Exception as e:
        logger.error("Error in monitor command: %s", str(e), exc_info=True)
        sys.exit(1)

"""Monitor UI implementation for displaying events from other processes."""

import logging
import sys
import time
import select
import threading
import queue
from datetime import datetime
from threading import Lock, Event as ThreadEvent
from typing import List, Optional, Set

from rich import box
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from plexure_api_search.monitoring.events import Event, subscriber

logger = logging.getLogger(__name__)


class MonitorUI:
    """Monitor UI that displays events from other processes."""

    def __init__(
        self, layout: Layout, live: Live, interval: float = 2.0
    ):
        """Initialize monitor UI.

        Args:
            layout: Rich layout instance
            live: Live display instance
            interval: Update interval for non-event updates in seconds
        """
        self.layout = layout
        self.live = live
        self.interval = float(interval)
        self._start_time = datetime.now()
        self._recent_events: List[Event] = []
        self._lock = Lock()
        self._last_update = time.time()
        self._force_refresh = False
        self._active_publishers: Set[int] = set()
        
        # Event processing
        self._stop_event = ThreadEvent()
        self._event_queue = queue.Queue(maxsize=1000)
        self._event_thread = None
        self._update_thread = None

        # Navigation
        self._current_page = 0
        self._events_per_page = 20
        self._total_pages = 0
        self._last_known_pids = set()

        # Initial layout update
        self._update_layout()

        # Subscribe to events
        subscriber.subscribe(self._handle_event)
        logger.info("Monitor started and listening for events")

    def _handle_event(self, event: Event) -> None:
        """Queue event for processing.

        Args:
            event: The event to process
        """
        try:
            # Queue event for processing
            try:
                self._event_queue.put_nowait(event)
            except queue.Full:
                logger.warning("Event queue full, dropping event")
                return

            # Log the event using normal logger
            logger.debug(
                f"Event received: {event.type.name} from "
                f"PID {event.metadata.get('pid')} ({event.metadata.get('process_name', 'N/A')})"
            )

        except Exception as e:
            logger.error(f"Error queueing event: {e}", exc_info=True)

    def _process_events(self) -> None:
        """Process events from the queue."""
        while not self._stop_event.is_set():
            try:
                # Process up to 10 events at once
                for _ in range(10):
                    try:
                        event = self._event_queue.get_nowait()
                    except queue.Empty:
                        break

                    with self._lock:
                        # Track publisher
                        if "pid" in event.metadata:
                            pid = event.metadata["pid"]
                            is_new_pid = pid not in self._active_publishers
                            self._active_publishers.add(pid)
                            
                            # If this is a new PID, move to first page
                            if is_new_pid:
                                self._current_page = 0

                        # Store event
                        self._recent_events.append(event)
                        if len(self._recent_events) > 1000:  # Increased event history
                            self._recent_events.pop(0)

                        # Force UI refresh
                        self._force_refresh = True

                    self._event_queue.task_done()

                # Short sleep to prevent CPU spinning
                time.sleep(0.01)

            except Exception as e:
                logger.error(f"Error in event processing loop: {e}", exc_info=True)

    def _update_ui(self) -> None:
        """Update UI in a separate thread."""
        while not self._stop_event.is_set():
            try:
                current_time = time.time()
                time_since_last_update = current_time - self._last_update

                if self._force_refresh or time_since_last_update >= self.interval:
                    with self._lock:
                        self._update_layout()
                        self.live.refresh()
                        self._last_update = current_time
                        self._force_refresh = False

                # Sleep for a shorter interval to be more responsive
                time.sleep(min(0.1, self.interval / 4))

            except Exception as e:
                logger.error(f"Error in UI update loop: {e}", exc_info=True)

    def next_page(self) -> None:
        """Move to next page of events."""
        if self._current_page < self._total_pages - 1:
            self._current_page += 1
            self._force_refresh = True

    def previous_page(self) -> None:
        """Move to previous page of events."""
        if self._current_page > 0:
            self._current_page -= 1
            self._force_refresh = True

    def first_page(self) -> None:
        """Move to first page of events."""
        if self._current_page != 0:
            self._current_page = 0
            self._force_refresh = True

    def last_page(self) -> None:
        """Move to last page of events."""
        if self._current_page != self._total_pages - 1:
            self._current_page = max(0, self._total_pages - 1)
            self._force_refresh = True

    def start(self) -> None:
        """Start the monitor UI loop."""
        try:
            logger.info("Starting monitor UI...")
            
            # Start event processing thread
            self._event_thread = threading.Thread(
                target=self._process_events,
                name="EventProcessor",
                daemon=True
            )
            self._event_thread.start()

            # Start UI update thread
            self._update_thread = threading.Thread(
                target=self._update_ui,
                name="UIUpdater",
                daemon=True
            )
            self._update_thread.start()

            # Keep main thread alive for Ctrl+C and handle navigation
            try:
                while not self._stop_event.is_set():
                    # Check for keyboard input
                    if sys.stdin in select.select([sys.stdin], [], [], 0.1)[0]:
                        key = sys.stdin.read(1)
                        if key == '\x1b':  # Escape sequence
                            key = sys.stdin.read(2)
                            if key == '[A':  # Up arrow
                                self.previous_page()
                            elif key == '[B':  # Down arrow
                                self.next_page()
                            elif key == '[H':  # Home
                                self.first_page()
                            elif key == '[F':  # End
                                self.last_page()

            except KeyboardInterrupt:
                logger.info("Monitor stopped by user")
                self._stop_event.set()

        except Exception as e:
            logger.error(f"Error in monitor UI: {e}", exc_info=True)
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Clean up resources."""
        # Signal threads to stop
        self._stop_event.set()

        # Wait for threads to finish
        if self._event_thread and self._event_thread.is_alive():
            self._event_thread.join(timeout=1.0)
        if self._update_thread and self._update_thread.is_alive():
            self._update_thread.join(timeout=1.0)

        # Unsubscribe from events
        subscriber.unsubscribe(self._handle_event)
        logger.info("Monitor stopped")

    def _make_header(self) -> Panel:
        """Create the header panel."""
        uptime = int((datetime.now() - self._start_time).total_seconds())
        publishers = len(self._active_publishers)
        queue_size = self._event_queue.qsize()
        current_page = self._current_page + 1
        total_pages = max(1, self._total_pages)
        
        header_text = Text.assemble(
            ("External Process Monitor - ", "bold green"),
            (f"Uptime: {uptime}s", "bold green"),
            (" | ", "dim"),
            (f"Active Publishers: {publishers}", "bold blue"),
            (" | ", "dim"),
            (f"Event Queue: {queue_size}", "cyan"),
            (" | ", "dim"),
            (f"Page {current_page}/{total_pages}", "yellow"),
            ("\n\n", ""),
            ("Navigation: ", "bold"),
            ("↑/↓", "cyan"),
            (" Move pages • ", "dim"),
            ("Home/End", "cyan"),
            (" First/Last page", "dim"),
        )
        return Panel(header_text, title="Monitor Status", border_style="blue")

    def _make_events_panel(self) -> Panel:
        """Create the events panel."""
        table = Table(box=box.ROUNDED, show_lines=True)
        table.add_column("Time", style="cyan", no_wrap=True, width=8)
        table.add_column("Process", style="magenta", width=15)
        table.add_column("PID", style="blue", width=8)
        table.add_column("Type", style="yellow", width=20)
        table.add_column("Description", style="white", min_width=50, max_width=100)

        # Group events by PID with last seen time
        events_by_pid = {}
        for event in self._recent_events:
            pid = event.metadata.get("pid", "N/A")
            if pid not in events_by_pid:
                events_by_pid[pid] = {
                    "events": [],
                    "last_seen": event.timestamp,
                    "process_name": event.metadata.get("process_name", "N/A")
                }
            events_by_pid[pid]["events"].append(event)
            if event.timestamp > events_by_pid[pid]["last_seen"]:
                events_by_pid[pid]["last_seen"] = event.timestamp

        # Sort PIDs by last seen timestamp (most recent first)
        sorted_pids = sorted(
            events_by_pid.keys(),
            key=lambda pid: events_by_pid[pid]["last_seen"],
            reverse=True
        )

        # Update total pages
        self._total_pages = len(sorted_pids)

        # Check for new PIDs
        current_pids = set(sorted_pids)
        if current_pids - self._last_known_pids:
            self._current_page = 0
        self._last_known_pids = current_pids

        # Get current PID based on page
        if sorted_pids:
            page_idx = min(self._current_page, len(sorted_pids) - 1)
            current_pid = sorted_pids[page_idx]
            pid_info = events_by_pid[current_pid]
            events = pid_info["events"]

            # Show most recent events first
            for event in reversed(events[-self._events_per_page:]):
                description = event.description or "N/A"
                if len(description) > 100:
                    description = description[:97] + "..."

                table.add_row(
                    event.timestamp.strftime("%H:%M:%S"),
                    pid_info["process_name"],
                    str(current_pid),
                    event.type.name,
                    description,
                )

            title = (
                f"Events for PID {current_pid} ({len(events)} events) - "
                f"Page {self._current_page + 1}/{self._total_pages} - "
                f"{len(sorted_pids)} Active PIDs"
            )
        else:
            table.add_row(
                datetime.now().strftime("%H:%M:%S"),
                "N/A",
                "N/A",
                "N/A",
                "Waiting for events...",
            )
            title = "No Active PIDs"

        return Panel(
            table,
            title=title,
            border_style="yellow",
            padding=(0, 1),
        )

    def _update_layout(self) -> None:
        """Update all UI panels."""
        try:
            self.layout["header"].update(self._make_header())
            self.layout["events"].update(self._make_events_panel())
        except Exception as e:
            logger.error(f"Error updating layout: {e}", exc_info=True)


def create_monitor_layout() -> Layout:
    """Create the monitor layout.

    Returns:
        Layout configured for the monitor
    """
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=5),  # Increased for navigation help
        Layout(name="events", ratio=1),
    )
    return layout 
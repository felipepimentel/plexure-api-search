"""Monitor UI for displaying events."""

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import DataTable, Footer, Header, Static

from ...services.events import Event, SubscriberService

logger = logging.getLogger(__name__)

class MonitorUI(App):
    """Monitor UI for displaying events."""

    CSS = """
    Screen {
        layout: grid;
        grid-size: 1;
        grid-rows: 1fr;
    }

    Container {
        height: 100%;
        width: 100%;
    }

    DataTable {
        height: 100%;
        width: 100%;
        border: solid green;
    }
    """

    def __init__(self):
        """Initialize monitor UI."""
        super().__init__()
        self.subscriber = SubscriberService()
        self.events: List[Event] = []
        self.table: Optional[DataTable] = None

    def compose(self) -> ComposeResult:
        """Create child widgets.
        
        Returns:
            Iterator of widgets
        """
        yield Header()
        yield Container(
            DataTable(
                show_header=True,
                zebra_stripes=True,
            )
        )
        yield Footer()

    def on_mount(self) -> None:
        """Handle mount event."""
        # Initialize table
        self.table = self.query_one(DataTable)
        self.table.add_columns(
            "Time",
            "Component",
            "Type",
            "Description",
            "Status",
        )

        # Start subscriber
        self.subscriber.start()

        # Start event polling
        self.set_interval(0.1, self.poll_events)

    def on_unmount(self) -> None:
        """Handle unmount event."""
        self.subscriber.stop()

    def poll_events(self) -> None:
        """Poll for new events."""
        event = self.subscriber.poll(timeout=100)
        if event:
            self.add_event(event)

    def add_event(self, event: Event) -> None:
        """Add event to table.
        
        Args:
            event: Event to add
        """
        # Add to list
        self.events.append(event)

        # Add to table
        self.table.add_row(
            event.timestamp.strftime("%H:%M:%S"),
            event.component,
            event.type,
            event.description,
            "✓" if event.success else "✗",
        )

        # Scroll to bottom
        self.table.scroll_end(animate=False) 
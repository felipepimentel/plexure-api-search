"""Monitor UI components."""

from datetime import datetime
from typing import List

from rich import box
from rich.console import Group
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ...monitoring.events import Event, EventType, event_manager


def make_header(interval: int) -> Panel:
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
        show_lines=True
    )
    
    events_table.add_column("⏰ Time", style="cyan", width=8, no_wrap=True)
    events_table.add_column(
        "📋 Type",
        style="magenta",
        width=25,
        overflow="fold"
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
        width=50,
        overflow="fold"
    )
    
    for event in reversed(events[-10:]):  # Show last 10 events
        status = "✓" if event.success else "✗"
        status_style = "green" if event.success else "red"
        
        # Format event type for better readability
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


def create_monitor_layout() -> Layout:
    """Create the monitor layout."""
    layout = Layout()
    
    # Vertical layout
    layout.split_column(
        Layout(name="header", size=4),
        Layout(name="metrics", size=8),
        Layout(name="events", ratio=2),
        Layout(name="errors", size=8)
    )
    
    return layout 
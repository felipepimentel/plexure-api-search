"""Monitor command for viewing events."""

import click

from ...cli.ui.monitor import MonitorUI

@click.command()
@click.option(
    "--no-tui",
    is_flag=True,
    help="Disable TUI and only log events",
)
def monitor(no_tui: bool):
    """Monitor events from other processes."""
    ui = MonitorUI()
    ui.run()

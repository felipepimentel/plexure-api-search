"""Command line interface package."""

import click
from dotenv import load_dotenv

from .commands.search import search
from .commands.index import index
from .commands.monitor import monitor

# Load environment variables
load_dotenv()

@click.group()
def cli() -> None:
    """Plexure API Search CLI"""
    pass

# Register commands
cli.add_command(search)
cli.add_command(index)
cli.add_command(monitor)

if __name__ == "__main__":
    cli() 
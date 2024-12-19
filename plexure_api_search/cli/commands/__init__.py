"""CLI commands package."""

import click

from .index import index
from .monitor import monitor
from .search import search

@click.group()
def cli():
    """Plexure API Search CLI."""
    pass

cli.add_command(index)
cli.add_command(monitor)
cli.add_command(search) 
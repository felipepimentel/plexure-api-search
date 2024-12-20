"""
Main Entry Point for Plexure API Search

This module serves as the main entry point for the Plexure API Search package when run as a command-line
application. It initializes the CLI interface, sets up logging, and handles the execution of commands.

The module provides the following functionality:
- Initializes the application configuration and environment
- Sets up logging with appropriate levels and handlers
- Registers and handles CLI commands (search, index, config, etc.)
- Provides error handling and graceful shutdown

Example Usage:
    $ python -m plexure_api_search search "find authentication endpoints"
    $ python -m plexure_api_search index --clear
    $ python -m plexure_api_search config show

The module uses Click for CLI argument parsing and command handling, providing a user-friendly
interface for interacting with the search engine.
"""

import sys
from typing import Optional, Sequence

import click

from .cli.commands import cli
from .initialize import initialize


def main(args: Optional[Sequence[str]] = None) -> int:
    """Main entry point.

    Args:
        args: Command line arguments.
            Defaults to sys.argv[1:].

    Returns:
        Exit code.
    """
    try:
        # Initialize package
        initialize()

        # Run CLI
        cli(args=args)
        return 0

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

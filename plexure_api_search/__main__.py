"""Main entry point for Plexure API Search."""

import sys
from typing import Optional, Sequence

import click

from . import initialize, Environment
from .cli.commands import cli

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

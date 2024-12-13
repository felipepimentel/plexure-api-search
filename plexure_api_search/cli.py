"""Command line interface for Plexure API Search."""

import argparse
import sys
from typing import Optional

from rich.console import Console

from plexure_api_search.config import Config, ConfigManager
from plexure_api_search.indexer import APIIndexer
from plexure_api_search.monitoring import Logger
from plexure_api_search.searcher import SearchEngine


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI.
    
    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Plexure API Search - Search and explore API endpoints"
    )
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Index command
    index_parser = subparsers.add_parser("index", help="Index API contracts")
    index_parser.add_argument(
        "--api-dir",
        help="Directory containing API contracts"
    )
    index_parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="Force reindexing even if index exists"
    )
    index_parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Skip indexing if index exists"
    )
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search API endpoints")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument(
        "--api-dir",
        help="Directory containing API contracts"
    )
    
    # Config command
    config_parser = subparsers.add_parser("config", help="Manage configuration")
    config_parser.add_argument(
        "--show",
        action="store_true",
        help="Show current configuration"
    )
    config_parser.add_argument(
        "--set",
        nargs=2,
        metavar=("KEY", "VALUE"),
        help="Set configuration value"
    )
    
    return parser


def handle_index(args: argparse.Namespace, config: Config, logger: Logger) -> int:
    """Handle index command.
    
    Args:
        args: Parsed command line arguments.
        config: Configuration object.
        logger: Logger instance.
        
    Returns:
        Exit code.
    """
    try:
        api_dir = args.api_dir or config.api_dir
        if not api_dir:
            logger.error("No API directory specified")
            return 1
            
        indexer = APIIndexer(api_dir=api_dir)
        if not args.skip_index:
            indexer.index_apis(force_reindex=args.force_reindex)
            
        return 0
        
    except Exception as e:
        logger.error(f"Indexing failed: {str(e)}")
        return 1


def handle_search(args: argparse.Namespace, config: Config, logger: Logger) -> int:
    """Handle search command.
    
    Args:
        args: Parsed command line arguments.
        config: Configuration object.
        logger: Logger instance.
        
    Returns:
        Exit code.
    """
    try:
        api_dir = args.api_dir or config.api_dir
        if not api_dir:
            logger.error("No API directory specified")
            return 1
            
        # First ensure index exists
        indexer = APIIndexer(api_dir=api_dir)
        indexer.index_apis(force_reindex=False)
        
        # Then perform search
        engine = SearchEngine()
        results = engine.search(args.query)
        engine.display_results(results)
        
        return 0
        
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        return 1


def handle_config(args: argparse.Namespace, config_manager: ConfigManager, logger: Logger) -> int:
    """Handle config command.
    
    Args:
        args: Parsed command line arguments.
        config_manager: Configuration manager instance.
        logger: Logger instance.
        
    Returns:
        Exit code.
    """
    try:
        if args.show:
            config_manager.show_config()
            return 0
            
        if args.set:
            key, value = args.set
            config_manager.set_config(key, value)
            logger.info(f"Configuration updated: {key} = {value}")
            return 0
            
        logger.error("No config action specified")
        return 1
        
    except Exception as e:
        logger.error(f"Configuration failed: {str(e)}")
        return 1


def main(argv: Optional[list[str]] = None) -> int:
    """Main entry point for CLI.
    
    Args:
        argv: Optional list of command line arguments.
        
    Returns:
        Exit code.
    """
    parser = create_parser()
    args = parser.parse_args(argv)
    
    # Initialize components
    config_manager = ConfigManager()
    config = config_manager.load_config()
    logger = Logger()
    console = Console()
    
    try:
        if args.command == "index":
            return handle_index(args, config, logger)
            
        elif args.command == "search":
            return handle_search(args, config, logger)
            
        elif args.command == "config":
            return handle_config(args, config_manager, logger)
            
        else:
            parser.print_help()
            return 0
            
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/]")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 
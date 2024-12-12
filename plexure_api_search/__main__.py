"""Main module for the Plexure API Search tool."""

import argparse

from rich.console import Console

from plexure_api_search.constants import DEFAULT_API_DIR
from plexure_api_search.indexer import APIIndexer
from plexure_api_search.searcher import APISearcher


def main():
    """Main function for the API search tool."""
    console = Console()

    parser = argparse.ArgumentParser(
        description="Plexure API Search - Search and explore API endpoints"
    )

    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Index command
    index_parser = subparsers.add_parser("index", help="Index API contracts")
    index_parser.add_argument(
        "--api-dir", default=DEFAULT_API_DIR, help="Directory containing API contracts"
    )
    index_parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="Force reindexing even if index exists",
    )
    index_parser.add_argument(
        "--skip-index", action="store_true", help="Skip indexing if index exists"
    )

    # Search command
    search_parser = subparsers.add_parser("search", help="Search API endpoints")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument(
        "--api-dir", default=DEFAULT_API_DIR, help="Directory containing API contracts"
    )

    args = parser.parse_args()

    try:
        if args.command == "index":
            # Run indexing
            if not args.skip_index:
                indexer = APIIndexer(api_dir=args.api_dir)
                indexer.index_apis(force_reindex=args.force_reindex)

        elif args.command == "search":
            # First ensure index exists
            indexer = APIIndexer(api_dir=args.api_dir)
            indexer.index_apis(force_reindex=False)

            # Then perform search
            searcher = APISearcher()
            results = searcher.search(args.query)
            searcher.display_results(results)

        else:
            parser.print_help()

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/]")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

"""Command line interface for API search."""

import json
import logging
from typing import Any, Dict, List, Optional

import click
from dotenv import load_dotenv
from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

from .indexing import APIIndexer
from .search import Consistency
from .search.expansion import QueryExpander
from .search.searcher import APISearcher
from .search.understanding import ZeroShotUnderstanding

# Load environment variables
load_dotenv()

# Initialize console
console = Console()


# Configure logging based on verbosity
def setup_logging(verbosity: int):
    """Configure logging based on verbosity level."""
    if verbosity == 0:
        logging.basicConfig(level=logging.ERROR)
    elif verbosity == 1:
        logging.basicConfig(level=logging.WARNING)
    elif verbosity == 2:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.DEBUG)

    # Suppress specific loggers unless in debug mode
    if verbosity < 3:
        logging.getLogger("pinecone").setLevel(logging.WARNING)
        logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
        logging.getLogger("pinecone_plugin_interface").setLevel(logging.WARNING)


def render_config_info(config: Dict[str, Any]) -> Panel:
    """Render configuration information."""
    config_text = Text()
    config_text.append("\n🔧 Pinecone Configuration\n", style="bold yellow")

    # Show non-sensitive config info
    safe_keys = ["index_name", "environment", "cloud", "region"]
    for key in safe_keys:
        if key in config:
            config_text.append(f"{key.title()}: ", style="bright_blue")
            config_text.append(f"{config[key]}\n", style="white")

    return Panel(
        config_text,
        title="[bold]Configuration",
        border_style="blue",
        box=box.ROUNDED,
        padding=(1, 2),
    )


def create_search_layout() -> Layout:
    """Create the main search layout."""
    layout = Layout(name="root")

    # Create a more compact layout
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1),
        Layout(name="footer", size=3),
    )

    # Split main area horizontally
    layout["main"].split_row(
        Layout(name="results", ratio=2, minimum_size=50),
        Layout(name="side_panel", ratio=1, minimum_size=30),
    )

    return layout


def render_header(query: str) -> Panel:
    """Render the header panel."""
    title = Text()
    title.append("🔍 ", style="bold yellow")
    title.append("Plexure API Search", style="bold cyan")
    title.append(" • ", style="dim")
    title.append(f'"{query}"', style="italic green")

    return Panel(
        Align.center(title),
        box=box.ROUNDED,
        border_style="blue",
        padding=(0, 2),
    )


def render_footer() -> Panel:
    """Render the footer panel."""
    footer_text = Text()
    footer_text.append("↑/↓", style="bold yellow")
    footer_text.append(" Navigate • ", style="dim")
    footer_text.append("f", style="bold magenta")
    footer_text.append(" Filter • ", style="dim")
    footer_text.append("s", style="bold cyan")
    footer_text.append(" Sort • ", style="dim")
    footer_text.append("h", style="bold green")
    footer_text.append(" Help • ", style="dim")
    footer_text.append("q", style="bold red")
    footer_text.append(" Quit", style="dim")

    return Panel(
        Align.center(footer_text),
        box=box.ROUNDED,
        border_style="blue",
        padding=(0, 2),
    )


def render_search_progress() -> Panel:
    """Render a simple search progress indicator."""
    return Panel(
        "🔍 [bold cyan]Searching APIs...[/bold cyan]",
        box=box.ROUNDED,
        border_style="blue",
        padding=(1, 2),
    )


def render_result_details(result: Dict[str, Any]) -> Panel:
    """Render detailed view of a search result."""
    details = Text()

    # Score
    score_percentage = float(result["score"]) * 100
    score_color = (
        "green"
        if score_percentage > 80
        else "yellow"
        if score_percentage > 50
        else "red"
    )
    details.append("\n💯 Relevance Score: ", style="bold")
    details.append(f"{score_percentage:.1f}%\n", style=f"bold {score_color}")

    # API Info
    details.append("\n📚 API Information\n", style="bold cyan")
    details.append(f"Name: {result['api_name']}\n", style="bright_blue")
    details.append(f"Version: {result['api_version']}\n", style="blue")

    # Endpoint Info
    details.append("\n🔗 Endpoint Details\n", style="bold magenta")
    method_colors = {
        "GET": "green",
        "POST": "yellow",
        "PUT": "blue",
        "DELETE": "red",
        "PATCH": "magenta",
    }
    details.append("Method: ", style="bright_magenta")
    details.append(
        f"{result['method']}\n",
        style=f"bold {method_colors.get(result['method'], 'white')}",
    )
    details.append("Path: ", style="bright_magenta")
    details.append(f"{result['path']}\n", style="white")

    # Authentication
    auth_style = "bold red" if result["requires_auth"] else "bold green"
    details.append("\n🔒 Authentication: ", style="bold")
    details.append(
        "Required\n" if result["requires_auth"] else "Not Required\n",
        style=auth_style,
    )

    # Description
    if result.get("description"):
        details.append("\n📝 Description\n", style="bold yellow")
        details.append(f"{result['description']}\n", style="bright_yellow")

    # Parameters
    if result.get("parameters"):
        details.append("\n⚙️ Parameters\n", style="bold green")
        for param in result["parameters"]:
            details.append(f"• {param}\n", style="bright_green")

    # Responses
    if result.get("responses"):
        details.append("\n📤 Responses\n", style="bold blue")
        for response in result["responses"]:
            details.append(f"• {response}\n", style="bright_blue")

    # Tags
    if result.get("tags"):
        details.append("\n🏷️ Tags\n", style="bold magenta")
        details.append(", ".join(result["tags"]) + "\n", style="bright_magenta")

    # Deprecation warning
    if result.get("deprecated"):
        details.append("\n⚠️ DEPRECATED\n", style="bold red")

    return Panel(
        details,
        title="[bold]API Endpoint Details",
        border_style="cyan",
        padding=(1, 2),
    )


@click.group()
def cli():
    """API Search CLI"""
    pass


class SearchState:
    """Manage search state and navigation."""

    def __init__(self, results: Dict[str, Any]):
        self.results = results.get("results", [])
        # Convert related queries to list of dictionaries if not already
        related = results.get("related_queries", [])
        if isinstance(related, list):
            self.related_queries = [
                {
                    "query": q.get("query", ""),
                    "category": q.get("category", ""),
                    "description": q.get("description", ""),
                    "score": float(q.get("score", 0)),
                }
                if isinstance(q, dict)
                else {"query": str(q), "category": "", "description": "", "score": 0.0}
                for q in related
            ]
        else:
            self.related_queries = []
        self.selected_index = 0
        self.is_details_view = False
        self.filter_auth_only = False
        self.sort_by = "relevance"  # or "method", "api", "path"
        self.filter_text = ""

    @property
    def filtered_results(self) -> List[Dict[str, Any]]:
        """Get filtered results based on current state."""
        results = self.results

        # Apply auth filter
        if self.filter_auth_only:
            results = [r for r in results if r.get("requires_auth", False)]

        # Apply text filter
        if self.filter_text:
            results = [
                r
                for r in results
                if self.filter_text.lower() in r.get("path", "").lower()
                or self.filter_text.lower() in r.get("api_name", "").lower()
            ]

        # Apply sorting
        if self.sort_by == "method":
            results = sorted(results, key=lambda x: x.get("method", ""))
        elif self.sort_by == "api":
            results = sorted(results, key=lambda x: x.get("api_name", ""))
        elif self.sort_by == "path":
            results = sorted(results, key=lambda x: x.get("path", ""))
        # Default is relevance (already sorted)

        return results

    @property
    def current_result(self) -> Optional[Dict[str, Any]]:
        """Get currently selected result."""
        results = self.filtered_results
        if not results:
            return None
        return results[min(self.selected_index, len(results) - 1)]

    def next_result(self):
        """Move to next result."""
        if self.filtered_results:
            self.selected_index = (self.selected_index + 1) % len(self.filtered_results)

    def prev_result(self):
        """Move to previous result."""
        if self.filtered_results:
            self.selected_index = (self.selected_index - 1) % len(self.filtered_results)


def render_help() -> Panel:
    """Render help panel with keyboard shortcuts."""
    help_text = Text()
    help_text.append("\n🎯 Navigation\n", style="bold yellow")
    help_text.append("↑/↓", style="bold cyan")
    help_text.append(" Move selection\n", style="dim")
    help_text.append("Enter", style="bold green")
    help_text.append(" Toggle details view\n", style="dim")
    help_text.append("Esc", style="bold red")
    help_text.append(" Back/Close\n", style="dim")

    help_text.append("\n🔍 Filtering & Sorting\n", style="bold yellow")
    help_text.append("f", style="bold magenta")
    help_text.append(" Filter results\n", style="dim")
    help_text.append("a", style="bold blue")
    help_text.append(" Toggle auth-only\n", style="dim")
    help_text.append("s", style="bold cyan")
    help_text.append(" Change sort order\n", style="dim")

    help_text.append("\n⚡ Quick Actions\n", style="bold yellow")
    help_text.append("c", style="bold green")
    help_text.append(" Copy endpoint path\n", style="dim")
    help_text.append("d", style="bold blue")
    help_text.append(" Download OpenAPI spec\n", style="dim")
    help_text.append("h", style="bold cyan")
    help_text.append(" Toggle this help\n", style="dim")
    help_text.append("q", style="bold red")
    help_text.append(" Quit\n", style="dim")

    return Panel(
        help_text,
        title="[bold]Keyboard Shortcuts",
        border_style="blue",
        box=box.ROUNDED,
        padding=(1, 2),
    )


def render_stats(state: SearchState) -> Panel:
    """Render search statistics."""
    stats = Text()
    total = len(state.results)
    filtered = len(state.filtered_results)
    auth_count = sum(1 for r in state.filtered_results if r.get("requires_auth", False))

    stats.append("\n📊 Results: ", style="bold yellow")
    stats.append(f"{filtered}/{total}\n", style="cyan")

    stats.append("🔒 Auth Required: ", style="bold yellow")
    stats.append(f"{auth_count}/{filtered}\n", style="cyan")

    stats.append("🔄 Sort: ", style="bold yellow")
    stats.append(f"{state.sort_by.title()}\n", style="cyan")

    if state.filter_text:
        stats.append("🔍 Filter: ", style="bold yellow")
        stats.append(f"{state.filter_text}\n", style="cyan")

    return Panel(
        stats,
        title="[bold]Search Stats",
        border_style="blue",
        box=box.ROUNDED,
        padding=(1, 2),
    )


def render_method_legend() -> Panel:
    """Render HTTP method color legend."""
    legend = Text()
    method_colors = {
        "GET": "green",
        "POST": "yellow",
        "PUT": "blue",
        "DELETE": "red",
        "PATCH": "magenta",
    }

    for method, color in method_colors.items():
        legend.append(f"{method}", style=f"bold {color}")
        legend.append(" ")

    return Panel(
        Align.center(legend),
        title="[bold]HTTP Methods",
        border_style="blue",
        box=box.ROUNDED,
        padding=(1, 0),
    )


def render_results_table(state: SearchState) -> Table:
    """Render table of search results."""
    table = Table(
        title="[bold]API Search Results",
        expand=True,
        border_style="cyan",
        box=box.ROUNDED,
    )

    # Add columns
    table.add_column("Score", justify="center", style="cyan", width=8)
    table.add_column("Method", justify="center", style="green", width=8)
    table.add_column("Path", style="blue")
    table.add_column("API", style="magenta", width=20)
    table.add_column("Auth", justify="center", style="yellow", width=6)

    # Add rows
    for i, result in enumerate(state.filtered_results):
        # Score formatting
        score_percentage = float(result["score"]) * 100
        score_color = (
            "green"
            if score_percentage > 80
            else "yellow"
            if score_percentage > 50
            else "red"
        )
        score_text = f"{score_percentage:.0f}%"

        # Method formatting
        method_colors = {
            "GET": "green",
            "POST": "yellow",
            "PUT": "blue",
            "DELETE": "red",
            "PATCH": "magenta",
        }
        method_text = Text(
            result["method"],
            style=method_colors.get(result["method"], "white"),
        )

        # Auth formatting
        auth_text = "🔒" if result["requires_auth"] else "🔓"

        # Row style for selected item
        row_style = "reverse" if i == state.selected_index else ""

        table.add_row(
            score_text,
            method_text,
            result["path"],
            result["api_name"],
            auth_text,
            style=row_style,
        )

    return table


def render_search_interface(
    state: SearchState, show_help: bool = False, query: str = ""
) -> Layout:
    """Render the main search interface."""
    layout = create_search_layout()

    # Set header
    layout["header"].update(render_header(query))

    # Create results panel with legend
    table_result = render_results_table(state)
    method_legend = render_method_legend()
    results_content = Group(table_result, method_legend)

    layout["results"].update(
        Panel(
            results_content,
            title="[bold]Search Results",
            border_style="blue",
            box=box.ROUNDED,
            padding=(0, 1),
        )
    )

    # Update side panel based on state
    if show_help:
        layout["side_panel"].update(render_help())
    else:
        # Split side panel into three sections
        layout["side_panel"].split(
            Layout(name="details", ratio=3),
            Layout(name="related", ratio=2),
            Layout(name="stats", ratio=1),
        )

        # Update details section
        if state.current_result:
            layout["side_panel"]["details"].update(
                render_result_details(state.current_result)
            )
        else:
            layout["side_panel"]["details"].update(
                Panel(
                    "[yellow]No results found[/yellow]",
                    title="Details",
                    border_style="red",
                )
            )

        # Update related queries section
        if state.related_queries:
            related_text = Text()
            for query_info in state.related_queries:
                score = float(query_info.get("score", 0)) * 100
                score_color = (
                    "green" if score > 80 else "yellow" if score > 50 else "red"
                )
                related_text.append("\n• ", style="bold")
                related_text.append(query_info.get("query", ""), style=score_color)
                related_text.append(f" ({query_info.get('category', '')})", style="dim")
                if query_info.get("description"):
                    related_text.append(
                        f"\n  {query_info['description']}", style="italic"
                    )

            layout["side_panel"]["related"].update(
                Panel(
                    related_text,
                    title="[bold]Related Queries",
                    border_style="cyan",
                    box=box.ROUNDED,
                    padding=(1, 2),
                )
            )
        else:
            layout["side_panel"]["related"].update(
                Panel(
                    "[dim]No related queries[/dim]",
                    title="[bold]Related Queries",
                    border_style="cyan",
                    box=box.ROUNDED,
                    padding=(1, 2),
                )
            )

        # Update stats section
        layout["side_panel"]["stats"].update(render_stats(state))

    # Set footer
    layout["footer"].update(render_footer())

    return layout


def handle_keyboard_input(state: SearchState) -> bool:
    """Handle keyboard input and update state. Returns False to quit."""
    key = click.getchar()

    if key == "q":
        return False
    elif key in ["j", "\x1b[B"]:  # Down arrow
        state.next_result()
    elif key in ["k", "\x1b[A"]:  # Up arrow
        state.prev_result()
    elif key == "a":
        state.filter_auth_only = not state.filter_auth_only
    elif key == "s":
        choices = ["relevance", "method", "api", "path"]
        state.sort_by = choices[(choices.index(state.sort_by) + 1) % len(choices)]
    elif key == "f":
        state.filter_text = Prompt.ask("Enter filter text")
    elif key == "c" and state.current_result:
        # Copy endpoint path to clipboard
        path = state.current_result.get("path", "")
        click.echo(path, nl=False)
        console.print("[green]✓[/green] Copied to clipboard!")

    return True


@cli.command()
@click.argument("query")
@click.option("--top-k", default=10, help="Number of results")
@click.option("--rerank/--no-rerank", default=True, help="Use reranking")
@click.option("--cache/--no-cache", default=True, help="Use cache")
@click.option("-v", "--verbose", count=True, help="Increase output verbosity")
@click.option("--show-config", is_flag=True, help="Show configuration details")
def search(
    query: str, top_k: int, rerank: bool, cache: bool, verbose: int, show_config: bool
):
    """Enhanced search command with rich TUI."""
    try:
        # Setup logging based on verbosity
        setup_logging(verbose)

        # Show config if requested
        if show_config:
            if not Confirm.ask("Continue with search?"):
                return

        # Initialize search
        searcher = APISearcher(top_k=top_k, use_cache=cache)
        # Show search progress
        with Live(
            render_search_progress(),
            console=console,
            refresh_per_second=4,
            transient=True,
        ) as live:
            results = searcher.search(query=query, use_cache=cache)

        # Initialize state
        state = SearchState(results)
        show_help = False
        # Main interaction loop
        with Live(
            render_search_interface(state, show_help, query),
            console=console,
            refresh_per_second=4,
            screen=True,  # Use alternate screen
            auto_refresh=False,  # Manual refresh for better performance
        ) as live:
            while True:
                # Handle keyboard input
                if not handle_keyboard_input(state):
                    break
                # Update display
                live.update(render_search_interface(state, show_help, query))
                live.refresh()

    except Exception as e:
        console.print(
            Panel(f"[red]Error:[/red] {str(e)}", border_style="red", title="Error")
        )


@cli.command()
@click.argument("query")
def analyze(query: str):
    """Analyze a search query."""
    searcher = APISearcher()
    expander = QueryExpander()

    # Get query analysis
    analysis = searcher.analyze_query(query)

    console.print("\n[bold]Query Analysis:[/bold]")

    # Show semantic variants
    console.print("\n[cyan]Semantic Variants:[/cyan]")
    for variant, weight in analysis["weights"].items():
        console.print(f"- {variant}: {weight:.3f}")

    # Show technical mappings
    console.print("\n[cyan]Technical Mappings:[/cyan]")
    for mapping in analysis["technical_mappings"]:
        console.print(f"- {mapping}")

    # Show use cases
    console.print("\n[cyan]Relevant Use Cases:[/cyan]")
    for use_case in analysis["use_cases"]:
        console.print(f"- {use_case}")


@cli.command()
def health():
    """Check project health and consistency."""
    health_checker = Consistency()
    summary = health_checker.get_health_summary()

    console.print("\n[bold]Project Health Check:[/bold]")

    # Overall status
    status_color = "green" if summary["status"] == "healthy" else "red"
    console.print(f"\nStatus: [{status_color}]{summary['status']}[/{status_color}]")

    # Component status
    for component, status in summary.items():
        if component != "status":
            status_color = "green" if status == "ok" else "red"
            console.print(
                f"{component.title()}: [{status_color}]{status}[/{status_color}]"
            )

    # Show full check results
    results = health_checker.run_full_check()
    console.print("\n[bold]Detailed Results:[/bold]")
    console.print(json.dumps(results, indent=2))


@cli.command()
def metrics():
    """Show search quality metrics."""
    searcher = APISearcher()

    # Get current metrics
    current = searcher.get_quality_metrics()
    trends = searcher.get_metric_trends()

    console.print("\n[bold]Current Quality Metrics:[/bold]")
    for metric, value in current.items():
        console.print(f"{metric}: {value:.3f}")

    # Show trends
    console.print("\n[bold]Metric Trends (Last 30 Days):[/bold]")
    for metric, values in trends.items():
        if values:
            avg = sum(v for v in values if v is not None) / len([
                v for v in values if v is not None
            ])
            console.print(f"{metric} average: {avg:.3f}")


@cli.command()
@click.argument("endpoint_id")
def analyze_endpoint(endpoint_id: str):
    """Analyze a specific API endpoint."""
    searcher = APISearcher()
    understanding = ZeroShotUnderstanding()

    # Get endpoint data
    results, _ = searcher.search(query=f"id:{endpoint_id}", include_metadata=True)

    if not results:
        console.print("[red]Endpoint not found[/red]")
        return

    endpoint = results[0]

    # Analyze endpoint
    category = understanding.categorize_endpoint(endpoint)
    dependencies = understanding.get_api_dependencies(endpoint)
    similar = understanding.get_similar_endpoints(endpoint)
    alternatives = understanding.get_alternative_endpoints(endpoint)

    # Display results
    console.print("\n[bold]Endpoint Analysis:[/bold]")

    console.print("\n[cyan]Basic Information:[/cyan]")
    console.print(f"API: {endpoint.get('api_name', 'Unknown')}")
    console.print(f"Version: {endpoint.get('api_version', 'Unknown')}")
    console.print(f"Path: {endpoint.get('path', 'Unknown')}")
    console.print(f"Method: {endpoint.get('method', 'Unknown')}")

    console.print("\n[cyan]Category Information:[/cyan]")
    console.print(f"Primary Category: {category.name} ({category.confidence:.3f})")
    console.print("Subcategories:", ", ".join(category.subcategories))
    console.print("Features:", ", ".join(category.features))

    console.print("\n[cyan]Relationships:[/cyan]")
    console.print("Dependencies:", ", ".join(dependencies) or "None")
    console.print("Similar Endpoints:", ", ".join(similar) or "None")
    console.print("Alternative Endpoints:", ", ".join(alternatives) or "None")


@cli.command()
@click.argument("query")
@click.argument("endpoint_id")
@click.option("--relevant/--not-relevant", help="Whether the result was relevant")
@click.option("--score", default=1.0, help="Feedback score (0 to 1)")
def feedback(query: str, endpoint_id: str, relevant: bool, score: float):
    """Provide feedback for search results."""
    searcher = APISearcher()

    searcher.update_feedback(
        query=query, endpoint_id=endpoint_id, is_relevant=relevant, score=score
    )

    console.print("[green]Feedback recorded successfully[/green]")


@cli.command()
@click.option("--force", is_flag=True, help="Force reindexing of all files")
@click.option("--validate", is_flag=True, help="Validate data before indexing")
@click.option("-v", "--verbose", count=True, help="Increase output verbosity")
def index(force: bool, validate: bool, verbose: int):
    """Index API specifications from directory."""
    try:
        # Setup logging
        setup_logging(verbose)

        # Initialize indexer
        indexer = APIIndexer()

        # Show indexing progress
        with console.status("[bold green]Indexing APIs...") as status:
            result = indexer.index_directory(force=force, validate=validate)

        # Show summary
        console.print("\n[bold]Indexing Summary:[/bold]")
        console.print(f"Total files processed: [cyan]{result['total_files']}[/cyan]")
        console.print(
            f"Total endpoints indexed: [green]{result['total_endpoints']}[/green]"
        )

        # Show indexed APIs
        if result["indexed_apis"]:
            console.print("\n[bold]Indexed APIs:[/bold]")
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("API Name")
            table.add_column("Version")
            table.add_column("Endpoints", justify="right")
            table.add_column("File")

            for api in result["indexed_apis"]:
                table.add_row(
                    api["api_name"], api["version"], str(api["endpoints"]), api["path"]
                )
            console.print(table)

        # Show failed files
        if result["failed_files"]:
            console.print("\n[bold red]Failed Files:[/bold red]")
            for failure in result["failed_files"]:
                console.print(f"• {failure['path']}")
                console.print(f"  Error: {failure['error']}")
                if "details" in failure:
                    for detail in failure["details"]:
                        console.print(f"  - {detail}")

        # Show quality metrics if available
        if result["quality_metrics"]:
            metrics = result["quality_metrics"]
            console.print("\n[bold]Data Quality Metrics:[/bold]")
            console.print(f"Completeness: [cyan]{metrics['completeness']:.2%}[/cyan]")
            console.print(f"Consistency:  [cyan]{metrics['consistency']:.2%}[/cyan]")
            console.print(f"Accuracy:     [cyan]{metrics['accuracy']:.2%}[/cyan]")
            console.print(f"Uniqueness:   [cyan]{metrics['uniqueness']:.2%}[/cyan]")

    except Exception as e:
        console.print(f"\n[red]Error:[/red] {str(e)}")
        if verbose > 0:
            console.print_exception()
        raise click.Abort()


if __name__ == "__main__":
    cli()

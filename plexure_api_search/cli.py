"""Command line interface for API search."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

import click
from dotenv import load_dotenv
from rich.console import Console, Group, RenderableType
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.text import Text
from rich.live import Live
from rich.prompt import Prompt, Confirm
from rich.style import Style
from rich import box
from rich.align import Align

from .config import PINECONE_API_KEY, PINECONE_ENVIRONMENT
from .config import PINECONE_INDEX as PINECONE_INDEX_NAME
from .consistency import ProjectHealth
from .expansion import QueryExpander
from .indexer import APIIndexer
from .pinecone_client import PineconeClient
from .searcher import APISearcher
from .understanding import ZeroShotUnderstanding

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
        padding=(1, 2)
    )


def create_search_layout() -> Layout:
    """Create the main search layout."""
    layout = Layout(name="root")
    
    # Create a more compact layout
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1),
        Layout(name="footer", size=3)
    )
    
    # Split main area horizontally
    layout["main"].split_row(
        Layout(name="results", ratio=2, minimum_size=50),
        Layout(name="side_panel", ratio=1, minimum_size=30)
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
        padding=(1, 2)
    )


def render_result_details(result: Dict) -> Panel:
    """Render detailed view of a search result."""
    details = Text()
    
    # Score
    score_percentage = result['score'] * 100
    score_color = "green" if score_percentage > 80 else "yellow" if score_percentage > 50 else "red"
    details.append(f"\n💯 Relevance Score: ", style="bold")
    details.append(f"{score_percentage:.1f}%\n", style=f"bold {score_color}")
    
    # API Info
    details.append("\n📚 API Information\n", style="bold cyan")
    details.append(f"Name: {result['api_name']}\n", style="bright_blue")
    details.append(f"Version: {result['api_version']}\n", style="blue")
    
    # Endpoint Info
    details.append("\n🔗 Endpoint Details\n", style="bold magenta")
    method_colors = {
        "GET": "green", "POST": "yellow", "PUT": "blue",
        "DELETE": "red", "PATCH": "magenta"
    }
    details.append(f"Method: ", style="bright_magenta")
    details.append(f"{result['method']}\n", style=f"bold {method_colors.get(result['method'], 'white')}")
    details.append(f"Path: ", style="bright_magenta")
    details.append(f"{result['path']}\n", style="white")
    
    # Authentication
    auth_style = "bold red" if result["requires_auth"] else "bold green"
    auth_text = "Required" if result["requires_auth"] else "Not Required"
    auth_icon = "🔒" if result["requires_auth"] else "🔓"
    details.append(f"\n{auth_icon} Authentication\n", style="bold yellow")
    details.append(f"{auth_text}\n", style=auth_style)
    
    # Description
    if result["description"] or result["summary"]:
        details.append("\n📝 Description\n", style="bold green")
        desc = result["description"] or result["summary"]
        # Wrap description at 60 characters
        wrapped_desc = "\n".join(desc[i:i+60] for i in range(0, len(desc), 60))
        details.append(f"{wrapped_desc}\n", style="bright_green")
    
    # Parameters
    if result["parameters"]:
        details.append("\n⚙️ Parameters\n", style="bold blue")
        for param in result["parameters"]:
            param_parts = param.split(":")
            if len(param_parts) >= 3:
                name, type_, desc = param_parts[0], param_parts[1], ":".join(param_parts[2:])
                details.append(f"• ", style="bright_blue")
                details.append(f"{name}", style="bold bright_blue")
                details.append(f" ({type_})", style="cyan")
                details.append(f": {desc}\n", style="bright_blue")
            else:
                details.append(f"• {param}\n", style="bright_blue")
    
    # Responses
    if result["responses"]:
        details.append("\n📫 Responses\n", style="bold cyan")
        for response in result["responses"]:
            code, desc = response.split(":", 1) if ":" in response else (response, "")
            details.append(f"• ", style="bright_cyan")
            details.append(f"{code}", style="bold bright_cyan")
            if desc:
                details.append(f": {desc}", style="bright_cyan")
            details.append("\n")
    
    # Tags
    if result["tags"]:
        details.append("\n🏷️ Tags\n", style="bold magenta")
        details.append(" ".join(f"#{tag}" for tag in result["tags"]), style="bright_magenta")
    
    return Panel(
        details,
        title="[bold]Endpoint Details",
        subtitle="[dim]Press ↑/↓ to view other results",
        border_style="cyan",
        box=box.HEAVY,
        padding=(1, 2),
    )


def load_config() -> Dict[str, Any]:
    """Load configuration from file."""
    config_path = Path("config.json")
    if config_path.exists():
        with open(config_path, "r") as f:
            return json.load(f)

    # Create searcher config with only the required parameters
    searcher_config = {
        "index_name": PINECONE_INDEX_NAME,
        "api_key": PINECONE_API_KEY,
        "environment": PINECONE_ENVIRONMENT,
        "cloud": os.getenv("PINECONE_CLOUD", "aws"),
        "region": os.getenv("PINECONE_REGION", "us-east-1"),
    }

    return searcher_config


@click.group()
def cli():
    """API Search CLI"""
    pass


class SearchState:
    """Manage search state and navigation."""
    def __init__(self, results: List[Dict]):
        self.results = results
        self.selected_index = 0
        self.is_details_view = False
        self.filter_auth_only = False
        self.sort_by = "relevance"  # or "method", "api", "path"
        self.filter_text = ""
    
    @property
    def filtered_results(self) -> List[Dict]:
        """Get filtered results based on current state."""
        results = self.results
        
        # Apply auth filter
        if self.filter_auth_only:
            results = [r for r in results if r["requires_auth"]]
            
        # Apply text filter
        if self.filter_text:
            results = [
                r for r in results 
                if self.filter_text.lower() in r["path"].lower() 
                or self.filter_text.lower() in r["api_name"].lower()
            ]
            
        # Apply sorting
        if self.sort_by == "method":
            results = sorted(results, key=lambda x: x["method"])
        elif self.sort_by == "api":
            results = sorted(results, key=lambda x: x["api_name"])
        elif self.sort_by == "path":
            results = sorted(results, key=lambda x: x["path"])
        # Default is relevance (already sorted)
        
        return results
    
    @property
    def current_result(self) -> Optional[Dict]:
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
        padding=(1, 2)
    )


def render_stats(state: SearchState) -> Panel:
    """Render search statistics."""
    stats = Text()
    total = len(state.results)
    filtered = len(state.filtered_results)
    auth_count = sum(1 for r in state.filtered_results if r["requires_auth"])
    
    stats.append(f"\n📊 Results: ", style="bold yellow")
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
        padding=(1, 2)
    )


def render_method_legend() -> Panel:
    """Render HTTP method color legend."""
    legend = Text()
    method_colors = {
        "GET": "green",
        "POST": "yellow",
        "PUT": "blue",
        "DELETE": "red",
        "PATCH": "magenta"
    }
    
    for method, color in method_colors.items():
        legend.append(f"{method}", style=f"bold {color}")
        legend.append(" ")
    
    return Panel(
        Align.center(legend),
        title="[bold]HTTP Methods",
        border_style="blue",
        box=box.ROUNDED,
        padding=(1, 0)
    )


def render_results_table(state: SearchState) -> Table:
    """Render results table with current state."""
    table = Table(
        show_header=True,
        header_style="bold",
        box=box.ROUNDED,
        title=Text("Search Results", style="bold cyan"),
        expand=True,
        show_lines=True
    )
    
    # Add columns with sort indicators
    sort_indicator = "↓" if state.sort_by == "relevance" else ""
    table.add_column(f"Relevance {sort_indicator}", justify="center", style="cyan", width=10)
    
    sort_indicator = "↓" if state.sort_by == "api" else ""
    table.add_column(f"API {sort_indicator}", style="bright_blue")
    
    sort_indicator = "↓" if state.sort_by == "method" else ""
    table.add_column(f"Method {sort_indicator}", justify="center", width=8)
    
    sort_indicator = "↓" if state.sort_by == "path" else ""
    table.add_column(f"Path {sort_indicator}")
    
    table.add_column("Auth", justify="center", width=4)
    
    # Add rows
    for i, result in enumerate(state.filtered_results):
        # Highlight selected row
        style_suffix = " reverse" if i == state.selected_index else ""
        
        # Format score as percentage
        score = f"{result['score']*100:.1f}%"
        score_style = ("green" if result['score'] > 0.8 
                      else "yellow" if result['score'] > 0.5 
                      else "red") + style_suffix
        
        # Format method with color
        method_colors = {
            "GET": "green",
            "POST": "yellow",
            "PUT": "blue",
            "DELETE": "red",
            "PATCH": "magenta"
        }
        method = Text(result["method"], 
                     style=method_colors.get(result["method"], "white") + style_suffix)
        
        # Format auth icon
        auth = "🔒" if result["requires_auth"] else "🔓"
        
        table.add_row(
            Text(score, style=score_style),
            Text(result["api_name"], style="bright_blue" + style_suffix),
            method,
            Text(result["path"], style="white" + style_suffix),
            Text(auth, style="white" + style_suffix)
        )
    
    return table


def render_search_interface(state: SearchState, show_help: bool = False, query: str = "") -> Layout:
    """Render the main search interface."""
    layout = create_search_layout()
    
    # Set header
    layout["header"].update(render_header(query))
    
    # Create results panel with legend
    results_content = Group(
        render_results_table(state),
        render_method_legend()
    )
    
    layout["results"].update(Panel(
        results_content,
        title="[bold]Search Results",
        border_style="blue",
        box=box.ROUNDED,
        padding=(0, 1)
    ))
    
    # Update side panel based on state
    if show_help:
        layout["side_panel"].update(render_help())
    else:
        if state.current_result:
            layout["side_panel"].split(
                Layout(name="details", ratio=3),
                Layout(name="stats", ratio=1)
            )
            layout["side_panel"]["details"].update(render_result_details(state.current_result))
            layout["side_panel"]["stats"].update(render_stats(state))
        else:
            layout["side_panel"].update(Panel(
                "[yellow]No results found[/yellow]",
                title="Details",
                border_style="red"
            ))
    
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
        path = state.current_result["path"]
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
def search(query: str, top_k: int, rerank: bool, cache: bool, verbose: int, show_config: bool):
    """Enhanced search command with rich TUI."""
    try:
        # Setup logging based on verbosity
        setup_logging(verbose)
        
        # Initialize components
        config = load_config()
        
        # Show config if requested
        if show_config:
            console.print(render_config_info(config))
            if not Confirm.ask("Continue with search?"):
                return
        
        # Initialize search
        pinecone_client = PineconeClient(**config)
        searcher = APISearcher(pinecone_client)
        searcher.set_top_k(top_k)
        
        # Show search progress
        with Live(
            render_search_progress(),
            console=console,
            refresh_per_second=4,
            transient=True
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
            auto_refresh=False  # Manual refresh for better performance
        ) as live:
            while True:
                # Handle keyboard input
                if not handle_keyboard_input(state):
                    break
                
                # Update display
                live.update(render_search_interface(state, show_help, query))
                live.refresh()
        
    except Exception as e:
        console.print(Panel(
            f"[red]Error:[/red] {str(e)}",
            border_style="red",
            title="Error"
        ))


@cli.command()
@click.argument("query")
def analyze(query: str):
    """Analyze a search query."""
    config = load_config()
    searcher = APISearcher(**config)
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
    health_checker = ProjectHealth()
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
    config = load_config()
    searcher = APISearcher(**config)

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
    config = load_config()
    searcher = APISearcher(**config)
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
    console.print(f"API: {endpoint['api_name']}")
    console.print(f"Version: {endpoint['api_version']}")
    console.print(f"Path: {endpoint['path']}")
    console.print(f"Method: {endpoint['method']}")

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
    config = load_config()
    searcher = APISearcher(**config)

    searcher.update_feedback(
        query=query, endpoint_id=endpoint_id, is_relevant=relevant, score=score
    )

    console.print("[green]Feedback recorded successfully[/green]")


if __name__ == "__main__":
    cli()

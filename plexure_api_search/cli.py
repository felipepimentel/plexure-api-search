"""Command line interface for API search."""

import json
import os
from pathlib import Path
from typing import Any, Dict

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

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


@cli.command()
@click.option(
    "--force",
    is_flag=True,
    help="Force reindex all APIs",
)
def index(force: bool):
    """Index API contracts."""
    try:
        # Load configuration
        config = load_config()

        # Initialize Pinecone client
        pinecone_client = PineconeClient(
            api_key=config["api_key"],
            environment=config["environment"],
            index_name=config["index_name"],
            cloud=config.get("cloud", "aws"),
            region=config.get("region", "us-east-1"),
        )

        # Initialize and run indexer
        indexer = APIIndexer(pinecone_client=pinecone_client)
        indexer.index_apis(force=force)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise


@cli.command()
@click.argument("query")
@click.option("--top-k", default=10, help="Number of results to return")
@click.option(
    "--include-metadata", is_flag=True, help="Include full metadata in results"
)
@click.option("--analyze", is_flag=True, help="Show query analysis")
def search(query: str, top_k: int, include_metadata: bool, analyze: bool):
    """Search for API endpoints."""
    config = load_config()  # Will only get searcher-specific config
    searcher = APISearcher(**config)

    if analyze:
        analysis = searcher.analyze_query(query)
        console.print("\n[bold]Query Analysis:[/bold]")
        for key, value in analysis.items():
            console.print(f"[cyan]{key}:[/cyan] {value}")
        console.print()

    results, evaluation = searcher.search(
        query=query, include_metadata=include_metadata
    )

    # Display results
    table = Table(title="Search Results")
    table.add_column("Score", justify="right", style="cyan", no_wrap=True)
    table.add_column("API", style="green")
    table.add_column("Version", style="yellow")
    table.add_column("Method", style="magenta", no_wrap=True)
    table.add_column("Path", style="blue")
    table.add_column("Description", style="white")

    for result in results[:top_k]:
        try:
            # Debug print the raw result
            print("\nRaw result data:", json.dumps(result, indent=2))

            # Get values with defaults
            score = f"{result['score']:.3f}"
            api_name = result["api_name"]
            api_version = result["api_version"]
            method = result["method"]
            path = result["path"]
            description = result.get("description") or result.get(
                "summary", "No description"
            )

            # Truncate description if too long
            if len(description) > 100:
                description = description[:97] + "..."

            table.add_row(score, api_name, api_version, method, path, description)
        except Exception as e:
            print(f"Error processing result: {e}")
            print(f"Raw result data: {result}")
            continue

    console.print(table)

    # Print metrics
    console.print("\n[bold]Search Metrics:[/bold]")
    console.print(f"MRR: {evaluation.mrr:.3f}")
    console.print(f"NDCG: {evaluation.ndcg:.3f}")
    console.print(f"Precision@1: {evaluation.precision_at_1:.3f}")
    console.print(f"Precision@5: {evaluation.precision_at_5:.3f}")
    console.print(f"Recall: {evaluation.recall:.3f}")
    console.print(f"Latency: {evaluation.latency_ms:.2f}ms")


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

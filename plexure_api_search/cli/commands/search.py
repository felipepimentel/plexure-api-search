"""Search command for finding API endpoints."""

import click

from ...search import api_searcher

@click.command()
@click.argument("query")
@click.option(
    "--top-k",
    type=int,
    default=10,
    help="Number of results to return",
)
@click.option(
    "--min-score",
    type=float,
    default=0.5,
    help="Minimum similarity score",
)
@click.option(
    "--expand/--no-expand",
    default=True,
    help="Whether to expand query",
)
@click.option(
    "--rerank/--no-rerank",
    default=True,
    help="Whether to rerank results",
)
def search(
    query: str,
    top_k: int,
    min_score: float,
    expand: bool,
    rerank: bool,
) -> None:
    """Search for API endpoints."""
    results = api_searcher.search(
        query=query,
        top_k=top_k,
        min_score=min_score,
        expand_query=expand,
        rerank_results=rerank,
    )

    if not results:
        click.echo("No results found.")
        return

    for result in results:
        click.echo(f"\nScore: {result['score']:.3f}")
        click.echo(f"Method: {result.get('method', 'N/A')}")
        click.echo(f"Endpoint: {result.get('endpoint', 'N/A')}")
        if description := result.get('description'):
            click.echo(f"Description: {description}")
        if parameters := result.get('parameters'):
            click.echo("\nParameters:")
            for param in parameters:
                click.echo(f"  - {param['name']}: {param['type']}")
                if param_desc := param.get('description'):
                    click.echo(f"    {param_desc}")
        click.echo("-" * 80) 
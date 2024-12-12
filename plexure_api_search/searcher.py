"""API Search module for searching and displaying API endpoints."""

import json
import httpx
from typing import Dict, List, Any
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

from plexure_api_search.constants import (
    OPENROUTER_URL,
    PINECONE_ENVIRONMENT,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    OPENROUTER_API_KEY,
    DEFAULT_TOP_K,
    QUERY_ANALYSIS_PROMPT,
    RELEVANCE_EXPLANATION_PROMPT,
    SEARCH_SUMMARY_PROMPT,
    METHOD_COLORS,
    SCORE_ADJUSTMENTS,
    FEATURE_ICONS,
    HTTP_HEADERS,
    SENTENCE_TRANSFORMER_MODEL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    TABLE_SETTINGS,
    ERROR_MESSAGES
)

class APISearcher:
    """Handles API endpoint searching and result display."""
    
    def __init__(self):
        self.console = Console()
        
        # Validate environment variables
        if not PINECONE_API_KEY:
            raise ValueError(ERROR_MESSAGES["api_key_missing"].format(key="PINECONE_API_KEY"))
        if not OPENROUTER_API_KEY:
            raise ValueError(ERROR_MESSAGES["api_key_missing"].format(key="OPENROUTER_API_KEY"))
        
        self.model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
        self.headers = {
            **HTTP_HEADERS,
            "Authorization": f"Bearer {OPENROUTER_API_KEY}"
        }
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index = self.pc.Index(PINECONE_INDEX_NAME)

    def _calculate_custom_score(self, vector_score: float, metadata: Dict[str, Any], query_analysis: Dict[str, Any]) -> float:
        """Calculate custom relevance score based on query analysis and ranking priorities."""
        score = vector_score
        search_params = query_analysis.get("search_parameters", {})
        
        # Version match bonus
        search_version = search_params.get("version")
        api_version = metadata.get("api_version")
        if search_version and api_version and search_version == api_version:
            score += SCORE_ADJUSTMENTS['version_match']
        
        # Method match bonus
        search_method = search_params.get("method", "")
        api_method = metadata.get("method", "")
        if search_method and api_method and search_method.upper() == api_method.upper():
            score += SCORE_ADJUSTMENTS['method_match']
        
        # Path relevance bonus
        api_path = metadata.get("path", "").lower()
        if api_path:
            for path_term in search_params.get("path_contains", []):
                if path_term and path_term.lower() in api_path:
                    score += SCORE_ADJUSTMENTS['path_match']
        
        # Feature match bonus
        api_description = metadata.get("description", "").lower()
        api_summary = metadata.get("summary", "").lower()
        for feature in search_params.get("features", []):
            if feature:
                feature_lower = feature.lower()
                if (api_description and feature_lower in api_description) or \
                   (api_summary and feature_lower in api_summary):
                    score += SCORE_ADJUSTMENTS['feature_match']
        
        # Metadata filters bonus
        for key, value in search_params.get("metadata_filters", {}).items():
            if value is not None and key in metadata:
                metadata_val = metadata.get(key)
                if metadata_val is not None and str(metadata_val).lower() == str(value).lower():
                    score += SCORE_ADJUSTMENTS['metadata_match']
        
        return min(1.0, score)  # Normalize to max 1.0

    def _get_method_color(self, method: str) -> str:
        """Get color for HTTP method."""
        return METHOD_COLORS.get(method.upper(), 'white')

    def _get_version_distribution(self, results: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get distribution of API versions in results."""
        versions = {}
        for result in results:
            version = result["data"].get("api_version", "unknown")
            versions[version] = versions.get(version, 0) + 1
        return versions

    def _get_method_distribution(self, results: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get distribution of HTTP methods in results."""
        methods = {}
        for result in results:
            method = result["data"].get("method", "unknown")
            methods[method] = methods.get(method, 0) + 1
        return methods

    def _get_feature_summary(self, results: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get summary of API features in results."""
        features = {key: 0 for key in FEATURE_ICONS.keys()}
        
        for result in results:
            data = result["data"]
            for feature in features:
                if str(data.get(feature, "false")).lower() == "true":
                    features[feature] += 1
        
        return features

    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze user query to understand intent and enhance search."""
        try:
            response = httpx.post(
                OPENROUTER_URL,
                headers=self.headers,
                json={
                    "model": LLM_MODEL,
                    "messages": [
                        {"role": "system", "content": QUERY_ANALYSIS_PROMPT},
                        {"role": "user", "content": query},
                    ],
                    "temperature": LLM_TEMPERATURE["query_analysis"],
                },
            )

            if response.status_code == 200:
                content = response.json()["choices"][0]["message"]["content"]
                return json.loads(content)

        except Exception as e:
            self.console.print(
                ERROR_MESSAGES["query_analysis_failed"].format(error=str(e))
            )

        return {"enhanced_query": query}

    def _explain_relevance(
        self, query_analysis: Dict[str, Any], metadata: Dict[str, Any]
    ) -> str:
        """Generate a detailed explanation of why this result is relevant."""
        try:
            # Safely get values with null checks
            search_params = query_analysis.get("search_parameters", {})
            method = metadata.get("method", "")
            path = metadata.get("path", "")
            version = metadata.get("api_version")
            description = metadata.get("description", "")
            summary = metadata.get("summary", "")

            # Safe comparison for version match
            version_match = (
                search_params.get("version") is not None
                and version is not None
                and search_params.get("version") == version
            )

            # Safe comparison for method match
            search_method = search_params.get("method", "")
            method_match = (
                search_method and method and search_method.upper() == method.upper()
            )

            # Safe path matches check
            path_matches = []
            for term in search_params.get("path_contains", []):
                if term and path and term.lower() in path.lower():
                    path_matches.append(term)

            # Safe feature matches check
            feature_matches = []
            for feature in search_params.get("features", []):
                if feature and (
                    (description and feature.lower() in description.lower())
                    or (summary and feature.lower() in summary.lower())
                ):
                    feature_matches.append(feature)

            # Safe metadata matches check
            metadata_matches = {}
            for key, value in search_params.get("metadata_filters", {}).items():
                if value is not None and key in metadata:
                    metadata_val = metadata.get(key)
                    if (
                        metadata_val is not None
                        and str(metadata_val).lower() == str(value).lower()
                    ):
                        metadata_matches[key] = value

            context = {
                "query_analysis": query_analysis,
                "metadata": metadata,
                "match_details": {
                    "version_match": version_match,
                    "method_match": method_match,
                    "path_matches": path_matches,
                    "feature_matches": feature_matches,
                    "metadata_matches": metadata_matches,
                },
            }

            response = httpx.post(
                OPENROUTER_URL,
                headers=self.headers,
                json={
                    "model": LLM_MODEL,
                    "messages": [
                        {"role": "system", "content": RELEVANCE_EXPLANATION_PROMPT},
                        {"role": "user", "content": json.dumps(context)},
                    ],
                    "temperature": LLM_TEMPERATURE["relevance_explanation"],
                },
            )

            if response.status_code == 200:
                explanation = response.json()["choices"][0]["message"]["content"]
                return explanation

        except Exception as e:
            self.console.print(ERROR_MESSAGES["relevance_failed"].format(error=str(e)))

        return "No relevance explanation available."

    def _display_search_summary(self, results: List[Dict[str, Any]]) -> None:
        """Display a natural language summary of search results."""
        try:
            # Create context for the LLM
            context = {
                "total_results": len(results),
                "top_results": [
                    {
                        "method": r["data"]["method"],
                        "path": r["data"]["path"],
                        "version": r["data"]["api_version"],
                        "description": r["data"]["description"],
                        "score": r["score"],
                        "original_score": r["original_score"],
                    }
                    for r in results[:3]
                ],
                "version_distribution": self._get_version_distribution(results),
                "method_distribution": self._get_method_distribution(results),
                "feature_summary": self._get_feature_summary(results),
            }

            response = httpx.post(
                OPENROUTER_URL,
                headers=self.headers,
                json={
                    "model": LLM_MODEL,
                    "messages": [
                        {"role": "system", "content": SEARCH_SUMMARY_PROMPT},
                        {"role": "user", "content": json.dumps(context)},
                    ],
                    "temperature": LLM_TEMPERATURE["search_summary"],
                },
            )

            if response.status_code == 200:
                summary = response.json()["choices"][0]["message"]["content"]
                self.console.print(f"\n{summary}")

        except Exception as e:
            self.console.print(ERROR_MESSAGES["summary_failed"].format(error=str(e)))

    def display_results(self, results: List[Dict[str, Any]]) -> None:
        """Display search results in a readable format."""
        if not results:
            self.console.print(ERROR_MESSAGES["no_results"])
            return

        # First, get an overall summary
        self._display_search_summary(results)

        self.console.print("\n[bold blue]Detailed Results:[/]")

        for i, result in enumerate(results, 1):
            data = result["data"]

            # Create panel for each result
            table = Table(show_header=False, box=None)
            table.add_column(
                "Key",
                style=TABLE_SETTINGS["key_column_style"],
                width=TABLE_SETTINGS["key_column_width"],
            )
            table.add_column("Value")

            # Add basic info
            table.add_row(
                "Method",
                f"[bold {self._get_method_color(data['method'])}]{data['method']}[/]",
            )
            table.add_row("Path", data["path"])
            table.add_row("Version", f"[bold]{data.get('api_version', 'unknown')}[/]")
            table.add_row(
                "Score",
                f"[bold green]{result['score']:.2f}[/] (vector: {result['original_score']:.2f})",
            )

            # Add description if available
            if data.get("description"):
                table.add_row("Description", Markdown(data["description"]))

            # Add relevance explanation
            if result.get("relevance_explanation"):
                table.add_row("Relevance", Markdown(result["relevance_explanation"]))

            # Add features section
            features = []
            for feature, info in FEATURE_ICONS.items():
                if str(data.get(feature, "false")).lower() == "true":
                    features.append(
                        f"[{info['style']}]{info['icon']} {info['label']}[/]"
                    )

            if features:
                table.add_row("Features", " ".join(features))

            # Add tags if available
            if data.get("tags"):
                table.add_row(
                    "Tags", ", ".join(f"[bold cyan]#{tag}[/]" for tag in data["tags"])
                )

            # Add parameters if available
            if data.get("parameters"):
                params_table = Table(title="Parameters", show_header=True)
                params_table.add_column("Name", style="cyan")
                params_table.add_column("Type", style="green")
                params_table.add_column("Required", style="yellow")
                params_table.add_column("Description")

                for param in data["parameters"]:
                    name, param_type = param.split(":")
                    params_table.add_row(name, param_type, "", "")

                table.add_row("Parameters", params_table)

            # Create panel with the table
            panel = Panel(
                table,
                title=f"[bold]{i}. {data['method']} {data['path']}[/]",
                border_style=self._get_method_color(data["method"]),
            )
            self.console.print(panel)
            self.console.print()

    def search(self, query: str) -> List[Dict[str, Any]]:
        """Search for endpoints using hybrid search and enhance with LLM interpretation."""
        try:
            # First, analyze the query to understand user intent
            query_analysis = self._analyze_query(query)

            # Apply filters based on query analysis
            filters = {}
            search_params = query_analysis.get("search_parameters", {})

            if search_params.get("version"):
                filters["api_version"] = search_params["version"]
            if search_params.get("method"):
                filters["method"] = search_params["method"].upper()

            metadata_filters = search_params.get("metadata_filters", {})
            for key, value in metadata_filters.items():
                if value is not None:
                    filters[key] = str(value)

            # Perform hybrid search with filters
            search_results = self.index.query(
                vector=self.model.encode(
                    query_analysis.get("enhanced_query", query)
                ).tolist(),
                top_k=DEFAULT_TOP_K,
                include_metadata=True,
                filter=filters if filters else None,
            )

            # Format and rank results
            results = []
            ranking_priorities = query_analysis.get("ranking_priorities", [])

            for match in search_results.matches:
                # Calculate custom score based on ranking priorities
                custom_score = self._calculate_custom_score(
                    match.score, match.metadata, query_analysis
                )

                results.append({
                    "id": f"{match.metadata['method']} {match.metadata['path']}",
                    "score": custom_score,
                    "original_score": match.score,
                    "data": match.metadata,
                    "relevance_explanation": self._explain_relevance(
                        query_analysis, match.metadata
                    ),
                })

            # Sort by custom score
            results.sort(key=lambda x: x["score"], reverse=True)

            return results

        except Exception as e:
            self.console.print(ERROR_MESSAGES["search_failed"].format(error=str(e)))
            return []


def main():
    """Main function for running the searcher directly."""
    import argparse

    parser = argparse.ArgumentParser(description="Search API endpoints")
    parser.add_argument("query", help="Search query")

    args = parser.parse_args()

    searcher = APISearcher()
    results = searcher.search(args.query)
    searcher.display_results(results)


if __name__ == "__main__":
    main()

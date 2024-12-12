"""Main module for the Plexure API Search tool."""

import argparse
import glob
import json
import os
import time
import traceback
from typing import Any, Dict, List

import httpx
import yaml
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from rich.table import Table
from sentence_transformers import SentenceTransformer

# Initialize Rich console
console = Console()

# Load environment variables from .env file
load_dotenv()

from .constants import (
    DEFAULT_API_DIR,
    DEFAULT_HEADERS,
    DEFAULT_TOP_K,
    OPENROUTER_URL,
)


class QueryUnderstanding:
    def __init__(self):
        self.headers = DEFAULT_HEADERS

    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze the query to understand user intent and extract parameters"""
        try:
            response = httpx.post(
                OPENROUTER_URL,
                headers=self.headers,
                json={
                    "model": "mistralai/mistral-7b-instruct",
                    "messages": [
                        {
                            "role": "system",
                            "content": """Analyze the API search query and extract structured information.
                            Focus on identifying:
                            1. Main topic/domain
                            2. Specific functionality
                            3. Technical requirements
                            4. Performance expectations
                            5. Business context""",
                        },
                        {"role": "user", "content": query},
                    ],
                },
            )

            if response.status_code == 200:
                analysis = response.json()["choices"][0]["message"]["content"]
                return {"original_query": query, "analysis": analysis}
            else:
                print(f"Error analyzing query: {response.status_code}")
                return {"original_query": query, "analysis": None}

        except Exception as e:
            print(f"Error in query analysis: {str(e)}")
            return {"original_query": query, "analysis": None}


class APISearchEngine:
    def __init__(
        self,
        api_dir: str = DEFAULT_API_DIR,
        force_reindex: bool = False,
        skip_index: bool = False,
    ):
        self.api_dir = api_dir
        self.console = console

        # Get environment variables
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
        self.index_name = os.getenv("PINECONE_INDEX_NAME")

        if not all([self.pinecone_api_key, self.pinecone_environment, self.index_name]):
            raise ValueError(
                "Environment variables PINECONE_API_KEY, PINECONE_ENVIRONMENT and PINECONE_INDEX_NAME are required"
            )

        # Initialize Pinecone
        self.console.print("\n[bold blue]Initializing Pinecone...[/]")
        self.pc = Pinecone(api_key=self.pinecone_api_key)

        # Initialize or get Pinecone index
        try:
            if self.index_name not in self.pc.list_indexes().names():
                self.console.print(f"[yellow]Creating new index:[/] {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=384,  # dimension of all-MiniLM-L6-v2 embeddings
                    metric="dotproduct",  # Changed from cosine to dotproduct to support hybrid search
                    spec=ServerlessSpec(cloud="aws", region=self.pinecone_environment),
                )
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    transient=True,
                ) as progress:
                    progress.add_task("Waiting for index to be ready...", total=None)
                    while not self.pc.describe_index(self.index_name).status["ready"]:
                        time.sleep(1)
            else:
                self.console.print(f"[green]Using existing index:[/] {self.index_name}")
        except Exception as e:
            self.console.print(f"[red]Error initializing Pinecone index:[/] {str(e)}")
            raise

        # Initialize components
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.index = self.pc.Index(self.index_name)
        self.query_understanding = QueryUnderstanding()

        if not skip_index:
            self.console.print(
                f"\n[bold blue]Initializing with API contracts from directory:[/] {api_dir}"
            )
            self.initialize_with_contracts(force_reindex)
        else:
            self.console.print(
                "\n[yellow]Skipping indexing, using existing index...[/]"
            )

    def _load_api_specs(self) -> List[Dict[str, Any]]:
        """Load API specifications from YAML files"""
        specs = []

        # Get all YAML files (both .yml and .yaml extensions)
        yaml_patterns = [
            os.path.join(self.api_dir, "**/*.yml"),
            os.path.join(self.api_dir, "**/*.yaml"),
        ]

        yaml_files = []
        for pattern in yaml_patterns:
            yaml_files.extend(glob.glob(pattern, recursive=True))

        self.console.print(
            f"\n[bold]Processing {len(yaml_files)} API specification files from {self.api_dir}[/]"
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
        ) as progress:
            task = progress.add_task("Loading files...", total=len(yaml_files))

            for yaml_file in yaml_files:
                try:
                    with open(yaml_file, "r") as f:
                        spec = yaml.safe_load(f)
                        if spec:
                            specs.append(spec)
                            progress.update(
                                task,
                                advance=1,
                                description=f"Loaded: {os.path.basename(yaml_file)}",
                            )
                except Exception as e:
                    self.console.print(f"[red]Error loading {yaml_file}:[/] {str(e)}")

        return specs

    def _extract_text_from_details(self, details: Any) -> str:
        """Extract text from API details handling different data structures"""
        if isinstance(details, dict):
            summary = details.get("summary", "")
            description = details.get("description", "")
            return f"{summary}\n{description}"
        elif isinstance(details, list):
            # If details is a list, try to extract text from each item
            texts = []
            for item in details:
                if isinstance(item, dict):
                    texts.append(item.get("summary", ""))
                    texts.append(item.get("description", ""))
                elif isinstance(item, str):
                    texts.append(item)
            return "\n".join(filter(None, texts))
        elif isinstance(details, str):
            return details
        return ""

    def _process_spec(self, spec: Dict[str, Any]) -> None:
        """Process a single API spec and add it to the index"""
        try:
            # Extract paths and their operations
            paths = spec.get("paths", {})
            for path, operations in paths.items():
                if not isinstance(operations, dict):
                    self.console.print(
                        f"[yellow]Warning: Skipping invalid path {path}[/]"
                    )
                    continue

                for method, details in operations.items():
                    # Skip if not an HTTP method
                    if method.startswith("$") or method in [
                        "parameters",
                        "servers",
                        "summary",
                        "description",
                    ]:
                        continue

                    try:
                        # Create text for embedding
                        text = f"{method} {path}\n"
                        text += self._extract_text_from_details(details)

                        # Generate embedding
                        embedding = self.model.encode(text).tolist()

                        # Create metadata in Pinecone-compatible format
                        meta = self._prepare_metadata(details, path, method, spec)

                        # Add to index
                        endpoint_id = f"vec_{len(self.index)}"
                        self.index.upsert(
                            vectors=[(endpoint_id, embedding, meta)], namespace=""
                        )

                        self.console.print(
                            f"[green]Indexed:[/] {method.upper()} {path}"
                        )

                    except Exception as e:
                        self.console.print(
                            f"[yellow]Warning: Error processing endpoint {method} {path}: {str(e)}[/]"
                        )
                        continue

        except Exception as e:
            self.console.print(f"[red]Error processing spec: {str(e)}[/]")

    def _prepare_metadata(
        self, details: Dict[str, Any], path: str, method: str, spec: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare metadata according to Pinecone's requirements"""
        try:
            # Extract and format tags
            operation_tags = [str(tag) for tag in details.get("tags", [])]
            global_tags = []
            for tag in spec.get("tags", []):
                if isinstance(tag, dict):
                    tag_name = tag.get("name", "")
                    if tag_name:
                        global_tags.append(tag_name)
                elif isinstance(tag, str):
                    global_tags.append(tag)

            # Extract basic metadata fields
            metadata = {
                "path": path,
                "method": method.upper(),
                "full_path": path,  # Full path including base path
                "api": str(spec.get("info", {}).get("title", "")),
                "api_version": str(spec.get("info", {}).get("version", "")),
                "api_description": str(spec.get("info", {}).get("description", "")),
                "api_contact_name": str(
                    spec.get("info", {}).get("contact", {}).get("name", "")
                ),
                "api_contact_email": str(
                    spec.get("info", {}).get("contact", {}).get("email", "")
                ),
                "api_base_path": str(spec.get("servers", [{}])[0].get("url", "")),
                "summary": str(details.get("summary", "")),
                "description": str(details.get("description", "")),
                "operationId": str(details.get("operationId", "")),
                "deprecated": str(details.get("deprecated", False)),
                "category": self._extract_category(details, method),
                "complexity_score": float(self._calculate_complexity_score(details)),
                "tags": operation_tags,
                "global_tags": global_tags,
                "has_auth": str(self._has_security(details, spec)),
                "has_examples": str(
                    bool(details.get("examples") or details.get("example"))
                ),
                "has_schema": str(
                    bool(
                        details.get("requestBody", {})
                        .get("content", {})
                        .get("application/json", {})
                        .get("schema")
                    )
                ),
                "supports_pagination": str(self._supports_pagination(details)),
                "success_response": str(self._has_success_response(details)),
                "security_schemes": [
                    str(scheme) for scheme in self._get_security_schemes(details, spec)
                ],
                "parameters_count": len(self._get_all_parameters(details, path, spec)),
                "path_parameters": len([
                    p
                    for p in self._get_all_parameters(details, path, spec)
                    if isinstance(p, dict) and p.get("in") == "path"
                ]),
                "query_parameters": len([
                    p
                    for p in self._get_all_parameters(details, path, spec)
                    if isinstance(p, dict) and p.get("in") == "query"
                ]),
                "required_parameters": len([
                    p
                    for p in self._get_all_parameters(details, path, spec)
                    if isinstance(p, dict) and p.get("required", False)
                ]),
                "response_codes": [
                    str(code) for code in details.get("responses", {}).keys()
                ],
                "file": str(spec.get("info", {}).get("title", "")) + ".yaml",
            }

            # Add parameters details
            parameters = []
            for param in self._get_all_parameters(details, path, spec):
                try:
                    if isinstance(param, dict):
                        param_info = {
                            "name": str(param.get("name", "")),
                            "in": str(param.get("in", "")),
                            "required": str(param.get("required", False)),
                            "type": str(param.get("schema", {}).get("type", "")),
                            "description": str(param.get("description", "")),
                        }
                        parameters.append(f"{param_info['name']}:{param_info['type']}")
                except Exception as e:
                    self.console.print(
                        f"[yellow]Warning: Error processing parameter {param}: {str(e)}[/]"
                    )
                    continue

            metadata["parameters"] = parameters

            # Ensure all values are simple types
            for key, value in metadata.items():
                if isinstance(value, (dict, list)):
                    if not value:
                        metadata[key] = []  # Empty list for empty collections
                    elif isinstance(value, list) and not all(
                        isinstance(x, (str, int, float, bool)) for x in value
                    ):
                        metadata[key] = [
                            str(x) for x in value
                        ]  # Convert all list items to strings
                elif not isinstance(value, (str, int, float, bool)):
                    metadata[key] = str(value)  # Convert any other types to string

            return metadata

        except Exception as e:
            self.console.print(
                f"[yellow]Warning: Error preparing metadata for {method} {path}: {str(e)}[/]"
            )
            return self._get_minimal_metadata(path, method)

    def _get_minimal_metadata(self, path: str, method: str) -> Dict[str, Any]:
        """Return minimal valid metadata"""
        return {
            "path": path,
            "method": method.upper(),
            "full_path": path,
            "api": "",
            "api_version": "",
            "api_description": "",
            "api_contact_name": "",
            "api_contact_email": "",
            "api_base_path": "",
            "summary": "",
            "description": "",
            "operationId": "",
            "deprecated": "false",
            "category": "",
            "complexity_score": 1.0,
            "tags": [],
            "global_tags": [],
            "has_auth": "false",
            "has_examples": "false",
            "has_schema": "false",
            "supports_pagination": "false",
            "success_response": "false",
            "security_schemes": [],
            "parameters_count": 0,
            "path_parameters": 0,
            "query_parameters": 0,
            "required_parameters": 0,
            "response_codes": [],
            "file": "",
            "parameters": [],
        }

    def _extract_category(self, details: Dict[str, Any], method: str = "") -> str:
        """Extract operation category based on method and path"""
        # Use provided method or try to get from details
        operation_method = method.upper() or details.get("method", "").upper()

        if operation_method == "GET":
            return "read"
        elif operation_method == "POST":
            return "create"
        elif operation_method == "PUT":
            return "update"
        elif operation_method == "DELETE":
            return "delete"
        elif operation_method == "PATCH":
            return "update"
        return "other"

    def _calculate_complexity_score(self, details: Dict[str, Any]) -> float:
        """Calculate API complexity score based on various factors"""
        score = 1.0

        # Add complexity for parameters
        params = details.get("parameters", [])
        score += len(params) * 0.1

        # Add complexity for required parameters
        required_params = [p for p in params if p.get("required", False)]
        score += len(required_params) * 0.2

        # Add complexity for authentication
        if self._has_security(details, {}):
            score += 0.5

        # Add complexity for request body
        if details.get("requestBody"):
            score += 0.3

        # Add complexity for multiple response codes
        responses = details.get("responses", {})
        score += len(responses) * 0.1

        return round(score, 2)

    def _has_security(self, details: Dict[str, Any], spec: Dict[str, Any]) -> bool:
        """Check if endpoint requires authentication"""
        # Check operation level security
        if details.get("security"):
            return True

        # Check global security
        if spec.get("security"):
            return True

        return False

    def _supports_pagination(self, details: Dict[str, Any]) -> bool:
        """Check if endpoint supports pagination"""
        params = details.get("parameters", [])
        pagination_params = ["page", "limit", "offset", "size", "per_page"]
        return any(p.get("name", "").lower() in pagination_params for p in params)

    def _has_success_response(self, details: Dict[str, Any]) -> bool:
        """Check if endpoint has success response codes"""
        responses = details.get("responses", {})
        success_codes = ["200", "201", "202", "204"]
        return any(code in success_codes for code in responses.keys())

    def _get_security_schemes(
        self, details: Dict[str, Any], spec: Dict[str, Any]
    ) -> List[str]:
        """Get list of security schemes used by endpoint"""
        schemes = []

        # Get operation level security
        operation_security = details.get("security", [])
        for security in operation_security:
            schemes.extend(security.keys())

        # Get global security
        global_security = spec.get("security", [])
        for security in global_security:
            schemes.extend(security.keys())

        return list(set(schemes))  # Remove duplicates

    def _get_all_parameters(
        self, details: Dict[str, Any], path: str, spec: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Get all parameters from path and operation levels"""
        parameters = []

        # Get path parameters
        path_params = spec.get("paths", {}).get(path, {}).get("parameters", [])
        if isinstance(path_params, list):
            for param in path_params:
                if isinstance(param, dict):
                    parameters.append(param)

        # Get operation parameters
        operation_params = details.get("parameters", [])
        if isinstance(operation_params, list):
            for param in operation_params:
                if isinstance(param, dict):
                    parameters.append(param)

        return parameters

    def initialize_with_contracts(self, force_reindex: bool = False) -> None:
        """Initialize the search engine with API contracts"""
        try:
            # Check if reindexing is needed
            if not force_reindex:
                try:
                    stats = self.index.describe_index_stats()
                    if stats.total_vector_count > 0:
                        print("Using existing Pinecone index")
                        return
                except Exception as e:
                    print(f"Warning: Error checking index stats: {str(e)}")

            # Load and process API specs
            specs = self._load_api_specs()

            if force_reindex:
                print("Force reindex requested, clearing existing vectors...")
                try:
                    self.index.delete(delete_all=True, namespace="")
                except Exception as e:
                    print(f"Warning: Error clearing index: {str(e)}")

            # Process and upsert vectors
            vectors = []
            metadata = []

            for spec in specs:
                paths = spec.get("paths", {})
                for path, operations in paths.items():
                    for method, details in operations.items():
                        try:
                            # Create text for embedding
                            text = f"{method} {path}\n"
                            text += self._extract_text_from_details(details)

                            # Generate embedding
                            embedding = self.model.encode(text).tolist()

                            # Create metadata in Pinecone-compatible format
                            meta = self._prepare_metadata(details, path, method, spec)

                            vectors.append(embedding)
                            metadata.append(meta)
                        except Exception as e:
                            print(
                                f"Warning: Error processing endpoint {method} {path}: {str(e)}"
                            )
                            continue

            if not vectors:
                print("No vectors to index!")
                return

            print(f"Indexing {len(vectors)} vectors...")

            # Upsert in batches
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch_vectors = vectors[i : i + batch_size]
                batch_metadata = metadata[i : i + batch_size]

                vector_ids = [f"vec_{j}" for j in range(i, i + len(batch_vectors))]

                try:
                    self.index.upsert(
                        vectors=list(zip(vector_ids, batch_vectors, batch_metadata)),
                        namespace="",
                    )
                    print(
                        f"Indexed batch {i // batch_size + 1}/{(len(vectors) - 1) // batch_size + 1}"
                    )
                except Exception as e:
                    print(f"Error upserting batch: {str(e)}")
                    print("First metadata example:")
                    if batch_metadata:
                        print(json.dumps(batch_metadata[0], indent=2))
                    raise

            print(f"Successfully indexed {len(vectors)} endpoints")

        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            raise

    def search(self, query: str) -> List[Dict[str, Any]]:
        """Search for endpoints using hybrid search"""
        try:
            # Analyze query for better understanding
            query_analysis = self.query_understanding.analyze_query(query)
            enhanced_query = query_analysis.get("analysis", query)

            # Generate query embedding
            query_embedding = self.model.encode(enhanced_query).tolist()

            # Perform hybrid search
            search_results = self.index.query(
                vector=query_embedding, top_k=DEFAULT_TOP_K, include_metadata=True
            )

            # Format results
            results = []
            for match in search_results.matches:
                results.append({
                    "id": f"{match.metadata['method']} {match.metadata['path']}",
                    "score": match.score,
                    "data": match.metadata,
                })

            return results

        except Exception as e:
            print(f"Error during search: {str(e)}")
            return []

    def display_results(self, results: List[Dict[str, Any]]) -> None:
        """Display search results in a readable format"""
        if not results:
            self.console.print("\n[yellow]No matching endpoints found.[/]")
            return

        self.console.print("\n[bold blue]Search Results:[/]")

        for result in results:
            data = result["data"]

            # Create panel for each result
            table = Table(show_header=False, box=None)
            table.add_column("Key", style="bold cyan")
            table.add_column("Value")

            # Add basic info
            table.add_row("Method", data["method"])
            table.add_row("Path", data["path"])
            table.add_row("Score", f"{result['score']:.2f}")

            # Add description if available
            if data.get("description"):
                table.add_row("Description", Markdown(data["description"]))

            # Add tags if available
            if data.get("tags"):
                table.add_row("Tags", ", ".join(data["tags"]))

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
                title=f"[bold]{data['method']} {data['path']}[/]",
                border_style="blue",
            )
            self.console.print(panel)
            self.console.print()


def run_single_search(searcher: APISearchEngine, query: str) -> None:
    """Execute a single search and display results."""
    results = searcher.search(query)
    searcher.display_results(results)


def run_interactive_mode(searcher: APISearchEngine) -> None:
    """Run in interactive mode, accepting queries until exit."""
    while True:
        query = input("\nEnter your search query (or 'exit' to quit): ")
        if query.lower() in ["exit", "quit", "q"]:
            break

        run_single_search(searcher, query)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plexure API Search Tool")
    parser.add_argument(
        "--api-dir", default=DEFAULT_API_DIR, help="Directory containing API specs"
    )
    parser.add_argument(
        "--force-reindex", action="store_true", help="Force reindexing of API specs"
    )
    parser.add_argument(
        "--skip-index", action="store_true", help="Skip indexing and use existing index"
    )
    parser.add_argument(
        "--query",
        default="find pet endpoints",
        help="Search query. If not provided, runs in interactive mode",
    )

    args = parser.parse_args()

    try:
        searcher = APISearchEngine(
            api_dir=args.api_dir,
            force_reindex=args.force_reindex,
            skip_index=args.skip_index,
        )

        if args.query:
            # Run single query mode
            run_single_search(searcher, args.query)
        else:
            # Run interactive mode
            run_interactive_mode(searcher)

    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"\nError: {str(e)}")
        traceback.print_exc()

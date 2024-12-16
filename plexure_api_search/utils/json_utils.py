"""JSON utilities for cleaning and validating content."""

import json
import logging
import re
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def clean_json_string(content: str) -> str:
    """Clean and fix malformed JSON string.

    Args:
        content: Raw JSON string

    Returns:
        Cleaned and valid JSON string
    """
    try:
        # Log original content
        logger.debug(f"Original content to clean: {content}")

        # Return empty object if content is empty
        if not content or not content.strip():
            logger.error("Empty content provided")
            return "{}"

        # Remove any markdown code block markers
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        logger.debug(f"Content after markdown cleanup: {content}")

        # Try to parse as is first
        try:
            parsed = json.loads(content)
            return json.dumps(parsed, indent=2)
        except json.JSONDecodeError:
            logger.debug("Initial parse failed, attempting cleanup")

        # Basic cleanup
        content = re.sub(r"\n\s*\n", "\n", content)  # Remove empty lines
        content = re.sub(r"^\s*{\s*\n", "{", content)  # Clean start
        content = re.sub(r"\n\s*}\s*$", "}", content)  # Clean end

        # Handle newlines in strings
        content = re.sub(r'\n\s*', ' ', content)  # Replace newlines with spaces
        content = re.sub(r'\s+', ' ', content)  # Normalize whitespace

        # Fix escaped quotes in property names
        content = re.sub(r'{\s*\\"([^"]+)\\":', r'{\1":', content)  # Start of object
        content = re.sub(r',\s*\\"([^"]+)\\":', r',"\1":', content)  # Middle of object
        
        # Fix double escaped quotes
        content = content.replace('\\"', '"')  # Remove escaped quotes
        
        # Fix property names with escaped quotes
        content = re.sub(r'([{,])\s*"\\?"([^"]+)\\?":\s*', r'\1"\2":', content)
        
        # Fix string values
        content = re.sub(r':\s*"\\?"([^"]+)\\?"(?=[,}])', r':"\1"', content)
        
        # Remove any remaining problematic characters
        content = re.sub(r'[\x00-\x1F\x7F]', '', content)  # Remove control characters

        logger.debug(f"Content after cleanup: {content}")

        # Try to parse after cleanup
        try:
            parsed = json.loads(content)
            return json.dumps(parsed, indent=2)
        except json.JSONDecodeError as e:
            logger.debug(f"Parse after cleanup failed: {e}")

            # Try more aggressive fixes
            try:
                # Extract just the JSON structure
                matches = re.findall(r'{[^{}]*}', content)
                if matches:
                    for potential_json in matches:
                        try:
                            # Clean up the extracted JSON
                            cleaned = potential_json
                            # Fix property names
                            cleaned = re.sub(r'([{,])\s*"\\?"([^"]+)\\?":\s*', r'\1"\2":', cleaned)
                            # Fix string values
                            cleaned = re.sub(r':\s*"\\?"([^"]+)\\?"(?=[,}])', r':"\1"', cleaned)
                            # Remove escaped quotes
                            cleaned = cleaned.replace('\\"', '"')
                            parsed = json.loads(cleaned)
                            return json.dumps(parsed, indent=2)
                        except json.JSONDecodeError:
                            continue

                logger.error("Failed to extract valid JSON structure")
                logger.error(f"Final content that failed to parse: {content}")

            except Exception as e:
                logger.error(f"Error during aggressive cleanup: {e}")

    except Exception as e:
        logger.error(f"Error cleaning JSON string: {e}")

    # Return minimal valid JSON object
    return "{}"


def validate_json_structure(content: Dict[str, Any], required_fields: set) -> bool:
    """Validate that JSON content has required fields.

    Args:
        content: JSON content to validate
        required_fields: Set of required field names

    Returns:
        True if valid, False otherwise
    """
    try:
        return all(field in content for field in required_fields)
    except Exception as e:
        logger.error(f"Error validating JSON structure: {e}")
        return False


def get_minimal_enrichment_structure() -> Dict[str, Any]:
    """Return a minimal fallback structure for enrichment data.

    Returns:
        Minimal valid enrichment structure
    """
    return {
        "detailed_description": "Failed to parse enriched data",
        "use_cases": ["No data available"],
        "best_practices": ["No data available"],
        "related_endpoints": ["No data available"],
        "parameters_description": {
            "required": ["No data available"],
            "optional": ["No data available"],
        },
        "response_scenarios": {
            "success": "No data available",
            "errors": ["No data available"],
        },
    } 
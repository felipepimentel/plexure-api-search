"""Optimized vector operations."""

import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from numpy.linalg import norm

logger = logging.getLogger(__name__)


def cosine_similarity(
    query_vector: Union[np.ndarray, torch.Tensor],
    vectors: Union[np.ndarray, torch.Tensor],
    batch_size: int = 1000,
) -> np.ndarray:
    """Calculate cosine similarity between query vector and vectors.

    This implementation is optimized for large-scale similarity calculations
    by using batched operations and numpy's optimized linear algebra.

    Args:
        query_vector: Query vector
        vectors: Matrix of vectors to compare against
        batch_size: Size of batches for processing

    Returns:
        Array of similarity scores
    """
    # Convert to numpy if needed
    if isinstance(query_vector, torch.Tensor):
        query_vector = query_vector.detach().cpu().numpy()
    if isinstance(vectors, torch.Tensor):
        vectors = vectors.detach().cpu().numpy()

    # Ensure 2D arrays
    if query_vector.ndim == 1:
        query_vector = query_vector.reshape(1, -1)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)

    # Normalize query vector
    query_norm = norm(query_vector, axis=1, keepdims=True)
    query_vector = np.divide(
        query_vector,
        query_norm,
        out=np.zeros_like(query_vector),
        where=query_norm != 0,
    )

    # Process in batches
    num_vectors = vectors.shape[0]
    similarities = np.zeros(num_vectors)

    for i in range(0, num_vectors, batch_size):
        batch = vectors[i : i + batch_size]

        # Normalize batch
        batch_norm = norm(batch, axis=1, keepdims=True)
        batch_normalized = np.divide(
            batch,
            batch_norm,
            out=np.zeros_like(batch),
            where=batch_norm != 0,
        )

        # Calculate similarity
        batch_similarities = np.dot(query_vector, batch_normalized.T).flatten()
        similarities[i : i + batch_size] = batch_similarities

    return similarities


def euclidean_distance(
    query_vector: Union[np.ndarray, torch.Tensor],
    vectors: Union[np.ndarray, torch.Tensor],
    batch_size: int = 1000,
) -> np.ndarray:
    """Calculate Euclidean distance between query vector and vectors.

    This implementation is optimized for large-scale distance calculations
    by using batched operations and numpy's optimized linear algebra.

    Args:
        query_vector: Query vector
        vectors: Matrix of vectors to compare against
        batch_size: Size of batches for processing

    Returns:
        Array of distances
    """
    # Convert to numpy if needed
    if isinstance(query_vector, torch.Tensor):
        query_vector = query_vector.detach().cpu().numpy()
    if isinstance(vectors, torch.Tensor):
        vectors = vectors.detach().cpu().numpy()

    # Ensure 2D arrays
    if query_vector.ndim == 1:
        query_vector = query_vector.reshape(1, -1)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)

    # Process in batches
    num_vectors = vectors.shape[0]
    distances = np.zeros(num_vectors)

    for i in range(0, num_vectors, batch_size):
        batch = vectors[i : i + batch_size]

        # Calculate squared distances using the formula:
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
        query_norm = np.sum(query_vector ** 2)
        batch_norm = np.sum(batch ** 2, axis=1)
        dot_product = np.dot(query_vector, batch.T).flatten()

        batch_distances = np.sqrt(
            np.maximum(query_norm + batch_norm - 2 * dot_product, 0)
        )
        distances[i : i + batch_size] = batch_distances

    return distances


def dot_product(
    query_vector: Union[np.ndarray, torch.Tensor],
    vectors: Union[np.ndarray, torch.Tensor],
    batch_size: int = 1000,
) -> np.ndarray:
    """Calculate dot product between query vector and vectors.

    This implementation is optimized for large-scale dot product calculations
    by using batched operations and numpy's optimized linear algebra.

    Args:
        query_vector: Query vector
        vectors: Matrix of vectors to compare against
        batch_size: Size of batches for processing

    Returns:
        Array of dot products
    """
    # Convert to numpy if needed
    if isinstance(query_vector, torch.Tensor):
        query_vector = query_vector.detach().cpu().numpy()
    if isinstance(vectors, torch.Tensor):
        vectors = vectors.detach().cpu().numpy()

    # Ensure 2D arrays
    if query_vector.ndim == 1:
        query_vector = query_vector.reshape(1, -1)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)

    # Process in batches
    num_vectors = vectors.shape[0]
    products = np.zeros(num_vectors)

    for i in range(0, num_vectors, batch_size):
        batch = vectors[i : i + batch_size]
        batch_products = np.dot(query_vector, batch.T).flatten()
        products[i : i + batch_size] = batch_products

    return products


def top_k_indices(
    scores: np.ndarray,
    k: int,
    reverse: bool = True,
) -> np.ndarray:
    """Get indices of top k scores.

    This implementation is optimized for large-scale top-k selection
    by using numpy's optimized partition function.

    Args:
        scores: Array of scores
        k: Number of top scores to return
        reverse: Whether to sort in descending order

    Returns:
        Array of indices for top k scores
    """
    if k <= 0:
        return np.array([], dtype=np.int64)

    k = min(k, len(scores))
    if reverse:
        partition_idx = np.argpartition(-scores, k)[:k]
        top_k_idx = partition_idx[np.argsort(-scores[partition_idx])]
    else:
        partition_idx = np.argpartition(scores, k)[:k]
        top_k_idx = partition_idx[np.argsort(scores[partition_idx])]

    return top_k_idx


def normalize_vectors(
    vectors: Union[np.ndarray, torch.Tensor],
    batch_size: int = 1000,
) -> np.ndarray:
    """Normalize vectors to unit length.

    This implementation is optimized for large-scale normalization
    by using batched operations.

    Args:
        vectors: Matrix of vectors to normalize
        batch_size: Size of batches for processing

    Returns:
        Normalized vectors
    """
    # Convert to numpy if needed
    if isinstance(vectors, torch.Tensor):
        vectors = vectors.detach().cpu().numpy()

    # Ensure 2D array
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)

    # Process in batches
    num_vectors = vectors.shape[0]
    normalized = np.zeros_like(vectors)

    for i in range(0, num_vectors, batch_size):
        batch = vectors[i : i + batch_size]
        batch_norm = norm(batch, axis=1, keepdims=True)
        batch_normalized = np.divide(
            batch,
            batch_norm,
            out=np.zeros_like(batch),
            where=batch_norm != 0,
        )
        normalized[i : i + batch_size] = batch_normalized

    return normalized


def vector_mean(
    vectors: Union[np.ndarray, torch.Tensor],
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Calculate weighted mean of vectors.

    This implementation supports weighted averaging of vectors.

    Args:
        vectors: Matrix of vectors
        weights: Optional weights for each vector

    Returns:
        Mean vector
    """
    # Convert to numpy if needed
    if isinstance(vectors, torch.Tensor):
        vectors = vectors.detach().cpu().numpy()

    # Ensure 2D array
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)

    if weights is None:
        return np.mean(vectors, axis=0)

    # Normalize weights
    weights = np.array(weights, dtype=np.float32)
    weights = weights / np.sum(weights)

    # Calculate weighted mean
    return np.sum(vectors * weights[:, np.newaxis], axis=0)


def vector_quantize(
    vectors: Union[np.ndarray, torch.Tensor],
    num_bits: int = 8,
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Quantize vectors to reduce memory usage.

    This implementation quantizes vectors to specified number of bits
    while preserving the relative distances between vectors.

    Args:
        vectors: Matrix of vectors to quantize
        num_bits: Number of bits for quantization

    Returns:
        Tuple of quantized vectors and quantization parameters
    """
    # Convert to numpy if needed
    if isinstance(vectors, torch.Tensor):
        vectors = vectors.detach().cpu().numpy()

    # Ensure 2D array
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)

    # Calculate quantization parameters
    min_val = np.min(vectors)
    max_val = np.max(vectors)
    scale = (max_val - min_val) / (2 ** num_bits - 1)

    # Quantize vectors
    quantized = np.clip(
        np.round((vectors - min_val) / scale),
        0,
        2 ** num_bits - 1,
    ).astype(np.uint8)

    return quantized, (min_val, scale)


def vector_dequantize(
    quantized: np.ndarray,
    params: Tuple[float, float],
) -> np.ndarray:
    """Dequantize vectors.

    This implementation dequantizes vectors that were quantized using
    vector_quantize.

    Args:
        quantized: Matrix of quantized vectors
        params: Quantization parameters (min_val, scale)

    Returns:
        Dequantized vectors
    """
    min_val, scale = params
    return quantized.astype(np.float32) * scale + min_val


__all__ = [
    "cosine_similarity",
    "euclidean_distance",
    "dot_product",
    "top_k_indices",
    "normalize_vectors",
    "vector_mean",
    "vector_quantize",
    "vector_dequantize",
] 
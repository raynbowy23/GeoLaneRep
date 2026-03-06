"""Equivariant coordinate transforms for tangent-aligned local frames."""

import numpy as np
import torch


def canonicalize_polyline(
    points: np.ndarray,
    centroid: np.ndarray,
    tangent: np.ndarray,
) -> np.ndarray:
    """Transform points into tangent-aligned local frame.

    Args:
        points: (K, 2) trajectory segment in global pixel coords.
        centroid: (2,) center of the segment.
        tangent: (2,) unit tangent direction vector.

    Returns:
        (K, 2) points in local frame (origin=centroid, x-axis=tangent).
    """
    theta = np.arctan2(tangent[1], tangent[0])
    cos_t, sin_t = np.cos(-theta), np.sin(-theta)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
    centered = points - centroid
    return (R @ centered.T).T


def decanonicalize_polyline(
    local_points: np.ndarray,
    centroid: np.ndarray,
    tangent: np.ndarray,
) -> np.ndarray:
    """Inverse transform: local frame back to global pixel coords.

    Args:
        local_points: (K, 2) in tangent-aligned local frame.
        centroid: (2,) center in global coords.
        tangent: (2,) unit tangent direction vector.

    Returns:
        (K, 2) points in global pixel coords.
    """
    theta = np.arctan2(tangent[1], tangent[0])
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
    return (R @ local_points.T).T + centroid


def canonicalize_polyline_batch(
    points: torch.Tensor,
    centroids: torch.Tensor,
    tangents: torch.Tensor,
) -> torch.Tensor:
    """Batch canonicalization for PyTorch tensors.

    Args:
        points: (N, K, 2) global coordinates.
        centroids: (N, 2) segment centers.
        tangents: (N, 2) unit tangent vectors.

    Returns:
        (N, K, 2) in tangent-aligned local frames.
    """
    theta = torch.atan2(tangents[:, 1], tangents[:, 0]) # (N,)
    cos_t = torch.cos(-theta) # (N,)
    sin_t = torch.sin(-theta)
    # Rotation matrices (N, 2, 2)
    R = torch.stack([
        torch.stack([cos_t, -sin_t], dim=-1),
        torch.stack([sin_t, cos_t], dim=-1),
    ], dim=-2)
    centered = points - centroids.unsqueeze(1) # (N, K, 2)
    return torch.einsum("nij,nkj->nki", R, centered)


def decanonicalize_polyline_batch(
    local_points: torch.Tensor,
    centroids: torch.Tensor,
    tangents: torch.Tensor,
) -> torch.Tensor:
    """Batch inverse transform for PyTorch tensors.

    Args:
        local_points: (N, K, 2) in local frames.
        centroids: (N, 2) global centers.
        tangents: (N, 2) unit tangent vectors.

    Returns:
        (N, K, 2) in global pixel coordinates.
    """
    theta = torch.atan2(tangents[:, 1], tangents[:, 0])
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    R = torch.stack([
        torch.stack([cos_t, -sin_t], dim=-1),
        torch.stack([sin_t, cos_t], dim=-1),
    ], dim=-2)
    return torch.einsum("nij,nkj->nki", R, local_points) + centroids.unsqueeze(1)

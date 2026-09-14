"""Weighted connectome graph harmonics."""

import numpy as np


def _adjacency(adjacency, symmetrize=False, tolerance=1e-12):
    matrix = np.asarray(adjacency, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError("adjacency must be a non-empty square matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("adjacency must contain only finite values")
    if np.any(matrix < -tolerance):
        raise ValueError("connectome weights must be nonnegative")
    matrix = matrix.copy()
    matrix[np.abs(matrix) <= tolerance] = 0.0
    np.fill_diagonal(matrix, 0.0)
    if not np.allclose(matrix, matrix.T, atol=tolerance, rtol=0.0):
        if not symmetrize:
            raise ValueError("adjacency must be symmetric; use symmetrize=True for directed input")
        matrix = (matrix + matrix.T) / 2.0
    return matrix


def build_laplacian(adjacency, normalized=True, symmetrize=False):
    """Return (L, degree, cleaned_adjacency).

    normalized=True uses D^(-1/2)(D-A)D^(-1/2). Isolated nodes remain zero.
    """
    matrix = _adjacency(adjacency, symmetrize=symmetrize)
    degree = matrix.sum(axis=1)
    combinatorial = np.diag(degree) - matrix
    if not normalized:
        return combinatorial, degree, matrix
    inv = np.zeros_like(degree)
    nonzero = degree > 0
    inv[nonzero] = 1.0 / np.sqrt(degree[nonzero])
    scale = np.diag(inv)
    return scale @ combinatorial @ scale, degree, matrix


def eigendecompose_laplacian(adjacency, normalized=True, n_modes=None,
    zero_tolerance=1e-10, symmetrize=False):
    """Compute eigenmodes plus residual and orthogonality diagnostics."""
    laplacian, degree, matrix = build_laplacian(
        adjacency, normalized=normalized, symmetrize=symmetrize
    )
    values, vectors = np.linalg.eigh(laplacian)
    order = np.argsort(values)
    values, vectors = values[order], vectors[:, order]
    if n_modes is not None:
        if not isinstance(n_modes, (int, np.integer)) or n_modes < 1:
            raise ValueError("n_modes must be a positive integer or None")
        values, vectors = values[:n_modes], vectors[:, :n_modes]
    residuals = np.linalg.norm(
        laplacian @ vectors - vectors * values[np.newaxis, :], axis=0
    )
    zero = values <= zero_tolerance
    positive = values[~zero]
    return {
        "adjacency": matrix,
        "degrees": degree,
        "laplacian": laplacian,
        "normalized": normalized,
        "eigenvalues": values,
        "eigenvectors": vectors,
        "residuals": residuals,
        "orthogonality_error": float(np.linalg.norm(
            vectors.T @ vectors - np.eye(vectors.shape[1])
        )),
        "zero_mode_count": int(np.count_nonzero(zero)),
        "algebraic_connectivity": float(positive[0]) if positive.size else None,
    }


def project_signal(signal, eigenvectors, n_modes=None):
    """Project (nodes,) or (samples, nodes) into graph-harmonic coordinates."""
    values = np.asarray(signal, dtype=float)
    vectors = np.asarray(eigenvectors, dtype=float)
    if values.ndim not in (1, 2) or vectors.ndim != 2:
        raise ValueError("signal must be 1D/2D and eigenvectors must be 2D")
    if values.shape[-1] != vectors.shape[0]:
        raise ValueError("signal node count must match eigenvector rows")
    if not np.all(np.isfinite(values)):
        raise ValueError("signal must contain only finite values")
    if n_modes is not None:
        vectors = vectors[:, :n_modes]
    coefficients = values @ vectors
    reconstruction = coefficients @ vectors.T
    return {
        "coefficients": coefficients,
        "reconstruction": reconstruction,
        "spectral_power": np.mean(coefficients ** 2, axis=0),
        "reconstruction_error": float(
            np.linalg.norm(values - reconstruction) / max(np.linalg.norm(values), 1e-15)
        ),
    }


def analyze_connectome_harmonics(adjacency, signal=None, normalized=True,
    n_modes=None, symmetrize=False):
    """Eigendecompose a connectome and optionally project node signals."""
    result = eigendecompose_laplacian(
        adjacency, normalized=normalized, n_modes=n_modes, symmetrize=symmetrize
    )
    if signal is not None:
        result["signal_projection"] = project_signal(
            signal, result["eigenvectors"], n_modes=n_modes
        )
    return result

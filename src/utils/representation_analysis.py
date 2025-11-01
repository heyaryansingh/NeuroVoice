"""
Representation analysis utilities for model interpretability.

Implements t-SNE, CKA similarity, and latent space visualization.
"""

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def compute_tsne_embeddings(
    embeddings: np.ndarray,
    n_components: int = 2,
    perplexity: float = 30.0,
    random_state: int = 42,
    n_iter: int = 1000,
    init: str = 'pca',
) -> np.ndarray:
    """
    Compute t-SNE embeddings for visualization.
    
    Args:
        embeddings: Input embeddings array (n_samples, n_features)
        n_components: Number of dimensions for t-SNE output
        perplexity: Perplexity parameter (typically 5-50)
        random_state: Random seed for reproducibility
        n_iter: Number of iterations
        init: Initialization method ('pca' or 'random')
    
    Returns:
        t-SNE transformed embeddings (n_samples, n_components)
    """
    # Reduce dimensionality with PCA first if too high-dimensional
    if embeddings.shape[1] > 50:
        pca = PCA(n_components=50, random_state=random_state)
        embeddings = pca.fit_transform(embeddings)
    
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=random_state,
        n_iter=n_iter,
        init=init,
        verbose=0,
    )
    
    embeddings_tsne = tsne.fit_transform(embeddings)
    return embeddings_tsne


def compute_cka_similarity(
    X: np.ndarray,
    Y: np.ndarray,
    kernel: str = 'linear',
) -> float:
    """
    Compute Centered Kernel Alignment (CKA) similarity between two representations.
    
    CKA measures similarity between neural network representations.
    Reference: "Similarity of Neural Network Representations Revisited" (Kornblith et al., 2019)
    
    Args:
        X: First representation matrix (n_samples, n_features_X)
        Y: Second representation matrix (n_samples, n_features_Y)
        kernel: Kernel type ('linear' or 'rbf')
    
    Returns:
        CKA similarity score (0-1, where 1 is identical)
    """
    # Center the matrices
    X_centered = X - X.mean(axis=0, keepdims=True)
    Y_centered = Y - Y.mean(axis=0, keepdims=True)
    
    if kernel == 'linear':
        # Linear kernel: K = XX^T
        K_X = X_centered @ X_centered.T
        K_Y = Y_centered @ Y_centered.T
    elif kernel == 'rbf':
        # RBF kernel approximation
        from sklearn.metrics.pairwise import rbf_kernel
        K_X = rbf_kernel(X_centered, gamma=1.0 / X.shape[1])
        K_Y = rbf_kernel(Y_centered, gamma=1.0 / Y.shape[1])
    else:
        raise ValueError(f"Unknown kernel: {kernel}")
    
    # Compute CKA
    hsic = np.trace(K_X @ K_Y)
    hsic_XX = np.trace(K_X @ K_X)
    hsic_YY = np.trace(K_Y @ K_Y)
    
    if hsic_XX == 0 or hsic_YY == 0:
        return 0.0
    
    cka = hsic / np.sqrt(hsic_XX * hsic_YY)
    return float(cka)


def visualize_latent_space(
    embeddings: np.ndarray,
    labels: Optional[np.ndarray] = None,
    save_path: Optional[Path] = None,
    title: str = "Latent Space Visualization",
    method: str = 'tsne',
    n_components: int = 2,
    figsize: Tuple[int, int] = (10, 8),
) -> np.ndarray:
    """
    Visualize latent space embeddings using t-SNE or PCA.
    
    Args:
        embeddings: Embeddings to visualize (n_samples, n_features)
        labels: Optional labels for coloring (n_samples,)
        save_path: Path to save figure (optional)
        title: Plot title
        method: Visualization method ('tsne' or 'pca')
        n_components: Number of dimensions (2 or 3)
        figsize: Figure size
    
    Returns:
        Transformed embeddings
    """
    if method == 'tsne':
        transformed = compute_tsne_embeddings(
            embeddings,
            n_components=n_components,
        )
    elif method == 'pca':
        pca = PCA(n_components=n_components, random_state=42)
        transformed = pca.fit_transform(embeddings)
        title += f" (PCA: {pca.explained_variance_ratio_[:n_components].sum()*100:.1f}% variance)"
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Create visualization
    if n_components == 2:
        fig, ax = plt.subplots(figsize=figsize)
        
        if labels is not None:
            unique_labels = np.unique(labels)
            for label in unique_labels:
                mask = labels == label
                ax.scatter(
                    transformed[mask, 0],
                    transformed[mask, 1],
                    label=f'Class {label}',
                    alpha=0.6,
                    s=20,
                )
            ax.legend()
        else:
            ax.scatter(transformed[:, 0], transformed[:, 1], alpha=0.6, s=20)
        
        ax.set_xlabel(f'{method.upper()} Component 1')
        ax.set_ylabel(f'{method.upper()} Component 2')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
    
    elif n_components == 3:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        if labels is not None:
            unique_labels = np.unique(labels)
            for label in unique_labels:
                mask = labels == label
                ax.scatter(
                    transformed[mask, 0],
                    transformed[mask, 1],
                    transformed[mask, 2],
                    label=f'Class {label}',
                    alpha=0.6,
                    s=20,
                )
            ax.legend()
        else:
            ax.scatter(
                transformed[:, 0],
                transformed[:, 1],
                transformed[:, 2],
                alpha=0.6,
                s=20,
            )
        
        ax.set_xlabel(f'{method.upper()} Component 1')
        ax.set_ylabel(f'{method.upper()} Component 2')
        ax.set_zlabel(f'{method.upper()} Component 3')
        ax.set_title(title)
        plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return transformed


def compute_representation_distance(
    embeddings1: np.ndarray,
    embeddings2: np.ndarray,
    metric: str = 'cosine',
) -> float:
    """
    Compute distance between two representations.
    
    Args:
        embeddings1: First representation (n_samples, n_features)
        embeddings2: Second representation (n_samples, n_features)
        metric: Distance metric ('cosine', 'euclidean', 'manhattan')
    
    Returns:
        Average distance between representations
    """
    from sklearn.metrics.pairwise import cosine_distances, euclidean_distances, manhattan_distances
    
    if metric == 'cosine':
        distances = cosine_distances(embeddings1, embeddings2)
    elif metric == 'euclidean':
        distances = euclidean_distances(embeddings1, embeddings2)
    elif metric == 'manhattan':
        distances = manhattan_distances(embeddings1, embeddings2)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    # Return mean distance (excluding diagonal for same embeddings)
    if embeddings1 is embeddings2:
        # Remove diagonal
        np.fill_diagonal(distances, np.nan)
        return float(np.nanmean(distances))
    else:
        return float(distances.mean())


def compare_representations(
    embeddings_list: list,
    labels_list: Optional[list] = None,
    names: Optional[list] = None,
    save_path: Optional[Path] = None,
) -> np.ndarray:
    """
    Compare multiple representations using CKA similarity matrix.
    
    Args:
        embeddings_list: List of embedding arrays
        labels_list: Optional list of labels for each embedding set
        names: Optional names for each representation
        save_path: Path to save similarity matrix heatmap
    
    Returns:
        CKA similarity matrix
    """
    n_reps = len(embeddings_list)
    similarity_matrix = np.zeros((n_reps, n_reps))
    
    for i in range(n_reps):
        for j in range(n_reps):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                similarity_matrix[i, j] = compute_cka_similarity(
                    embeddings_list[i],
                    embeddings_list[j],
                )
    
    # Visualize similarity matrix
    import seaborn as sns
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        similarity_matrix,
        annot=True,
        fmt='.3f',
        cmap='viridis',
        xticklabels=names if names else [f'Rep {i}' for i in range(n_reps)],
        yticklabels=names if names else [f'Rep {i}' for i in range(n_reps)],
    )
    plt.title('Representation Similarity (CKA)')
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return similarity_matrix


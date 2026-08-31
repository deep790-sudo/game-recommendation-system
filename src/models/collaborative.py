"""
Collaborative Filtering Engine using Truncated SVD (Matrix Factorization).

Decomposes the sparse user×game interaction matrix into latent factor
matrices, enabling prediction of user preferences for unseen games.
"""

import logging
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)


class CollaborativeFilteringEngine:
    """
    Collaborative filtering via Truncated SVD on the user-item interaction matrix.

    The interaction matrix is factorized as:
        R ≈ U × Σ × V^T

    Where:
        - U: user latent factors (n_users × n_components)
        - Σ: singular values (diagonal)
        - V^T: item latent factors (n_components × n_games)

    Predicted rating for user u and game g:
        r̂(u, g) = user_factors[u] · game_factors[g]

    Attributes:
        n_components: Number of latent factors.
        user_factors: User latent factor matrix.
        game_factors: Game latent factor matrix.
        user_to_idx: Mapping from user_id to matrix row index.
        game_to_idx: Mapping from game_id (app_id) to matrix column index.
    """

    def __init__(self, n_components: int = 100, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state

        self.svd: TruncatedSVD | None = None
        self.user_factors: np.ndarray | None = None
        self.game_factors: np.ndarray | None = None
        self.user_to_idx: dict[int, int] | None = None
        self.game_to_idx: dict[int, int] | None = None
        self.idx_to_game: dict[int, int] | None = None
        self.global_mean: float = 0.0
        self._interaction_matrix: sparse.csr_matrix | None = None

    def fit(
        self,
        interaction_matrix: sparse.csr_matrix,
        user_to_idx: dict[int, int],
        game_to_idx: dict[int, int],
    ) -> "CollaborativeFilteringEngine":
        """
        Fit the SVD model on the user-item interaction matrix.

        Args:
            interaction_matrix: Sparse CSR matrix (n_users × n_games).
            user_to_idx: user_id → row index mapping.
            game_to_idx: app_id → column index mapping.
        """
        self.user_to_idx = user_to_idx
        self.game_to_idx = game_to_idx
        self.idx_to_game = {idx: gid for gid, idx in game_to_idx.items()}
        self._interaction_matrix = interaction_matrix

        # Compute global mean for centering
        self.global_mean = interaction_matrix.data.mean() if interaction_matrix.nnz > 0 else 0.0

        n_users, n_games = interaction_matrix.shape
        actual_components = min(self.n_components, n_users - 1, n_games - 1)

        logger.info(
            f"Fitting SVD with {actual_components} components on "
            f"{n_users:,} × {n_games:,} matrix ({interaction_matrix.nnz:,} non-zero)..."
        )

        self.svd = TruncatedSVD(
            n_components=actual_components,
            random_state=self.random_state,
            algorithm="randomized",
            n_iter=10,
        )

        # U × Σ (user factors with singular values absorbed)
        self.user_factors = self.svd.fit_transform(interaction_matrix)

        # V^T (game factors) — shape: (n_components, n_games)
        # We want game_factors as (n_games, n_components) for dot product
        self.game_factors = self.svd.components_.T

        explained_var = self.svd.explained_variance_ratio_.sum()
        logger.info(
            f"SVD fit complete. Explained variance: {explained_var:.2%} "
            f"with {actual_components} components."
        )

        return self

    def predict_score(self, user_id: int, game_id: int) -> float:
        """Predict the engagement score for a specific user-game pair."""
        if user_id not in self.user_to_idx or game_id not in self.game_to_idx:
            return 0.0

        u_idx = self.user_to_idx[user_id]
        g_idx = self.game_to_idx[game_id]

        score = float(np.dot(self.user_factors[u_idx], self.game_factors[g_idx]))
        return max(score, 0.0)  # Clip negative predictions

    def recommend_for_user(
        self,
        user_id: int,
        n: int = 10,
        exclude_ids: set[int] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Recommend top-N games for a user based on predicted engagement scores.

        Args:
            user_id: The user to generate recommendations for.
            n: Number of recommendations.
            exclude_ids: Set of app_ids to exclude (e.g., already played).

        Returns:
            List of dicts with 'app_id' and 'cf_score'.
        """
        if user_id not in self.user_to_idx:
            logger.warning(f"User {user_id} not in collaborative filtering model.")
            return []

        u_idx = self.user_to_idx[user_id]

        # Predict scores for all games at once (vectorized)
        all_scores = np.dot(self.user_factors[u_idx], self.game_factors.T)

        # Mask out already-interacted games
        if self._interaction_matrix is not None:
            interacted = set(self._interaction_matrix[u_idx].nonzero()[1])
        else:
            interacted = set()

        if exclude_ids:
            exclude_indices = {
                self.game_to_idx[gid] for gid in exclude_ids
                if gid in self.game_to_idx
            }
            interacted.update(exclude_indices)

        # Zero out excluded games
        for idx in interacted:
            all_scores[idx] = -np.inf

        # Get top-N indices
        top_indices = np.argsort(all_scores)[::-1][:n]

        results = []
        for idx in top_indices:
            if all_scores[idx] == -np.inf:
                break
            game_id = self.idx_to_game[idx]
            results.append({
                "app_id": int(game_id),
                "cf_score": float(max(all_scores[idx], 0.0)),
            })

        return results

    def get_similar_users(self, user_id: int, n: int = 10) -> list[tuple[int, float]]:
        """Find users with similar taste profiles (nearest neighbors in latent space)."""
        if user_id not in self.user_to_idx:
            return []

        u_idx = self.user_to_idx[user_id]
        user_vec = self.user_factors[u_idx].reshape(1, -1)

        # Normalize for cosine similarity
        all_users_norm = normalize(self.user_factors, axis=1)
        user_vec_norm = normalize(user_vec, axis=1)

        similarities = np.dot(all_users_norm, user_vec_norm.T).flatten()
        similarities[u_idx] = -1  # Exclude self

        idx_to_user = {idx: uid for uid, idx in self.user_to_idx.items()}
        top_indices = np.argsort(similarities)[::-1][:n]

        return [
            (idx_to_user[idx], float(similarities[idx]))
            for idx in top_indices
        ]

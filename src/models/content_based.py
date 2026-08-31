"""
Content-Based Recommendation Engine.

Uses TF-IDF vectorization on game tags/genres/metadata and cosine similarity
to find games similar to a user's play history. This engine handles the
cold-start problem: it only needs game metadata (no interaction history).
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class ContentBasedEngine:
    """
    Content-based recommendation engine using TF-IDF + cosine similarity.

    Attributes:
        tfidf_matrix: TF-IDF feature matrix for all games.
        vectorizer: Fitted TfidfVectorizer instance.
        game_ids: Array of app_ids aligned with tfidf_matrix rows.
        game_idx_map: Mapping from app_id to matrix row index.
    """

    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 5,
        max_df: float = 0.95,
    ):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df

        self.vectorizer: TfidfVectorizer | None = None
        self.tfidf_matrix: np.ndarray | None = None
        self.game_ids: np.ndarray | None = None
        self.game_idx_map: dict[int, int] | None = None

    def fit(self, games: pd.DataFrame) -> "ContentBasedEngine":
        """
        Fit the TF-IDF vectorizer on game content features.

        Args:
            games: DataFrame with 'app_id' and 'content_features' columns.
        """
        if "content_features" not in games.columns:
            raise ValueError("Games DataFrame must have 'content_features' column. "
                             "Run build_game_content_features() first.")

        # Filter games with empty content features
        valid_mask = games["content_features"].str.len() > 0
        valid_games = games[valid_mask].copy()

        if len(valid_games) == 0:
            raise ValueError("No games with content features found.")

        logger.info(f"Fitting TF-IDF on {len(valid_games):,} games...")

        self.game_ids = valid_games["app_id"].values
        self.game_idx_map = {gid: idx for idx, gid in enumerate(self.game_ids)}

        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            token_pattern=r"(?u)\b[\w+]+\b",
        )

        self.tfidf_matrix = self.vectorizer.fit_transform(
            valid_games["content_features"]
        )

        logger.info(
            f"TF-IDF matrix: {self.tfidf_matrix.shape[0]:,} games × "
            f"{self.tfidf_matrix.shape[1]:,} features"
        )

        return self

    def get_similar_games(
        self,
        game_id: int,
        n: int = 10,
        exclude_ids: set[int] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Find the N most similar games to a given game.

        Args:
            game_id: The app_id of the query game.
            n: Number of recommendations to return.
            exclude_ids: Set of app_ids to exclude (e.g., already played).

        Returns:
            List of dicts with 'app_id' and 'similarity_score'.
        """
        if game_id not in self.game_idx_map:
            logger.warning(f"Game {game_id} not in content index.")
            return []

        idx = self.game_idx_map[game_id]
        game_vector = self.tfidf_matrix[idx]

        # Compute similarity to all games
        similarities = cosine_similarity(game_vector, self.tfidf_matrix).flatten()

        # Exclude self and already-played games
        exclude_ids = exclude_ids or set()
        exclude_ids.add(game_id)

        # Get top-N
        results = []
        sorted_indices = np.argsort(similarities)[::-1]

        for candidate_idx in sorted_indices:
            candidate_id = self.game_ids[candidate_idx]
            if candidate_id in exclude_ids:
                continue
            results.append({
                "app_id": int(candidate_id),
                "similarity_score": float(similarities[candidate_idx]),
            })
            if len(results) >= n:
                break

        return results

    def recommend_for_user(
        self,
        user_game_ids: list[int],
        n: int = 10,
        exclude_ids: set[int] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Recommend games based on a user's play history using content similarity.

        Creates a user profile by averaging the TF-IDF vectors of their
        played games, then finds the most similar unseen games.

        Args:
            user_game_ids: List of app_ids the user has played.
            n: Number of recommendations.
            exclude_ids: Additional app_ids to exclude.

        Returns:
            List of dicts with 'app_id' and 'content_score'.
        """
        # Build user profile from played games
        valid_indices = [
            self.game_idx_map[gid]
            for gid in user_game_ids
            if gid in self.game_idx_map
        ]

        if not valid_indices:
            logger.warning("No valid games in user history for content-based recs.")
            return []

        # Average TF-IDF vectors of played games → user taste profile
        user_profile = self.tfidf_matrix[valid_indices].mean(axis=0)

        # Compute similarity to all games
        similarities = cosine_similarity(
            np.asarray(user_profile), self.tfidf_matrix
        ).flatten()

        # Exclude played games
        exclude_set = set(user_game_ids)
        if exclude_ids:
            exclude_set.update(exclude_ids)

        # Get top-N
        results = []
        sorted_indices = np.argsort(similarities)[::-1]

        for candidate_idx in sorted_indices:
            candidate_id = self.game_ids[candidate_idx]
            if candidate_id in exclude_set:
                continue
            results.append({
                "app_id": int(candidate_id),
                "content_score": float(similarities[candidate_idx]),
            })
            if len(results) >= n:
                break

        return results

    def get_top_features(self, game_id: int, n: int = 10) -> list[str]:
        """Get the top TF-IDF features (tags) for a game — useful for explanations."""
        if game_id not in self.game_idx_map:
            return []

        idx = self.game_idx_map[game_id]
        feature_names = self.vectorizer.get_feature_names_out()
        scores = self.tfidf_matrix[idx].toarray().flatten()
        top_indices = np.argsort(scores)[::-1][:n]

        return [feature_names[i] for i in top_indices if scores[i] > 0]

    def score_items(
        self,
        user_game_ids: list[int],
        target_game_ids: list[int],
    ) -> dict[int, float]:
        """
        Score specific target games against a user's taste profile.

        Unlike recommend_for_user which generates candidates, this method
        scores a pre-defined set of games. Used by the hybrid blender to
        compute CB scores for CF-generated candidates.
        """
        valid_indices = [
            self.game_idx_map[gid]
            for gid in user_game_ids
            if gid in self.game_idx_map
        ]

        if not valid_indices:
            return {gid: 0.0 for gid in target_game_ids}

        user_profile = self.tfidf_matrix[valid_indices].mean(axis=0)

        target_indices = []
        target_ids_valid = []
        for gid in target_game_ids:
            if gid in self.game_idx_map:
                target_indices.append(self.game_idx_map[gid])
                target_ids_valid.append(gid)

        if not target_indices:
            return {gid: 0.0 for gid in target_game_ids}

        target_matrix = self.tfidf_matrix[target_indices]
        similarities = cosine_similarity(
            np.asarray(user_profile), target_matrix
        ).flatten()

        scores = {gid: 0.0 for gid in target_game_ids}
        for gid, sim in zip(target_ids_valid, similarities):
            scores[gid] = float(sim)

        return scores

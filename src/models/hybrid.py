"""
Hybrid Recommendation Engine.

Combines collaborative filtering, content-based similarity, and Wilson
quality ranking into a single recommendation pipeline with automatic
cold-start detection and adaptive weighting.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.models.collaborative import CollaborativeFilteringEngine
from src.models.content_based import ContentBasedEngine
from src.models.wilson_score import compute_game_quality_scores

logger = logging.getLogger(__name__)


# Cold-start thresholds for adaptive weighting
COLD_THRESHOLD = 3    # < 3 interactions → fully content-based
WARM_THRESHOLD = 10   # 3-10 → blended, > 10 → mostly collaborative

# Patterns in titles that indicate non-game content (DLC, tools, demos, etc.)
_JUNK_TITLE_PATTERNS = [
    " dlc", " - dlc", "soundtrack", " ost", "artbook", "art book",
    "wallpaper", "skin pack", "costume pack", "demo", "playtest",
    "beta", "dedicated server", "sdk", "editor", "mod tool",
    "season pass", "bonus content", "digital deluxe", "upgrade pack",
]

# Minimum review count for a recommendation to be surfaced
MIN_REVIEWS_FOR_REC = 10


class HybridRecommender:
    """
    Hybrid recommendation engine with cold-start handling.

    Blending strategy:
        - Cold users (< 3 interactions): 100% content-based
        - Warm users (3-10): 60% CF + 40% CB
        - Active users (> 10): 80% CF + 20% CB

    After blending, scores are re-ranked by Wilson quality score to
    penalize low-quality popular games and surface hidden gems.

    Attributes:
        cf_engine: Fitted CollaborativeFilteringEngine.
        cb_engine: Fitted ContentBasedEngine.
        games: Games DataFrame with Wilson scores.
        wilson_weight: How much Wilson score influences final ranking (0-1).
    """

    def __init__(self, wilson_weight: float = 0.05):
        """
        Args:
            wilson_weight: Weight of Wilson quality adjustment in final score.
                          0.0 = no quality adjustment, 1.0 = heavy quality bias.
                          Default 0.05 gives a subtle nudge toward quality.
        """
        self.wilson_weight = wilson_weight
        self.cf_engine: CollaborativeFilteringEngine | None = None
        self.cb_engine: ContentBasedEngine | None = None
        self.games: pd.DataFrame | None = None
        self.train_interactions: pd.DataFrame | None = None
        self._wilson_lookup: dict[int, float] = {}
        self._game_title_lookup: dict[int, str] = {}

    def fit(
        self,
        cf_engine: CollaborativeFilteringEngine,
        cb_engine: ContentBasedEngine,
        games: pd.DataFrame,
        train_interactions: pd.DataFrame,
    ) -> "HybridRecommender":
        """
        Initialize the hybrid recommender with pre-fitted component engines.

        Args:
            cf_engine: A fitted CollaborativeFilteringEngine.
            cb_engine: A fitted ContentBasedEngine.
            games: Games DataFrame (Wilson scores will be added if missing).
            train_interactions: Training interaction data (for user history lookup).
        """
        self.cf_engine = cf_engine
        self.cb_engine = cb_engine
        self.train_interactions = train_interactions

        # Compute Wilson scores if not already present
        if "wilson_score" not in games.columns:
            games = compute_game_quality_scores(games)
        self.games = games

        # Build lookup dicts for fast access
        self._wilson_lookup = dict(
            zip(games["app_id"], games["wilson_score"])
        )
        self._game_title_lookup = dict(
            zip(games["app_id"], games["title"])
        )

        # Build set of recommendable games (exclude DLC, soundtracks, demos, etc.)
        self._recommendable_ids = set()
        for _, row in games.iterrows():
            title_lower = str(row.get("title", "")).lower()
            reviews = row.get("user_reviews", 0)

            # Skip junk titles
            if any(pat in title_lower for pat in _JUNK_TITLE_PATTERNS):
                continue
            # Skip games with too few reviews
            if reviews < MIN_REVIEWS_FOR_REC:
                continue

            self._recommendable_ids.add(row["app_id"])

        n_filtered = len(games) - len(self._recommendable_ids)
        logger.info(
            f"Hybrid recommender initialized. "
            f"{len(self._recommendable_ids):,} recommendable games "
            f"({n_filtered:,} filtered as DLC/junk/low-review)."
        )
        return self

    def _get_user_history(self, user_id: int) -> list[int]:
        """Get list of game IDs a user has interacted with."""
        if self.train_interactions is None:
            return []
        user_games = self.train_interactions[
            self.train_interactions["user_id"] == user_id
        ]["app_id"].tolist()
        return user_games

    def _get_blend_weights(self, n_interactions: int) -> tuple[float, float]:
        """
        Determine CF/CB blend weights based on user interaction count.

        Returns:
            Tuple of (cf_weight, cb_weight) that sum to 1.0.
        """
        if n_interactions < COLD_THRESHOLD:
            return 0.0, 1.0   # Pure content-based for cold users
        elif n_interactions < WARM_THRESHOLD:
            # Linear interpolation: CF goes from 50% → 80%
            progress = (n_interactions - COLD_THRESHOLD) / (WARM_THRESHOLD - COLD_THRESHOLD)
            cf_weight = 0.5 + 0.3 * progress
            cb_weight = 1.0 - cf_weight
            return cf_weight, cb_weight
        else:
            return 0.85, 0.15  # Mostly collaborative for active users

    @staticmethod
    def _normalize_scores(scores: dict[int, float]) -> dict[int, float]:
        """Min-max normalize scores to [0, 1]."""
        if not scores:
            return scores
        values = np.array(list(scores.values()))
        min_val, max_val = values.min(), values.max()
        if max_val - min_val < 1e-10:
            return {k: 0.5 for k in scores}
        return {
            k: (v - min_val) / (max_val - min_val)
            for k, v in scores.items()
        }

    def recommend(
        self,
        user_id: int,
        n: int = 10,
        explain: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Generate hybrid recommendations for a user.

        Args:
            user_id: The user to recommend for.
            n: Number of recommendations.
            explain: If True, include explanation of score breakdown.

        Returns:
            List of recommendation dicts with keys:
                - app_id, title, final_score
                - cf_score, cb_score, wilson_score (if explain=True)
                - source: 'cf', 'cb', or 'hybrid'
                - reason: human-readable explanation
        """
        user_history = self._get_user_history(user_id)
        n_interactions = len(user_history)
        exclude_set = set(user_history)

        cf_weight, cb_weight = self._get_blend_weights(n_interactions)

        # Fetch more candidates than needed for re-ranking
        candidate_pool_size = n * 5

        # --- Collaborative Filtering Scores ---
        cf_scores: dict[int, float] = {}
        if cf_weight > 0 and self.cf_engine is not None:
            cf_recs = self.cf_engine.recommend_for_user(
                user_id, n=candidate_pool_size, exclude_ids=exclude_set
            )
            cf_scores = {r["app_id"]: r["cf_score"] for r in cf_recs}

        # --- Content-Based Scores ---
        cb_scores: dict[int, float] = {}
        if cb_weight > 0 and self.cb_engine is not None and user_history:
            cb_recs = self.cb_engine.recommend_for_user(
                user_history, n=candidate_pool_size, exclude_ids=exclude_set
            )
            cb_scores = {r["app_id"]: r["content_score"] for r in cb_recs}

        # --- Cross-score: compute missing scores for all candidates ---
        # Without this, candidates from one engine get 0 from the other,
        # which causes the hybrid to underperform pure CF/CB.
        all_candidates = set(cf_scores.keys()) | set(cb_scores.keys())

        # Score CF candidates with CB engine
        cf_only = [gid for gid in cf_scores if gid not in cb_scores]
        if cf_only and self.cb_engine is not None and user_history:
            cb_cross_scores = self.cb_engine.score_items(user_history, cf_only)
            cb_scores.update(cb_cross_scores)

        # Score CB candidates with CF engine
        cb_only = [gid for gid in cb_scores if gid not in cf_scores]
        if cb_only and self.cf_engine is not None:
            for gid in cb_only:
                cf_scores[gid] = self.cf_engine.predict_score(user_id, gid)

        # --- Normalize scores to [0, 1] ---
        cf_scores_norm = self._normalize_scores(cf_scores)
        cb_scores_norm = self._normalize_scores(cb_scores)

        # --- Blend scores ---
        blended = {}
        for game_id in all_candidates:
            cf_s = cf_scores_norm.get(game_id, 0.0)
            cb_s = cb_scores_norm.get(game_id, 0.0)

            # Weighted blend
            blend_score = cf_weight * cf_s + cb_weight * cb_s

            # Wilson quality adjustment
            wilson_s = self._wilson_lookup.get(game_id, 0.5)
            final_score = (1 - self.wilson_weight) * blend_score + self.wilson_weight * wilson_s

            blended[game_id] = {
                "app_id": game_id,
                "final_score": final_score,
                "cf_score": cf_s,
                "cb_score": cb_s,
                "wilson_score": wilson_s,
                "cf_weight": cf_weight,
                "cb_weight": cb_weight,
            }

        # --- Sort and select top-N (filtering out DLC/junk) ---
        valid_blended = [
            v for v in blended.values()
            if v["app_id"] in self._recommendable_ids
        ]
        sorted_recs = sorted(valid_blended, key=lambda x: x["final_score"], reverse=True)[:n]

        # --- Add metadata and explanations ---
        results = []
        for rec in sorted_recs:
            game_id = rec["app_id"]
            result = {
                "app_id": game_id,
                "title": self._game_title_lookup.get(game_id, "Unknown"),
                "final_score": round(rec["final_score"], 4),
            }

            if explain:
                result["cf_score"] = round(rec["cf_score"], 4)
                result["cb_score"] = round(rec["cb_score"], 4)
                result["wilson_score"] = round(rec["wilson_score"], 4)

                # Determine primary source
                if cf_weight == 0:
                    result["source"] = "content-based"
                    result["reason"] = "Recommended based on similar game tags/genres (new user)"
                elif rec["cf_score"] > rec["cb_score"]:
                    result["source"] = "collaborative"
                    result["reason"] = "Users with similar taste also enjoyed this game"
                else:
                    result["source"] = "content-based"
                    result["reason"] = "This game shares tags/genres with games you've played"

                if rec["wilson_score"] > 0.8:
                    result["reason"] += " • Highly rated with statistical confidence"

            results.append(result)

        return results

    def recommend_by_games(
        self,
        game_ids: list[int],
        n: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Recommend games based on a list of input games (no user_id needed).

        This is the cold-start path used by the Streamlit demo when a new
        user selects games they've played.

        Args:
            game_ids: List of app_ids representing user's taste.
            n: Number of recommendations.

        Returns:
            List of recommendation dicts.
        """
        exclude_set = set(game_ids)

        cb_recs = self.cb_engine.recommend_for_user(
            game_ids, n=n * 5, exclude_ids=exclude_set
        )

        cb_scores = {r["app_id"]: r["content_score"] for r in cb_recs}
        cb_scores_norm = self._normalize_scores(cb_scores)

        results = []
        for game_id, cb_s in sorted(cb_scores_norm.items(), key=lambda x: x[1], reverse=True):
            # Skip DLC, junk, and low-review games
            if game_id not in self._recommendable_ids:
                continue

            wilson_s = self._wilson_lookup.get(game_id, 0.5)
            final_score = (1 - self.wilson_weight) * cb_s + self.wilson_weight * wilson_s

            result = {
                "app_id": game_id,
                "title": self._game_title_lookup.get(game_id, "Unknown"),
                "final_score": round(final_score, 4),
                "cb_score": round(cb_s, 4),
                "wilson_score": round(wilson_s, 4),
                "source": "content-based",
                "reason": "Similar tags/genres to your selected games",
            }

            if wilson_s > 0.8:
                result["reason"] += " • Highly rated with statistical confidence"

            results.append(result)
            if len(results) >= n:
                break

        return results

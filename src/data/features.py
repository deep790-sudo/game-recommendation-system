"""
Feature engineering for the recommendation system.

Transforms raw interaction data into:
- Engagement scores (combining review sentiment + playtime)
- Game content features (tags, genres, metadata)
- User profiles (aggregated preferences)
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

logger = logging.getLogger(__name__)


def compute_engagement_scores(interactions: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a continuous engagement score from binary recommendation + playtime.

    Formula:
        score = is_recommended_numeric * (1 + log1p(hours))

    This creates a continuous signal where:
        - Positive review + many hours → high score
        - Positive review + few hours → moderate score
        - Negative review → 0 (regardless of hours)

    The log transform prevents users with thousands of hours from
    dominating the signal.
    """
    interactions = interactions.copy()

    # Convert boolean/string recommendation to numeric
    if interactions["is_recommended"].dtype == bool:
        interactions["is_recommended_num"] = interactions["is_recommended"].astype(float)
    else:
        interactions["is_recommended_num"] = (
            interactions["is_recommended"].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})
            .fillna(interactions["is_recommended"].astype(float))
        )

    # Log-transform playtime to reduce skew
    interactions["log_hours"] = np.log1p(interactions["hours"].clip(lower=0))

    # Combined engagement score
    interactions["engagement_score"] = (
        interactions["is_recommended_num"] * (1 + interactions["log_hours"])
    )

    logger.info(
        f"Engagement scores: mean={interactions['engagement_score'].mean():.2f}, "
        f"median={interactions['engagement_score'].median():.2f}, "
        f"max={interactions['engagement_score'].max():.2f}"
    )

    return interactions


def build_interaction_matrix(
    interactions: pd.DataFrame,
    user_to_idx: dict[int, int],
    game_to_idx: dict[int, int],
) -> sparse.csr_matrix:
    """
    Build a sparse user×game interaction matrix from engagement scores.

    Returns:
        CSR sparse matrix of shape (n_users, n_games) with engagement scores.
    """
    # Ensure engagement scores are computed
    if "engagement_score" not in interactions.columns:
        interactions = compute_engagement_scores(interactions)

    # Map to contiguous indices
    row_indices = interactions["user_id"].map(user_to_idx)
    col_indices = interactions["app_id"].map(game_to_idx)

    # Drop any unmapped entries (users/games not in the mapping)
    valid_mask = row_indices.notna() & col_indices.notna()
    if not valid_mask.all():
        n_dropped = (~valid_mask).sum()
        logger.warning(f"Dropped {n_dropped:,} interactions with unmapped user/game IDs")
        row_indices = row_indices[valid_mask].astype(int)
        col_indices = col_indices[valid_mask].astype(int)
        values = interactions.loc[valid_mask, "engagement_score"].values
    else:
        row_indices = row_indices.astype(int)
        col_indices = col_indices.astype(int)
        values = interactions["engagement_score"].values

    n_users = len(user_to_idx)
    n_games = len(game_to_idx)

    matrix = sparse.csr_matrix(
        (values, (row_indices.values, col_indices.values)),
        shape=(n_users, n_games),
    )

    density = matrix.nnz / (n_users * n_games)
    logger.info(
        f"Interaction matrix: {n_users:,} × {n_games:,}, "
        f"{matrix.nnz:,} non-zero, density={density:.4%}"
    )

    return matrix


def build_game_content_features(games: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare game content features for the content-based engine.

    Combines tags, rating category, price tier, and platform info
    into a single text feature suitable for TF-IDF vectorization.
    """
    games = games.copy()

    # Clean and normalize tags
    games["tags_clean"] = (
        games["tags"]
        .fillna("")
        .str.lower()
        .str.replace(r"[^\w\s,]", "", regex=True)
    )

    # Create price tier feature
    games["price_tier"] = pd.cut(
        games.get("price_final", pd.Series(dtype=float)).fillna(0),
        bins=[-1, 0, 5, 15, 30, 60, float("inf")],
        labels=["free", "budget", "mid", "full_price", "premium", "luxury"],
    ).astype(str)

    # Platform features
    platform_cols = ["win", "mac", "linux"]
    for col in platform_cols:
        if col in games.columns:
            games[f"platform_{col}"] = games[col].map({True: col, False: ""}).fillna("")

    # Combine all text features into a single string for TF-IDF
    text_parts = [games["tags_clean"]]

    if "rating" in games.columns:
        text_parts.append(games["rating"].fillna("").str.lower().str.replace(" ", "_"))

    text_parts.append(games["price_tier"])

    games["content_features"] = (
        text_parts[0]
        .str.cat(text_parts[1:], sep=" ")
        .str.strip()
    )

    n_with_features = (games["content_features"].str.len() > 0).sum()
    logger.info(f"Content features built for {n_with_features:,}/{len(games):,} games")

    return games


def build_user_profiles(
    interactions: pd.DataFrame,
    games: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate user interaction history into user-level profiles.

    Features per user:
        - n_reviews: total number of reviews
        - avg_hours: average playtime across games
        - positive_ratio: fraction of positive reviews
        - top_tags: most common tags from reviewed games (for cold-start fallback)
    """
    if "engagement_score" not in interactions.columns:
        interactions = compute_engagement_scores(interactions)

    # Basic aggregations
    user_stats = (
        interactions.groupby("user_id")
        .agg(
            n_reviews=("app_id", "count"),
            avg_hours=("hours", "mean"),
            total_hours=("hours", "sum"),
            positive_ratio=("is_recommended_num", "mean"),
            avg_engagement=("engagement_score", "mean"),
        )
        .reset_index()
    )

    # Classify user activity level (used for hybrid weighting)
    user_stats["activity_level"] = pd.cut(
        user_stats["n_reviews"],
        bins=[0, 3, 10, float("inf")],
        labels=["cold", "warm", "active"],
    )

    logger.info(
        f"User profiles: {len(user_stats):,} users | "
        f"Cold: {(user_stats['activity_level'] == 'cold').sum():,}, "
        f"Warm: {(user_stats['activity_level'] == 'warm').sum():,}, "
        f"Active: {(user_stats['activity_level'] == 'active').sum():,}"
    )

    return user_stats


def prepare_all_features(data: dict[str, Any]) -> dict[str, Any]:
    """
    Master feature engineering pipeline.

    Takes the output of loader.load_and_prepare() and adds:
        - engagement scores to train/test
        - interaction matrix (sparse)
        - game content features
        - user profiles
    """
    train = compute_engagement_scores(data["train"])
    test = compute_engagement_scores(data["test"])

    interaction_matrix = build_interaction_matrix(
        train, data["user_to_idx"], data["game_to_idx"]
    )

    games = build_game_content_features(data["games"])
    user_profiles = build_user_profiles(train, games)

    return {
        **data,
        "train": train,
        "test": test,
        "interaction_matrix": interaction_matrix,
        "games": games,
        "user_profiles": user_profiles,
    }

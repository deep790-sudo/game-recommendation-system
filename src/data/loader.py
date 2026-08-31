"""
Data loading and preprocessing pipeline for Steam game recommendation data.

Handles loading raw CSVs + JSON metadata, filtering sparse entities,
temporal train/test splitting, and memory optimization.
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_MIN_USER_REVIEWS = 5
DEFAULT_MIN_GAME_REVIEWS = 20


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns and convert low-cardinality strings to categories."""
    df = df.copy()
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["object"]).columns:
        if df[col].nunique() / max(len(df), 1) < 0.5:
            df[col] = df[col].astype("category")
    return df


def load_raw_data(data_dir: str | Path) -> dict[str, Any]:
    """
    Load raw CSV files and JSON metadata from the data directory.

    Expected files:
        - games.csv: app_id, title, date_release, rating, positive_ratio,
                     user_reviews, price_final, etc.
        - games_metadata.json: {app_id: {description, tags}} — tags/descriptions
        - recommendations.csv: app_id, user_id, is_recommended, hours, date, etc.
        - users.csv: user_id, products, reviews
    """
    data_dir = Path(data_dir)

    logger.info("Loading games.csv...")
    games = pd.read_csv(data_dir / "games.csv")
    logger.info(f"  → {len(games):,} games, columns: {list(games.columns)}")

    logger.info("Loading games_metadata.json...")
    metadata_path = data_dir / "games_metadata.json"
    if metadata_path.exists():
        games_metadata = {}
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    # Extract app_id and store the rest as metadata
                    app_id = str(record.get("app_id", ""))
                    if app_id:
                        games_metadata[app_id] = record
                except json.JSONDecodeError:
                    continue
        logger.info(f"  → metadata for {len(games_metadata):,} games")
    else:
        logger.warning("games_metadata.json not found — content-based features will be limited")
        games_metadata = {}

    logger.info("Loading recommendations.csv...")
    recommendations = pd.read_csv(data_dir / "recommendations.csv")
    logger.info(f"  → {len(recommendations):,} recommendations")

    logger.info("Loading users.csv...")
    users = pd.read_csv(data_dir / "users.csv")
    logger.info(f"  → {len(users):,} users")

    return {
        "games": games,
        "games_metadata": games_metadata,
        "recommendations": recommendations,
        "users": users,
    }


def merge_metadata(games: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    """Merge JSON metadata (tags, descriptions) into the games DataFrame."""
    if not metadata:
        games["tags"] = ""
        games["description"] = ""
        return games

    # Build a DataFrame from the metadata dict
    meta_records = []
    for app_id_str, info in metadata.items():
        try:
            app_id = int(app_id_str)
        except (ValueError, TypeError):
            continue
        tags = info.get("tags", [])
        desc = info.get("description", "")
        # Tags can be a list or a comma-separated string
        if isinstance(tags, list):
            tags = ", ".join(tags)
        meta_records.append({"app_id": app_id, "tags": tags, "description": desc})

    meta_df = pd.DataFrame(meta_records)
    games = games.merge(meta_df, on="app_id", how="left")
    games["tags"] = games["tags"].fillna("")
    games["description"] = games["description"].fillna("")

    n_with_tags = (games["tags"].str.len() > 0).sum()
    logger.info(f"  → {n_with_tags:,}/{len(games):,} games have tags")

    return games


def filter_sparse_entities(
    recommendations: pd.DataFrame,
    min_user_reviews: int = DEFAULT_MIN_USER_REVIEWS,
    min_game_reviews: int = DEFAULT_MIN_GAME_REVIEWS,
) -> pd.DataFrame:
    """
    Iteratively remove users and games with too few interactions.

    This is standard practice in recommendation systems to remove noise
    and ensure meaningful signal in the interaction matrix.
    """
    initial_size = len(recommendations)

    for iteration in range(10):
        prev_size = len(recommendations)

        # Filter games with too few reviews
        game_counts = recommendations["app_id"].value_counts()
        valid_games = game_counts[game_counts >= min_game_reviews].index
        recommendations = recommendations[recommendations["app_id"].isin(valid_games)]

        # Filter users with too few reviews
        user_counts = recommendations["user_id"].value_counts()
        valid_users = user_counts[user_counts >= min_user_reviews].index
        recommendations = recommendations[recommendations["user_id"].isin(valid_users)]

        if len(recommendations) == prev_size:
            logger.info(f"  → converged after {iteration + 1} iterations")
            break

    logger.info(
        f"Filtered interactions: {initial_size:,} → {len(recommendations):,} "
        f"({len(recommendations) / initial_size:.1%} retained)"
    )
    n_users = recommendations["user_id"].nunique()
    n_games = recommendations["app_id"].nunique()
    logger.info(f"  → {n_users:,} users, {n_games:,} games remaining")

    return recommendations


def temporal_train_test_split(
    recommendations: pd.DataFrame,
    test_ratio: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Temporal split: each user's chronologically last N% of reviews become test set.

    More realistic than random splitting for recommendation systems, since in
    production you predict *future* preferences from *past* behavior.
    """
    recommendations = recommendations.sort_values(["user_id", "date"]).reset_index(drop=True)

    # Vectorized approach: compute split index per user
    user_counts = recommendations.groupby("user_id").cumcount(ascending=False)
    user_totals = recommendations.groupby("user_id")["app_id"].transform("count")
    n_test = (user_totals * test_ratio).clip(lower=1).astype(int)

    is_test = user_counts < n_test

    train = recommendations[~is_test].copy()
    test = recommendations[is_test].copy()

    logger.info(f"Train: {len(train):,} | Test: {len(test):,}")
    return train, test


def load_and_prepare(
    data_dir: str | Path,
    min_user_reviews: int = DEFAULT_MIN_USER_REVIEWS,
    min_game_reviews: int = DEFAULT_MIN_GAME_REVIEWS,
    test_ratio: float = 0.2,
) -> dict[str, Any]:
    """
    Full data pipeline: load → merge metadata → filter → split → optimize.

    Returns a dict with keys: 'games', 'users', 'train', 'test'.
    """
    # Load raw data
    raw = load_raw_data(data_dir)
    games = raw["games"]
    recommendations = raw["recommendations"]
    users = raw["users"]

    # Merge tags/descriptions from metadata
    games = merge_metadata(games, raw["games_metadata"])

    # Filter sparse entities
    recommendations = filter_sparse_entities(
        recommendations, min_user_reviews, min_game_reviews
    )

    # Keep only games and users that survived filtering
    valid_game_ids = set(recommendations["app_id"].unique())
    valid_user_ids = set(recommendations["user_id"].unique())
    games = games[games["app_id"].isin(valid_game_ids)].copy()
    users = users[users["user_id"].isin(valid_user_ids)].copy()

    # Temporal train/test split
    train, test = temporal_train_test_split(recommendations, test_ratio)

    # Optimize memory
    games = optimize_dtypes(games)
    train = optimize_dtypes(train)
    test = optimize_dtypes(test)
    users = optimize_dtypes(users)

    # Build ID mappings (contiguous indices for matrix operations)
    unique_users = sorted(train["user_id"].unique())
    unique_games = sorted(train["app_id"].unique())
    user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
    game_to_idx = {gid: idx for idx, gid in enumerate(unique_games)}
    idx_to_user = {idx: uid for uid, idx in user_to_idx.items()}
    idx_to_game = {idx: gid for gid, idx in game_to_idx.items()}

    logger.info(
        f"Final dataset: {len(games):,} games, {len(unique_users):,} users, "
        f"{len(train):,} train + {len(test):,} test interactions"
    )

    return {
        "games": games,
        "users": users,
        "train": train,
        "test": test,
        "user_to_idx": user_to_idx,
        "game_to_idx": game_to_idx,
        "idx_to_user": idx_to_user,
        "idx_to_game": idx_to_game,
    }

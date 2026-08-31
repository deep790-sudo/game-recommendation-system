"""
Wilson Score Quality Ranker.

Implements the Wilson lower-bound confidence interval for ranking games
by review quality. This surfaces high-quality games with statistical
confidence, avoiding the pitfall of ranking a game with 3/3 positive
reviews above one with 9500/10000.

Reference:
    Wilson, E.B. (1927). "Probable inference, the law of succession,
    and statistical inference". Journal of the American Statistical
    Association, 22(158), 209-212.
"""

import logging

import numpy as np
import pandas as pd
from scipy.stats import norm

logger = logging.getLogger(__name__)


def wilson_lower_bound(
    positive: int | np.ndarray,
    total: int | np.ndarray,
    confidence: float = 0.95,
) -> float | np.ndarray:
    """
    Calculate the Wilson score lower bound for a proportion.

    This is the lower bound of the confidence interval for a Bernoulli
    parameter (the "true" positive ratio), which accounts for sample size.

    A game with 3/3 positive reviews → Wilson ≈ 0.44
    A game with 950/1000 positive reviews → Wilson ≈ 0.93

    The latter is ranked higher because we have more statistical evidence.

    Args:
        positive: Number of positive reviews (or array).
        total: Total number of reviews (or array).
        confidence: Confidence level (default 95%).

    Returns:
        Wilson lower bound score(s) in [0, 1].
    """
    if isinstance(total, (int, float)):
        if total == 0:
            return 0.0
    else:
        # Array case
        total = np.asarray(total, dtype=float)
        positive = np.asarray(positive, dtype=float)

    z = norm.ppf(1 - (1 - confidence) / 2)

    p_hat = positive / np.maximum(total, 1)  # Avoid division by zero

    denominator = 1 + z**2 / np.maximum(total, 1)
    centre = p_hat + z**2 / (2 * np.maximum(total, 1))
    spread = z * np.sqrt(
        (p_hat * (1 - p_hat) + z**2 / (4 * np.maximum(total, 1)))
        / np.maximum(total, 1)
    )

    lower_bound = (centre - spread) / denominator

    # Clip to [0, 1] and handle edge cases
    if isinstance(lower_bound, np.ndarray):
        lower_bound = np.clip(lower_bound, 0, 1)
        lower_bound[total == 0] = 0.0
    else:
        lower_bound = max(0.0, min(1.0, lower_bound))

    return lower_bound


def compute_game_quality_scores(games: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Wilson quality scores for all games.

    Uses the positive_ratio (percentage) and user_reviews (total count)
    columns from the games DataFrame.

    Returns:
        Games DataFrame with added columns:
        - wilson_score: Wilson lower bound
        - quality_rank: rank by Wilson score (1 = highest quality)
        - is_hidden_gem: True if quality is high but review count is below median
    """
    games = games.copy()

    # Compute positive and total review counts
    if "positive_ratio" in games.columns and "user_reviews" in games.columns:
        total = games["user_reviews"].fillna(0).astype(float)
        positive_ratio = games["positive_ratio"].fillna(50).astype(float) / 100.0
        positive = (total * positive_ratio).round().astype(float)
    else:
        logger.warning(
            "Missing 'positive_ratio' or 'user_reviews' columns. "
            "Wilson scores will be zero."
        )
        games["wilson_score"] = 0.0
        games["quality_rank"] = range(1, len(games) + 1)
        games["is_hidden_gem"] = False
        return games

    # Compute Wilson lower bound
    games["wilson_score"] = wilson_lower_bound(positive.values, total.values)

    # Rank by Wilson score
    games["quality_rank"] = games["wilson_score"].rank(ascending=False, method="min").astype(int)

    # Identify hidden gems: high quality + low visibility
    median_reviews = total.median()
    quality_threshold = games["wilson_score"].quantile(0.75)
    games["is_hidden_gem"] = (
        (games["wilson_score"] >= quality_threshold)
        & (total < median_reviews)
        & (total >= 10)  # Minimum evidence threshold
    )

    n_gems = games["is_hidden_gem"].sum()
    logger.info(
        f"Wilson scores computed. "
        f"Range: [{games['wilson_score'].min():.3f}, {games['wilson_score'].max():.3f}]. "
        f"Hidden gems: {n_gems:,}"
    )

    return games


def get_hidden_gems(
    games: pd.DataFrame,
    n: int = 20,
    min_reviews: int = 10,
    tags_filter: str | None = None,
) -> pd.DataFrame:
    """
    Get top hidden gems — high Wilson score but under-discovered titles.

    Args:
        games: Games DataFrame with Wilson scores computed.
        n: Number of hidden gems to return.
        min_reviews: Minimum review count (for statistical validity).
        tags_filter: Optional tag substring to filter by genre.

    Returns:
        DataFrame of hidden gem games sorted by Wilson score.
    """
    mask = (
        (games["user_reviews"] >= min_reviews)
        & (games["user_reviews"] < games["user_reviews"].quantile(0.5))
    )

    if tags_filter and "tags" in games.columns:
        mask = mask & games["tags"].str.contains(tags_filter, case=False, na=False)

    gems = games[mask].nlargest(n, "wilson_score")

    return gems[
        ["app_id", "title", "tags", "wilson_score", "positive_ratio", "user_reviews"]
    ].reset_index(drop=True)

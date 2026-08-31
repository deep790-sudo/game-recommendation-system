"""
Evaluation metrics for recommendation systems.

Implements ranking metrics (Precision@K, Recall@K, NDCG@K) and
beyond-accuracy metrics (Coverage, Novelty) for comparing
recommendation strategies.
"""

import logging
from typing import Any, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Fraction of top-K recommendations that are relevant."""
    rec_at_k = recommended[:k]
    if not rec_at_k:
        return 0.0
    hits = sum(1 for r in rec_at_k if r in relevant)
    return hits / k


def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Fraction of relevant items that appear in top-K recommendations."""
    if not relevant:
        return 0.0
    rec_at_k = recommended[:k]
    hits = sum(1 for r in rec_at_k if r in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain at K.

    Accounts for the position of relevant items — hitting at rank 1
    is better than hitting at rank 10.
    """
    rec_at_k = recommended[:k]
    if not rec_at_k or not relevant:
        return 0.0

    # DCG
    dcg = sum(
        1.0 / np.log2(i + 2)  # i+2 because log2(1)=0
        for i, item in enumerate(rec_at_k)
        if item in relevant
    )

    # Ideal DCG (all relevant items at top)
    ideal_length = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_length))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def catalog_coverage(
    all_recommendations: list[list[int]],
    total_games: int,
) -> float:
    """Fraction of the catalog that appears in any recommendation list."""
    recommended_items = set()
    for recs in all_recommendations:
        recommended_items.update(recs)
    return len(recommended_items) / total_games if total_games > 0 else 0.0


def novelty(
    recommended: list[int],
    item_popularity: dict[int, float],
) -> float:
    """
    Average self-information of recommended items.

    Higher novelty means recommending less popular (more surprising) items.
    novelty = mean(-log2(popularity)) for each recommended item.
    """
    if not recommended:
        return 0.0
    scores = []
    for item in recommended:
        pop = item_popularity.get(item, 1e-10)
        scores.append(-np.log2(max(pop, 1e-10)))
    return float(np.mean(scores))


def evaluate_recommender(
    recommend_fn: Callable[[int, int], list[int]],
    test_data: pd.DataFrame,
    train_data: pd.DataFrame,
    k_values: list[int] = [5, 10, 20],
    n_sample_users: int = 5000,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Evaluate a recommender function across multiple users and K values.

    Args:
        recommend_fn: Function(user_id, n) → list of recommended app_ids.
        test_data: Test interactions DataFrame.
        train_data: Training interactions DataFrame (for popularity).
        k_values: List of K values for Precision@K, Recall@K, NDCG@K.
        n_sample_users: Number of users to evaluate (sampling for speed).
        random_state: Random seed for user sampling.

    Returns:
        Dict with metric results and per-user details.
    """
    rng = np.random.RandomState(random_state)

    # Get users with test interactions
    test_users = test_data["user_id"].unique()
    if len(test_users) > n_sample_users:
        test_users = rng.choice(test_users, size=n_sample_users, replace=False)

    logger.info(f"Evaluating on {len(test_users):,} sampled users...")

    # Build test ground truth: user → set of relevant game ids
    test_ground_truth = (
        test_data[test_data["user_id"].isin(test_users)]
        .groupby("user_id")["app_id"]
        .apply(set)
        .to_dict()
    )

    # Compute item popularity from training data (for novelty metric)
    game_counts = train_data["app_id"].value_counts()
    total_interactions = len(train_data)
    item_popularity = (game_counts / total_interactions).to_dict()

    max_k = max(k_values)

    # Collect metrics
    metrics = {f"precision@{k}": [] for k in k_values}
    metrics.update({f"recall@{k}": [] for k in k_values})
    metrics.update({f"ndcg@{k}": [] for k in k_values})
    metrics["novelty"] = []
    all_recs = []

    n_failed = 0
    for i, user_id in enumerate(test_users):
        if (i + 1) % 1000 == 0:
            logger.info(f"  Progress: {i + 1}/{len(test_users)}")

        relevant = test_ground_truth.get(user_id, set())
        if not relevant:
            continue

        try:
            recs = recommend_fn(user_id, max_k)
        except Exception:
            n_failed += 1
            continue

        if not recs:
            n_failed += 1
            continue

        all_recs.append(recs)

        for k in k_values:
            metrics[f"precision@{k}"].append(precision_at_k(recs, relevant, k))
            metrics[f"recall@{k}"].append(recall_at_k(recs, relevant, k))
            metrics[f"ndcg@{k}"].append(ndcg_at_k(recs, relevant, k))

        metrics["novelty"].append(novelty(recs, item_popularity))

    # Aggregate
    total_games = train_data["app_id"].nunique()
    results = {}
    for metric_name, values in metrics.items():
        if values:
            results[metric_name] = float(np.mean(values))

    results["coverage"] = catalog_coverage(all_recs, total_games)
    results["n_users_evaluated"] = len(test_users) - n_failed
    results["n_failed"] = n_failed

    return results


def compare_recommenders(
    recommenders: dict[str, Callable[[int, int], list[int]]],
    test_data: pd.DataFrame,
    train_data: pd.DataFrame,
    k_values: list[int] = [5, 10, 20],
    n_sample_users: int = 5000,
) -> pd.DataFrame:
    """
    Compare multiple recommender strategies side by side.

    Args:
        recommenders: Dict mapping strategy name → recommend function.
        test_data, train_data, k_values, n_sample_users: see evaluate_recommender.

    Returns:
        DataFrame with rows=strategies, columns=metrics.
    """
    all_results = {}

    for name, recommend_fn in recommenders.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Evaluating: {name}")
        logger.info(f"{'='*50}")

        results = evaluate_recommender(
            recommend_fn, test_data, train_data, k_values, n_sample_users
        )
        all_results[name] = results

    comparison_df = pd.DataFrame(all_results).T
    comparison_df.index.name = "Strategy"

    return comparison_df

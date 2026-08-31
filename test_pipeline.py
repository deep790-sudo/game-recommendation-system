"""
End-to-end pipeline test.

Runs the full pipeline: load data → feature engineering → fit all models
→ generate hybrid recommendations → sanity check the output.
"""

import logging
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    from src.data.loader import load_and_prepare
    from src.data.features import prepare_all_features
    from src.models.content_based import ContentBasedEngine
    from src.models.collaborative import CollaborativeFilteringEngine
    from src.models.wilson_score import compute_game_quality_scores
    from src.models.hybrid import HybridRecommender

    t0 = time.time()

    # ── 1. Load & prepare data ──────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1: Loading and preparing data...")
    data = load_and_prepare("data/raw")

    # ── 2. Feature engineering ──────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2: Feature engineering...")
    data = prepare_all_features(data)

    # ── 3. Fit content-based engine ─────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3: Fitting content-based engine...")
    cb_engine = ContentBasedEngine(max_features=5000)
    cb_engine.fit(data["games"])

    # Quick test: similar games to a popular title
    sample_game = data["games"].iloc[0]
    sample_id = sample_game["app_id"]
    sample_title = sample_game["title"]
    similar = cb_engine.get_similar_games(sample_id, n=5)
    logger.info(f"  Similar to '{sample_title}':")
    for s in similar:
        title = data["games"].loc[data["games"]["app_id"] == s["app_id"], "title"].values
        title = title[0] if len(title) > 0 else "Unknown"
        logger.info(f"    → {title} (score: {s['similarity_score']:.3f})")

    # ── 4. Fit collaborative filtering ──────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 4: Fitting collaborative filtering (SVD)...")
    cf_engine = CollaborativeFilteringEngine(n_components=100)
    cf_engine.fit(
        data["interaction_matrix"],
        data["user_to_idx"],
        data["game_to_idx"],
    )

    # Quick test: recommendations for a random user
    sample_user_id = list(data["user_to_idx"].keys())[0]
    cf_recs = cf_engine.recommend_for_user(sample_user_id, n=5)
    logger.info(f"  CF recs for user {sample_user_id}:")
    for r in cf_recs:
        title = data["games"].loc[data["games"]["app_id"] == r["app_id"], "title"].values
        title = title[0] if len(title) > 0 else "Unknown"
        logger.info(f"    → {title} (score: {r['cf_score']:.3f})")

    # ── 5. Wilson scores ────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 5: Computing Wilson quality scores...")
    games = compute_game_quality_scores(data["games"])

    top_wilson = games.nlargest(5, "wilson_score")[["title", "wilson_score", "user_reviews"]]
    logger.info("  Top 5 by Wilson score:")
    for _, row in top_wilson.iterrows():
        logger.info(f"    → {row['title']} (wilson: {row['wilson_score']:.3f}, reviews: {int(row['user_reviews']):,})")

    # ── 6. Hybrid recommendations ───────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 6: Generating hybrid recommendations...")
    hybrid = HybridRecommender(wilson_weight=0.05)
    hybrid.fit(cf_engine, cb_engine, games, data["train"])

    # Test with a real user
    hybrid_recs = hybrid.recommend(sample_user_id, n=10, explain=True)
    logger.info(f"  Hybrid recs for user {sample_user_id}:")
    for i, r in enumerate(hybrid_recs, 1):
        logger.info(
            f"    {i}. {r['title']} "
            f"(final: {r['final_score']:.3f}, "
            f"cf: {r['cf_score']:.3f}, "
            f"cb: {r['cb_score']:.3f}, "
            f"wilson: {r['wilson_score']:.3f}) "
            f"[{r['source']}]"
        )

    # Test cold-start path (game-based, no user history)
    sample_games = data["games"]["app_id"].head(3).tolist()
    cold_recs = hybrid.recommend_by_games(sample_games, n=5)
    logger.info(f"\n  Cold-start recs (based on games {sample_games}):")
    for i, r in enumerate(cold_recs, 1):
        logger.info(f"    {i}. {r['title']} (score: {r['final_score']:.3f}) [{r['source']}]")

    # ── Summary ─────────────────────────────────────────────────
    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info(f"✅ Full pipeline completed in {elapsed:.1f}s")
    logger.info(f"   Games: {len(games):,} | Users: {len(data['user_to_idx']):,}")
    logger.info(f"   CF components: {cf_engine.svd.n_components}")
    logger.info(f"   CB features: {cb_engine.tfidf_matrix.shape[1]:,}")
    logger.info(f"   Hybrid recs generated successfully.")


if __name__ == "__main__":
    main()

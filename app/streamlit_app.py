"""
Streamlit Demo App — Hybrid Game Recommendation System.

An interactive demo where users can:
1. Search and select games they've played
2. Get personalized recommendations with explanations
3. Explore hidden gems by genre
"""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.content_based import ContentBasedEngine
from src.models.collaborative import CollaborativeFilteringEngine
from src.models.wilson_score import compute_game_quality_scores, get_hidden_gems
from src.models.hybrid import HybridRecommender
from src.data.loader import load_and_prepare
from src.data.features import prepare_all_features


# ─────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎮 Game Recommender",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────
# Caching — load data and models once
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading data and training models...")
def load_models():
    """Load data and fit all recommendation models."""
    data_dir = PROJECT_ROOT / "data" / "raw"

    if not (data_dir / "games.csv").exists():
        st.error(
            "⚠️ Dataset not found! Please download the Steam dataset from Kaggle "
            "and place CSV/JSON files in `data/raw/`."
        )
        st.stop()

    # Load and prepare data
    data = load_and_prepare(data_dir)
    data = prepare_all_features(data)

    # Fit content-based engine
    cb_engine = ContentBasedEngine(max_features=5000)
    cb_engine.fit(data["games"])

    # Fit collaborative filtering engine
    cf_engine = CollaborativeFilteringEngine(n_components=100)
    cf_engine.fit(
        data["interaction_matrix"],
        data["user_to_idx"],
        data["game_to_idx"],
    )

    # Compute Wilson scores
    games = compute_game_quality_scores(data["games"])

    # Fit hybrid recommender
    hybrid = HybridRecommender(wilson_weight=0.05)
    hybrid.fit(cf_engine, cb_engine, games, data["train"])

    return {
        "hybrid": hybrid,
        "games": games,
        "data": data,
        "cb_engine": cb_engine,
        "cf_engine": cf_engine,
    }


# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
def render_sidebar(games: pd.DataFrame):
    st.sidebar.title("🎮 Game Recommender")
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "A **hybrid recommendation system** combining:\n"
        "- 🤝 Collaborative Filtering (SVD)\n"
        "- 📋 Content-Based Similarity (TF-IDF)\n"
        "- 📊 Wilson Score Quality Ranking"
    )
    st.sidebar.markdown("---")
    st.sidebar.metric("Games in catalog", f"{len(games):,}")
    st.sidebar.metric(
        "Avg Wilson Score",
        f"{games['wilson_score'].mean():.3f}",
    )
    st.sidebar.metric(
        "Hidden Gems Found",
        f"{games['is_hidden_gem'].sum():,}",
    )


# ─────────────────────────────────────────────────────────────
# Recommendation Tab
# ─────────────────────────────────────────────────────────────
def render_recommendations_tab(models: dict):
    st.header("🎯 Get Recommendations")
    st.markdown(
        "Select games you've played and enjoyed, and we'll recommend similar titles."
    )

    games = models["games"]
    hybrid = models["hybrid"]

    # Game search and selection — filter to games with 50+ reviews for a responsive dropdown
    searchable_games = (
        games[games["user_reviews"] >= 50]
        .sort_values("user_reviews", ascending=False)
    )
    game_titles = searchable_games["title"].dropna().unique().tolist()

    selected_titles = st.multiselect(
        "Search and select games you've played:",
        options=game_titles,
        placeholder="Start typing a game name...",
        max_selections=20,
    )

    n_recs = st.slider("Number of recommendations:", min_value=5, max_value=30, value=10)

    if selected_titles and st.button("🚀 Get Recommendations", type="primary"):
        # Map titles to app_ids
        title_to_id = dict(zip(games["title"], games["app_id"]))
        selected_ids = [title_to_id[t] for t in selected_titles if t in title_to_id]

        if not selected_ids:
            st.warning("Could not find the selected games in our catalog.")
            return

        with st.spinner("Computing recommendations..."):
            recs = hybrid.recommend_by_games(selected_ids, n=n_recs)

        if not recs:
            st.warning("No recommendations found. Try selecting different games.")
            return

        st.success(f"Found {len(recs)} recommendations!")

        # Display recommendations
        for i, rec in enumerate(recs, 1):
            game_info = games[games["app_id"] == rec["app_id"]]

            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])

                with col1:
                    st.markdown(f"**{i}. {rec['title']}**")
                    if not game_info.empty:
                        row = game_info.iloc[0]
                        tags = str(row.get("tags", ""))[:100]
                        if tags:
                            st.caption(f"🏷️ {tags}")

                with col2:
                    st.metric("Score", f"{rec['final_score']:.3f}")

                with col3:
                    wilson = rec.get("wilson_score", 0)
                    if wilson > 0.8:
                        st.markdown("⭐ **High Quality**")
                    elif wilson > 0.6:
                        st.markdown("👍 Good Quality")

                # Expandable explanation
                with st.expander("Why this recommendation?"):
                    st.markdown(f"**Source:** {rec.get('source', 'hybrid')}")
                    st.markdown(f"**Reason:** {rec.get('reason', 'N/A')}")

                    explanation_cols = st.columns(3)
                    with explanation_cols[0]:
                        st.metric("Content Score", f"{rec.get('cb_score', 0):.3f}")
                    with explanation_cols[1]:
                        st.metric("Wilson Quality", f"{rec.get('wilson_score', 0):.3f}")
                    with explanation_cols[2]:
                        if not game_info.empty:
                            st.metric(
                                "Reviews",
                                f"{int(game_info.iloc[0].get('user_reviews', 0)):,}",
                            )

                st.markdown("---")


# ─────────────────────────────────────────────────────────────
# Hidden Gems Tab
# ─────────────────────────────────────────────────────────────
def render_hidden_gems_tab(models: dict):
    st.header("💎 Hidden Gems")
    st.markdown(
        "High-quality games that are under-discovered — "
        "ranked by **Wilson score confidence interval**, not raw popularity."
    )

    games = models["games"]

    # Tag filter
    all_tags = set()
    for tags in games["tags"].dropna():
        for tag in str(tags).split(","):
            tag = tag.strip()
            if tag:
                all_tags.add(tag)

    sorted_tags = sorted(all_tags)[:200]  # Limit for performance
    selected_tag = st.selectbox(
        "Filter by genre/tag:",
        options=["All Genres"] + sorted_tags,
    )

    n_gems = st.slider("Number of hidden gems:", min_value=5, max_value=50, value=20)

    tag_filter = None if selected_tag == "All Genres" else selected_tag

    gems = get_hidden_gems(games, n=n_gems, min_reviews=10, tags_filter=tag_filter)

    if gems.empty:
        st.info("No hidden gems found for this filter. Try a different genre.")
        return

    # Display as styled dataframe
    st.dataframe(
        gems.rename(columns={
            "title": "Game",
            "tags": "Tags",
            "wilson_score": "Wilson Score",
            "positive_ratio": "Positive %",
            "user_reviews": "Total Reviews",
        }),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Wilson Score": st.column_config.ProgressColumn(
                min_value=0, max_value=1, format="%.3f",
            ),
            "Positive %": st.column_config.NumberColumn(format="%d%%"),
        },
    )


# ─────────────────────────────────────────────────────────────
# System Info Tab
# ─────────────────────────────────────────────────────────────
def render_system_tab(models: dict):
    st.header("ℹ️ System Architecture")

    st.markdown("""
    ### How It Works

    This system uses a **hybrid approach** combining three recommendation strategies:

    #### 1. 🤝 Collaborative Filtering (SVD)
    - Factorizes the user×game interaction matrix using **Truncated SVD**
    - Discovers latent factors: "users who liked X also liked Y"
    - Works best for users with substantial play history (10+ games)

    #### 2. 📋 Content-Based Filtering (TF-IDF)
    - Vectorizes game metadata (tags, genres, categories) using **TF-IDF**
    - Computes **cosine similarity** between game profiles
    - Handles **cold-start**: works even for brand-new users with no history

    #### 3. 📊 Wilson Score Quality Ranking
    - Uses **Wilson lower-bound confidence interval** on review sentiment
    - A game with 950/1000 positive reviews scores higher than 3/3
    - Surfaces **hidden gems**: high quality but under-discovered titles

    ### Adaptive Blending

    | User Type | Interactions | CF Weight | CB Weight |
    |-----------|-------------|-----------|-----------|
    | Cold | < 3 | 0% | 100% |
    | Warm | 3–10 | 50–80% | 20–50% |
    | Active | > 10 | 85% | 15% |

    Final score = `(1 - w) × blend_score + w × wilson_score` where `w = 0.05`
    """)

    data = models["data"]
    st.markdown("### Dataset Statistics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Games", f"{len(models['games']):,}")
    with col2:
        st.metric("Users", f"{len(data['user_to_idx']):,}")
    with col3:
        st.metric("Train Interactions", f"{len(data['train']):,}")
    with col4:
        st.metric("Test Interactions", f"{len(data['test']):,}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    models = load_models()
    render_sidebar(models["games"])

    tab1, tab2, tab3 = st.tabs([
        "🎯 Recommendations",
        "💎 Hidden Gems",
        "ℹ️ How It Works",
    ])

    with tab1:
        render_recommendations_tab(models)
    with tab2:
        render_hidden_gems_tab(models)
    with tab3:
        render_system_tab(models)


if __name__ == "__main__":
    main()

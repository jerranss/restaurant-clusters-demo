# maps_app.py — Interactive restaurant clusters & per-persona predictions

import math
import statistics as stats

import streamlit as st
import folium
from streamlit_folium import st_folium
from branca.colormap import linear

# ----- import from your project -----
from abstractions import (
    restaurant_name,
    restaurant_location,
    restaurant_categories,
    restaurant_scores,
)
from recommend import (
    k_means,
    ALL_RESTAURANTS,
    best_predictor,
    feature_set,
)

# -------------------------------------------------
# Page & layout
# -------------------------------------------------
st.set_page_config(page_title="Restaurant Clusters Demo", layout="wide")
st.title("Restaurant Clusters & Predictions — Interactive Demo")

with st.expander("Guide: Understanding the Visualization", expanded=False):
    st.markdown(
        """
**Clusters (k-means)**
- Restaurants are grouped into *k* clusters by geographic proximity.
- Each cluster has a centroid (yellow circle) at the average location of its points.
- Increasing *k* creates more, smaller clusters; decreasing *k* merges them into larger regions.

**Predicted Ratings**
- When “Color by Predicted Rating” is enabled, points are colored by a rating scale (blue → yellow).
  - Darker values indicate higher predicted rating (or average historical score if a predictor is unavailable).
  - Marker size also scales with the rating.
- When it is disabled, points are colored by their cluster.

**User Personas**
- Each persona represents a simulated user with distinct tastes:
  - `likes_everything` — broadly gives high ratings.
  - `likes_expensive` — prefers pricier restaurants.
  - `likes_southside` — favors Southside Berkeley.
  - `one_cluster` — reviews are concentrated in a small area.
  - `test_user` — a neutral baseline user.
Changing the persona updates how ratings are predicted.

**How to explore**
1. Start with *k = 1* for a smooth first view; increase *k* to explore neighborhood structure.
2. Choose a *Category* to filter.
3. Optionally enable *Color by Predicted Rating* and select a *User Persona* to view personalized preferences.
        """
    )

# -------------------------------------------------
# Helper functions
# -------------------------------------------------
@st.cache_data
def get_all_categories():
    """Unique sorted category names from the dataset (as stored)."""
    cats = set()
    for r in ALL_RESTAURANTS:
        for c in restaurant_categories(r):
            if isinstance(c, str) and c.strip():
                cats.add(c.strip())
    return sorted(cats)

def filter_restaurants_case_exact(query: str, restaurants):
    """Case-insensitive exact match on one category label. '(All)' returns all."""
    if not query or query == "(All)":
        return restaurants
    q = query.lower()
    return [
        r for r in restaurants
        if any((c or "").lower() == q for c in restaurant_categories(r))
    ]

def group_by_centroid(restaurants, centroids):
    """Return list of clusters (one list per centroid)."""
    buckets = {tuple(c): [] for c in centroids}
    for r in restaurants:
        lat, lon = restaurant_location(r)
        closest = min(centroids, key=lambda c: (lat - c[0]) ** 2 + (lon - c[1]) ** 2)
        buckets[tuple(closest)].append(r)
    return [buckets[tuple(c)] for c in centroids]

def mean_latlon(restaurants):
    xs = [restaurant_location(r)[0] for r in restaurants]
    ys = [restaurant_location(r)[1] for r in restaurants]
    return (sum(xs) / len(xs), sum(ys) / len(ys)) if restaurants else (0, 0)

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

@st.cache_data
def cached_centroids(loc_key, k):
    """Cache key wrapper so k-means results are cached per (locations, k)."""
    return (loc_key, k)

def as_loc_key(restaurants):
    """Turn the current restaurant list into a hashable key for caching."""
    return tuple(
        (restaurant_location(r)[0], restaurant_location(r)[1], restaurant_name(r))
        for r in restaurants
    )

# -------------------------------------------------
# Sidebar controls
# -------------------------------------------------
with st.sidebar:
    st.header("Controls")
    k = st.slider("Clusters (k)", min_value=1, max_value=10, value=1, step=1)

    categories = get_all_categories()
    cat = st.selectbox("Category", ["(All)"] + categories, index=0)

    show_labels = st.checkbox("Show restaurant names", value=False)
    color_by_pred = st.checkbox("Color by Predicted Rating", value=False)

    # Personas available in your users/ folder
    usernames = [
        "likes_everything",
        "likes_expensive",
        "likes_southside",
        "one_cluster",
        "test_user",
    ]
    active_user = st.selectbox("User Persona (for predictions)", usernames, index=1)

    st.caption("Tip: Start with k = 1 for smooth performance, then increase k to explore clusters.")

# -------------------------------------------------
# Data subset
# -------------------------------------------------
restaurants = filter_restaurants_case_exact(cat, ALL_RESTAURANTS)
if not restaurants:
    st.warning("No restaurants matched that category.")
    st.stop()

# -------------------------------------------------
# Clustering (with a lightweight cache key)
# -------------------------------------------------
loc_key = as_loc_key(restaurants)
_ = cached_centroids(loc_key, k)  # register inputs with Streamlit cache
centroids = k_means(restaurants, k)
clusters = group_by_centroid(restaurants, centroids)

# -------------------------------------------------
# Predicted scores (per-persona if possible; fallback to avg score)
# -------------------------------------------------
pred_scores = None
score_min = score_max = None
color_mode_label = "Cluster colors"

if color_by_pred:
    predictor = None

    # Try to load an actual user object if your project exposes load_user()
    user_obj = None
    try:
        from recommend import load_user
        user_obj = load_user(active_user)
    except Exception:
        user_obj = None  # some templates don't expose a loader; we'll still fall back

    # Try to build a true predictor; otherwise we'll compute average scores
    if user_obj is not None:
        try:
            predictor = best_predictor(user_obj, restaurants, feature_set())
        except Exception:
            predictor = None

    def score_for(r):
        # If a predictor worked, use it; otherwise average past scores or default to 3.0.
        if predictor is not None:
            try:
                return float(predictor(r))
            except Exception:
                pass
        s = restaurant_scores(r)
        return float(stats.mean(s)) if s else 3.0

    pred_scores = {restaurant_name(r): score_for(r) for r in restaurants}
    vals = list(pred_scores.values())
    if vals:
        score_min, score_max = min(vals), max(vals)
        if math.isclose(score_min, score_max):
            score_min, score_max = score_min - 0.1, score_max + 0.1
        color_mode_label = (
            "Predicted rating (per persona)" if predictor else "Average historical rating"
        )

# Small status line so it’s clear how coloring works right now
st.caption(f"Coloring mode: {color_mode_label}")

# -------------------------------------------------
# Map render
# -------------------------------------------------
center_lat, center_lon = mean_latlon(restaurants)
m = folium.Map(location=[center_lat, center_lon], zoom_start=13, control_scale=True)

# Legend / colors
if color_by_pred and pred_scores:
    cmap = linear.YlGnBu_09.scale(score_min, score_max)
    cmap.caption = "Rating scale"
    cmap.add_to(m)
else:
    palette = [
        "red", "blue", "green", "purple", "orange",
        "darkred", "lightblue", "darkgreen", "cadetblue", "pink",
    ]

# Draw centroids
for i, c in enumerate(centroids):
    folium.CircleMarker(
        location=[c[0], c[1]],
        radius=8,
        color="black",
        weight=2,
        fill=True,
        fill_color="yellow",
        fill_opacity=0.95,
        tooltip=f"Centroid {i + 1}",
    ).add_to(m)

# Draw restaurants
for i, cluster in enumerate(clusters):
    for r in cluster:
        lat, lon = restaurant_location(r)
        name = restaurant_name(r)
        cats = ", ".join(restaurant_categories(r))

        if color_by_pred and pred_scores:
            s = pred_scores.get(name, 3.0)
            color = linear.YlGnBu_09.scale(score_min, score_max)(s)
            radius = clamp(3 + (s - score_min) / (score_max - score_min + 1e-9) * 6, 3, 9)
            tooltip = f"{name} — {s:.2f}" if show_labels else f"{s:.2f}"
            popup = f"<b>{name}</b><br/>Rating: {s:.2f}<br/>{cats}"
        else:
            color = palette[i % len(palette)]
            radius = 4
            tooltip = name if show_labels else None
            popup = f"<b>{name}</b><br/>{cats}"

        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color=color,
            weight=1,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            tooltip=tooltip,
            popup=popup,
        ).add_to(m)

# Header metrics
left, mid, right = st.columns(3)
with left:
    st.metric("Restaurants", len(restaurants))
with mid:
    st.metric("Clusters (k)", k)
with right:
    st.metric("Category", cat if cat != "(All)" else "All")

# Render the map
st_folium(m, height=640, width=None)

from pathlib import Path
import re

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
SHORT_NEGATION_LABELS = {
    "not good": "Negative",
    "not good at all": "Negative",
    "not very good": "Negative",
    "not great": "Negative",
    "not fun": "Negative",
    "not worth it": "Negative",
    "not recommended": "Negative",
    "i did not enjoy this game": "Negative",
    "i didn't enjoy this game": "Negative",
    "not bad": "Positive",
    "not too bad": "Positive",
}

st.set_page_config(page_title="Steam Game Success Analysis", layout="wide")
st.title("Steam Game Success Analysis")


@st.cache_resource
def load_bundle(filename):
    return joblib.load(ROOT / filename)


@st.cache_data
def read_summary(filename):
    return pd.read_csv(ROOT / filename)


forecast_tab, sentiment_tab, explore_tab = st.tabs(
    ["New Game Forecast", "Review Sentiment", "Explore Steam"]
)

with forecast_tab:
    st.subheader("Forecast player reception")
    st.write("Estimate the share of positive reviews from a planned game's store details.")
    forecast = load_bundle("steam_model.joblib")

    with st.form("game_forecast"):
        price_col, genre_col, platform_col = st.columns(3)
        price = price_col.number_input("Planned price (USD)", min_value=0.0, step=0.01)
        genre = genre_col.selectbox("Primary genre", forecast["genres"])
        platforms = platform_col.selectbox(
            "Supported platforms", forecast["platform_options"],
            format_func=lambda value: " + ".join(part.title() for part in value.split(";")),
        )
        submitted = st.form_submit_button("Predict reception", type="primary")

    if submitted:
        game = pd.DataFrame([{"price": price, "genre": genre, "platforms": platforms}])
        prediction = forecast["model"].predict(game)[0]
        st.metric("Estimated share of positive reviews", f"{prediction:.1%}")

with sentiment_tab:
    st.subheader("Classify a player review")
    sentiment = load_bundle("review_sentiment.joblib")
    with st.form("review_sentiment"):
        review = st.text_area(
            "Review text", max_chars=5000,
            placeholder="The game is fun, and the controls feel responsive.",
        )
        analyze = st.form_submit_button("Analyze review", type="primary")

    if analyze:
        if review.strip():
            phrase = " ".join(re.findall(r"[a-z']+", review.lower().replace("’", "'")))
            label = SHORT_NEGATION_LABELS.get(phrase)
            if label is None:
                prediction = sentiment["model"].predict([review.strip()])[0]
                label = "Positive" if prediction == 1 else "Negative"
            st.metric("Predicted sentiment", label)
        else:
            st.error("Enter a review to analyze.")
    st.caption(
        f"TF-IDF model evaluation on balanced held-out reviews: "
        f"{sentiment['accuracy']:.1%} accuracy, {sentiment['f1']:.3f} positive-class F1."
    )
    st.caption("The app handles common short phrases such as “not good” directly.")

with explore_tab:
    st.subheader("Historical reception by genre")
    genre_summary = read_summary("genre_summary.csv")
    genre_summary = genre_summary.loc[genre_summary["games"] >= 50].nlargest(12, "games")
    genre_summary["Positive reviews (%)"] = genre_summary["positive_ratio"] * 100
    st.bar_chart(genre_summary, x="genre", y="Positive reviews (%)", horizontal=True)
    st.caption("Genres with the most games in the dataset. Each game needs at least 10 reviews.")

    st.subheader("Game clusters")
    clusters = read_summary("game_clusters.csv")
    clusters["cluster"] = clusters["cluster"].map(lambda value: f"Cluster {value + 1}")
    st.scatter_chart(clusters, x="pca1", y="pca2", color="cluster")
    st.caption("3,000 sampled games. PCA projects seven game characteristics into two dimensions.")

    st.subheader("SteamSpy tag associations")
    st.dataframe(
        read_summary("tag_rules.csv"), hide_index=True,
        column_config={
            "Support": st.column_config.NumberColumn(format="percent"),
            "Confidence": st.column_config.NumberColumn(format="percent"),
            "Lift": st.column_config.NumberColumn(format="%.2f×"),
        },
    )

    with st.expander("Notebook results · High vs. low reception"):
        st.write(
            "The notebook's Random Forest classifier uses historical Steam metadata, "
            "including playtime and estimated ownership."
        )
        results = st.columns(3)
        results[0].metric("Accuracy", "67.8%")
        results[1].metric("Precision", "68.2%")
        results[2].metric("F1", "0.676")

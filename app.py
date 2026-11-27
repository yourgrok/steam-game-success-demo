from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Steam Game Success Analysis")
st.title("Steam Game Success Analysis")
st.subheader("New Game Forecast")
st.write("Estimate the share of positive Steam reviews for a planned game.")


@st.cache_resource
def load_model():
    return joblib.load(Path(__file__).with_name("steam_model.joblib"))


bundle = load_model()

with st.form("game_forecast"):
    price = st.number_input("Planned price (USD)", min_value=0.0, step=0.01)
    genre = st.selectbox("Primary genre", bundle["genres"])
    platforms = st.selectbox(
        "Supported platforms",
        bundle["platform_options"],
        format_func=lambda value: value.replace(";", " + "),
    )
    submitted = st.form_submit_button("Predict reception")

if submitted:
    game = pd.DataFrame([{
        "price": price,
        "genre": genre,
        "platforms": platforms,
    }])
    prediction = bundle["model"].predict(game)[0]
    st.metric("Estimated positive reviews", f"{prediction:.1%}")

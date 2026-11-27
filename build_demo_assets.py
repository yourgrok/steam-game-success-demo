"""Build the Streamlit demo assets from the three CSVs used in Colab."""
import argparse
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent


def export_sentiment(path):
    reviews = pd.read_csv(path, usecols=['cleaned_review_text', 'review_score', 'app_id'])
    reviews = reviews.dropna(subset=['cleaned_review_text']).drop_duplicates(
        subset=['app_id', 'cleaned_review_text']
    )
    reviews = reviews[reviews['cleaned_review_text'].str.split().str.len() >= 3]
    samples = [
        group.sample(n=3000, random_state=42)
        for _, group in reviews.groupby('review_score')
    ]
    sample = pd.concat(samples)
    x_train, x_test, y_train, y_test = train_test_split(
        sample['cleaned_review_text'], sample['review_score'],
        test_size=1000, stratify=sample['review_score'], random_state=42,
    )
    model = make_pipeline(
        TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2),
        LogisticRegression(max_iter=1000),
    )
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)
    joblib.dump({
        'model': model, 'accuracy': accuracy_score(y_test, predicted),
        'f1': f1_score(y_test, predicted),
    }, ROOT / 'review_sentiment.joblib', compress=3)


def export_games(path):
    games = pd.read_csv(path)
    games.columns = games.columns.str.split(';').str[0].str.strip()
    games['price'] = pd.to_numeric(games['price'].astype(str).str.split(';').str[0])
    total = games['positive_ratings'] + games['negative_ratings']
    games = games.loc[total >= 10].copy()
    games['positive_ratio'] = games['positive_ratings'] / total.loc[games.index]
    genre_list = games['genres'].str.split(';')
    games['genre'] = genre_list.str[0]
    features = pd.DataFrame({
        'price': games['price'],
        'owners': np.log1p(games['owners'].str.split('-', expand=True).astype(float).mean(axis=1)),
        'playtime': np.log1p(games['average_playtime']),
        'genres': genre_list.str.len(),
        'platforms': games['platforms'].str.split(';').str.len(),
        'reception': games['positive_ratio'],
        'achievements': np.log1p(games['achievements']),
    }).fillna(0)
    scaled = RobustScaler().fit_transform(features.clip(
        lower=features.quantile(0.05), upper=features.quantile(0.95), axis=1
    ))
    with threadpool_limits(limits=2):
        games['cluster'] = KMeans(n_clusters=2, n_init=10, random_state=42).fit_predict(scaled)
        games[['pca1', 'pca2']] = PCA(n_components=2).fit_transform(scaled)
    games.groupby('genre').agg(
        games=('name', 'size'), positive_ratio=('positive_ratio', 'mean'),
    ).reset_index().to_csv(ROOT / 'genre_summary.csv', index=False)
    games.sample(n=3000, random_state=42)[
        ['name', 'cluster', 'pca1', 'pca2', 'price', 'positive_ratio']
    ].round(5).to_csv(ROOT / 'game_clusters.csv', index=False)


def export_tag_rules(path):
    tags = pd.read_csv(path).drop(columns='appid').gt(0)
    counts = tags.sum(axis=1)
    tags = tags.loc[counts.between(1, 80)]
    top_tags = tags.sum().nlargest(40).index
    rates = tags[top_tags].mean()
    rules = []
    for first, second in combinations(top_tags, 2):
        support = (tags[first] & tags[second]).mean()
        lift = support / (rates[first] * rates[second])
        if support >= 0.05 and lift >= 1.2:
            for left, right in [(first, second), (second, first)]:
                rules.append((left, right, support, support / rates[left], lift))
    pd.DataFrame(rules, columns=[
        'From tag', 'To tag', 'Support', 'Confidence', 'Lift',
    ]).sort_values('Lift', ascending=False).head(20).round(4).to_csv(
        ROOT / 'tag_rules.csv', index=False,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('games', type=Path)
    parser.add_argument('reviews', type=Path)
    parser.add_argument('tags', type=Path)
    args = parser.parse_args()
    export_sentiment(args.reviews)
    export_games(args.games)
    export_tag_rules(args.tags)


from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

RANDOM_STATE = 42
CLASSES = ["negativo","neutro","positivo"]

SPANISH_STOPWORDS = "spanish"

def build_nb_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(min_df=2, ngram_range=(1,2), stop_words=SPANISH_STOPWORDS)),
        ("clf", MultinomialNB(alpha=0.5))
    ])

def build_mlp_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(min_df=2, ngram_range=(1,2), stop_words=SPANISH_STOPWORDS)),
        ("clf", MLPClassifier(hidden_layer_sizes=(256,), random_state=RANDOM_STATE,
                              max_iter=300, early_stopping=True, n_iter_no_change=10))
    ])

def fit_predict(models_text: pd.Series, gold: pd.Series | None = None):
    X = models_text.fillna("")
    metrics = {}
    nb = build_nb_pipeline()
    nb.fit(X, gold if gold is not None else np.zeros(len(X)))
    pred_nb = nb.predict(X)
    mlp = build_mlp_pipeline()
    mlp.fit(X, gold if gold is not None else np.zeros(len(X)))
    pred_mlp = mlp.predict(X)

    if gold is not None and gold.notna().any():
        mask = gold.notna()
        g = gold[mask]
        pn = pd.Series(pred_nb)[mask]
        pm = pd.Series(pred_mlp)[mask]
        metrics["nb"] = classification_report(g, pn, output_dict=True, zero_division=0)
        metrics["mlp"] = classification_report(g, pm, output_dict=True, zero_division=0)
    return pred_nb, pred_mlp, metrics

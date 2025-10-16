
import os, pandas as pd, numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
import joblib

from .evaluate_train import export_metrics

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
OUT = BASE / "outputs"
ART = BASE / "src" / "artifacts"

def map_rating_to_sentiment(x):
    try:
        v = float(x)
    except Exception:
        return None
    if v <= 2.0: return "negativo"
    if v >= 4.0: return "positivo"
    return "neutro"

def load_rating_google():
    # Expect columns at least: Comentario, Rating (o similar)
    fname = DATA / "rating_google.xlsx"
    xl = pd.ExcelFile(fname)
    sheet = xl.sheet_names[0]
    df = pd.read_excel(fname, sheet_name=sheet)
    # Try to find rating column
    cand = [c for c in df.columns if str(c).lower() in {"rating","estrellas","stars","puntuacion"}]
    if not cand:
        raise RuntimeError("No se encontró columna de rating/estrellas en rating_google.xlsx")
    rcol = cand[0]
    # Ensure text column
    tcol = "Comentario" if "Comentario" in df.columns else df.columns[0]
    df = df[[tcol, rcol]].rename(columns={tcol:"Comentario", rcol:"Rating"}).dropna()
    df["sentimiento_gold"] = df["Rating"].map(map_rating_to_sentiment)
    df = df.dropna(subset=["Comentario","sentimiento_gold"])
    return df

def build_nb():
    return Pipeline([
        ("tfidf", TfidfVectorizer(min_df=2, ngram_range=(1,2), stop_words=['de', 'la', 'que', 'el', 'y', 'a', 'en', 'los', 'del', 'se', 'las', 'por', 'un', 'para', 'con', 'no', 'una', 'su', 'al', 'lo', 'como', 'más', 'pero', 'sus', 'le', 'ya', 'o', 'este', 'sí', 'porque', 'esta', 'entre', 'cuando', 'muy', 'sin', 'sobre', 'también', 'me', 'hasta', 'hay', 'donde', 'quien', 'desde', 'todo', 'nos', 'durante', 'todos', 'uno', 'les', 'ni', 'contra', 'otros', 'ese', 'eso', 'ante', 'ellos', 'e', 'esto', 'mí', 'antes', 'algunos', 'qué', 'unos', 'yo', 'otro', 'otras', 'otra', 'él', 'tanto', 'esa', 'estos', 'mucho', 'quienes', 'nada', 'muchos', 'cual', 'poco', 'ella', 'estar', 'estas', 'alguna', 'algo', 'nosotros', 'mi', 'mis', 'tú', 'te', 'ti', 'tu', 'tus', 'ellas', 'nosotras', 'vos', 'ustedes', 'ellos', 'ellas'])),
        ("clf", MultinomialNB(alpha=0.5))
    ])

def build_mlp():
    return Pipeline([
        ("tfidf", TfidfVectorizer(min_df=2, ngram_range=(1,2), stop_words=['de', 'la', 'que', 'el', 'y', 'a', 'en', 'los', 'del', 'se', 'las', 'por', 'un', 'para', 'con', 'no', 'una', 'su', 'al', 'lo', 'como', 'más', 'pero', 'sus', 'le', 'ya', 'o', 'este', 'sí', 'porque', 'esta', 'entre', 'cuando', 'muy', 'sin', 'sobre', 'también', 'me', 'hasta', 'hay', 'donde', 'quien', 'desde', 'todo', 'nos', 'durante', 'todos', 'uno', 'les', 'ni', 'contra', 'otros', 'ese', 'eso', 'ante', 'ellos', 'e', 'esto', 'mí', 'antes', 'algunos', 'qué', 'unos', 'yo', 'otro', 'otras', 'otra', 'él', 'tanto', 'esa', 'estos', 'mucho', 'quienes', 'nada', 'muchos', 'cual', 'poco', 'ella', 'estar', 'estas', 'alguna', 'algo', 'nosotros', 'mi', 'mis', 'tú', 'te', 'ti', 'tu', 'tus', 'ellas', 'nosotras', 'vos', 'ustedes', 'ellos', 'ellas'])),
        ("clf", MLPClassifier(hidden_layer_sizes=(256,), random_state=42,
                              max_iter=300, early_stopping=True, n_iter_no_change=10))
    ])

def main():
    OUT.mkdir(exist_ok=True, parents=True)
    ART.mkdir(exist_ok=True, parents=True)

    df = load_rating_google()
    X = df["Comentario"].astype(str)
    y = df["sentimiento_gold"].astype(str)
    # stratify split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    nb = build_nb()
    nb.fit(X_train, y_train)
    mlp = build_mlp()
    mlp.fit(X_train, y_train)

    # Save models
    joblib.dump(nb, ART / "nb_tfidf_google_ratings.joblib")
    joblib.dump(mlp, ART / "mlp_tfidf_google_ratings.joblib")

    # Evaluate on holdout
    pred_nb = nb.predict(X_test)
    pred_mlp = mlp.predict(X_test)

    export_metrics(y_test, {"NB": pred_nb, "MLP": pred_mlp}, OUT / "metricas_entrenamiento_desde_rating_google.xlsx")

    # Also export the split for reproducibility
    pd.DataFrame({"Comentario":X_test, "y_true":y_test, "y_nb":pred_nb, "y_mlp":pred_mlp}).to_excel(
        OUT / "holdout_predicciones_desde_rating_google.xlsx", index=False
    )

if __name__ == "__main__":
    main()

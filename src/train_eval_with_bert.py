
import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
import joblib
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from .evaluate_train import export_metrics
from .bert_utils import bert_predict_3, DEFAULT_BERT_MODEL

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
    fname = DATA / "rating_google.xlsx"
    xl = pd.ExcelFile(fname)
    sheet = xl.sheet_names[0]
    df = pd.read_excel(fname, sheet_name=sheet)
    cand = [c for c in df.columns if str(c).lower() in {"rating","estrellas","stars","puntuacion"}]
    if not cand:
        raise RuntimeError("No se encontró columna de rating/estrellas en rating_google.xlsx")
    rcol = cand[0]
    tcol = "Comentario" if "Comentario" in df.columns else df.columns[0]
    df = df[[tcol, rcol]].rename(columns={tcol:"Comentario", rcol:"Rating"}).dropna()
    df["sentimiento_gold"] = df["Rating"].map(map_rating_to_sentiment)
    df = df.dropna(subset=["Comentario","sentimiento_gold"])
    return df

def build_nb():
    return Pipeline([
        ("tfidf", TfidfVectorizer(min_df=2, ngram_range=(1,2))),
        ("clf", MultinomialNB(alpha=0.5))
    ])

def build_mlp():
    # Sin early_stopping para evitar el bug de validación con y categórica.
    return Pipeline([
        ("tfidf", TfidfVectorizer(min_df=2, ngram_range=(1,2))),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(256,),
            random_state=42,
            max_iter=200,
            early_stopping=False,   # <— clave
            n_iter_no_change=10
        ))
    ])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)

    df = load_rating_google()
    X = df["Comentario"].astype(str)
    y = df["sentimiento_gold"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    classes = np.array(["negativo","neutro","positivo"])
    w = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    wmap = {c: w[i] for i, c in enumerate(classes)}
    sample_weight_train = y_train.map(wmap).values

    # NB/MLP
    nb = build_nb(); nb.fit(X_train, y_train, clf__sample_weight=sample_weight_train)
    mlp = build_mlp(); mlp.fit(X_train, y_train, clf__sample_weight=sample_weight_train)

    joblib.dump(nb, ART / "nb_tfidf_google_ratings.joblib")
    joblib.dump(mlp, ART / "mlp_tfidf_google_ratings.joblib")

    pred_nb  = nb.predict(X_test)
    pred_mlp = mlp.predict(X_test)

    # BERT (requiere Internet local para descargar pesos la primera vez)
    try:
        pred_bert = bert_predict_3(X_test.tolist(), model_name=DEFAULT_BERT_MODEL, batch_size=16)
    except Exception as e:
        # Si falla (sin Internet), generar columna con 'neutro' y dejar aviso
        pred_bert = ["neutro"] * len(X_test)
        with open(OUT / "AVISO_BERT.txt","w",encoding="utf-8") as f:
            f.write(f"No se pudo ejecutar BERT localmente ({e}). Se rellenó 'neutro' como fallback.\n")
            f.write("Instale 'transformers' y 'torch', y asegure conexión a Internet para descargar pesos.\n")

    # Exportar métricas comparativas
    export_metrics(y_test, {"NB": pred_nb, "MLP": pred_mlp, "BERT": pred_bert},
                   OUT / "metricas_entrenamiento_con_bert.xlsx")

    # Export holdout detallado
    pd.DataFrame({"Comentario":X_test, "y_true":y_test,
                  "y_nb":pred_nb, "y_mlp":pred_mlp, "y_bert":pred_bert}).to_excel(
        OUT / "holdout_predicciones_con_bert.xlsx", index=False
    )

if __name__ == "__main__":
    main()

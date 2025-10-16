import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_class_weight

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .evaluate_train import export_metrics

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
OUT  = BASE / "outputs"
ART  = BASE / "src" / "artifacts"

BERT_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
LABEL_MAP_5 = {0:"muy negativo",1:"negativo",2:"neutro",3:"positivo",4:"muy positivo"}

def collapse_5_to_3(label_5: str) -> str:
    if label_5 in {"muy negativo","negativo"}: return "negativo"
    if label_5 in {"muy positivo","positivo"}: return "positivo"
    return "neutro"

def bert_probs(texts, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(BERT_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(BERT_NAME).to(device)
    model.eval()
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tok(batch, truncation=True, padding=True, return_tensors="pt").to(device)
            out = model(**enc).logits
            p = torch.softmax(out, dim=1).cpu().numpy()   # [bs,5]
            all_probs.append(p)
    return np.vstack(all_probs)

def map_rating_to_sentiment(x):
    try:
        v = float(x)
    except Exception:
        return None
    if v <= 2: return "negativo"
    if v >= 4: return "positivo"
    return "neutro"

def build_nb():
    return Pipeline([
        ("tfidf", TfidfVectorizer(min_df=2, ngram_range=(1,2), sublinear_tf=True)),
        ("clf", MultinomialNB(alpha=0.5))
    ])

def build_mlp():
    return Pipeline([
        ("tfidf", TfidfVectorizer(min_df=2, ngram_range=(1,2), sublinear_tf=True)),
        ("clf", MLPClassifier(hidden_layer_sizes=(256,), random_state=42, max_iter=200, early_stopping=False))
    ])

def ensure_three_classes(df_pseudo: pd.DataFrame, df_seed: pd.DataFrame, min_per_class=40) -> pd.DataFrame:
    """
    Garantiza que existan las 3 clases en el set de entrenamiento.
    Si falta alguna, añade ejemplos desde df_seed (gold de ratings) hasta cubrir min_per_class.
    """
    needed = {"negativo","neutro","positivo"}
    present = set(df_pseudo["sentimiento_gold"].unique())
    missing = list(needed - present)
    out = [df_pseudo]
    for cls in missing:
        cand = df_seed[df_seed["sentimiento_gold"] == cls]
        if len(cand) == 0:
            continue
        n_take = min_per_class
        out.append(cand.sample(min(n_take, len(cand)), random_state=42))
    # Además, si alguna clase presente tiene muy pocos, completar con seed
    counts = pd.concat(out).groupby("sentimiento_gold").size().to_dict()
    for cls in needed:
        have = counts.get(cls, 0)
        if have < min_per_class:
            cand = df_seed[df_seed["sentimiento_gold"] == cls]
            extra = min(min_per_class - have, max(0, len(cand)))
            if extra > 0:
                out.append(cand.sample(extra, random_state=42))
    return pd.concat(out, ignore_index=True)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)

    # 1) Cargar dataset anotado por inferencia previa (con BERT) y limpiar
    f = OUT / "dataset_anotado_con_nb_mlp_bert.xlsx"
    df_full = pd.read_excel(f)
    df_full = df_full[df_full["Comentario"].notna() & (df_full["Comentario"].str.strip()!="")].copy()

    # 2) Probabilidades BERT para confianza
    probs = bert_probs(df_full["Comentario"].tolist(), batch_size=16)
    max_idx  = probs.argmax(axis=1)
    max_prob = probs.max(axis=1)
    label_5  = [LABEL_MAP_5[i] for i in max_idx]
    label_3  = [collapse_5_to_3(x) for x in label_5]

    df_full["bert_label_3"] = label_3
    df_full["bert_max_prob"] = max_prob

    # 3) Pseudo-gold con threshold (bajar si no hay cobertura)
    THRESHOLDS = [0.80, 0.75, 0.70]
    pseudo = None
    for TH in THRESHOLDS:
        cand = df_full[df_full["bert_max_prob"] >= TH].copy()
        cand = cand.rename(columns={"bert_label_3":"sentimiento_gold"})
        cand = cand[["Comentario","sentimiento_gold"]]
        # Necesitamos al menos 2 clases para empezar
        if cand["sentimiento_gold"].nunique() >= 2 and len(cand) >= 200:
            pseudo = cand
            break
    if pseudo is None:
        # si es muy chico, tomar el mejor que tengamos
        pseudo = df_full.rename(columns={"bert_label_3":"sentimiento_gold"})[["Comentario","sentimiento_gold"]].copy()

    # 4) Seed (oro real) desde ratings para cubrir clases faltantes
    gold = pd.read_excel(DATA / "rating_google.xlsx")
    # detectar columna rating
    cand_cols = [c for c in gold.columns if str(c).lower() in {"rating","estrellas","stars","puntuacion"}]
    rcol = cand_cols[0]
    tcol = "Comentario" if "Comentario" in gold.columns else gold.columns[0]
    gold = gold[[tcol, rcol]].rename(columns={tcol:"Comentario", rcol:"Rating"}).dropna()
    gold["sentimiento_gold"] = gold["Rating"].map(map_rating_to_sentiment)
    gold = gold.dropna(subset=["Comentario","sentimiento_gold"]).copy()

    # Split del gold para una evaluación posterior (mantener comparabilidad)
    Xg = gold["Comentario"].astype(str); yg = gold["sentimiento_gold"].astype(str)
    X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(Xg, yg, test_size=0.25, random_state=42, stratify=yg)

    seed = pd.DataFrame({"Comentario": X_train_g, "sentimiento_gold": y_train_g})

    # Asegurar 3 clases en el entrenamiento (pseudo + un pequeño seed)
    train_df = ensure_three_classes(pseudo, seed, min_per_class=40)

    # 5) Entrenar NB/MLP con pesos balanceados (usando las clases reales presentes)
    Xp = train_df["Comentario"].astype(str)
    yp = train_df["sentimiento_gold"].astype(str)

    classes = np.unique(yp)
    w = compute_class_weight(class_weight="balanced", classes=classes, y=yp)
    wmap = {c: w[i] for i, c in enumerate(classes)}
    sw = yp.map(wmap).values

    nb  = build_nb();  nb.fit(Xp, yp,  clf__sample_weight=sw)
    mlp = build_mlp(); mlp.fit(Xp, yp, clf__sample_weight=sw)

    joblib.dump(nb,  ART / "nb_tfidf_pseudogold_bert.joblib")
    joblib.dump(mlp, ART / "mlp_tfidf_pseudogold_bert.joblib")

    # 6) Métricas en hold-out REAL (ratings, parte de test)
    pred_nb  = nb.predict(X_test_g)
    pred_mlp = mlp.predict(X_test_g)

    export_metrics(y_test_g, {"NB_pseudoBERT": pred_nb, "MLP_pseudoBERT": pred_mlp},
                   OUT / "metricas_reentrenadas_desde_pseudoBERT.xlsx")

    # Muestra del pseudo-gold usado (auditoría)
    train_df.sample(min(20, len(train_df)), random_state=42).to_excel(OUT / "pseudo_gold_muestra.xlsx", index=False)

if __name__ == "__main__":
    main()

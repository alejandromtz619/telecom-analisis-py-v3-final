
import os, pandas as pd, numpy as np
from pathlib import Path
import joblib

from .utils import detect_brand, safe_year, normalize_text, looks_out_of_context

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
OUT = BASE / "outputs"
ART = BASE / "src" / "artifacts"

def load_all_sources():
    frames = []
    # Google
    gfile = DATA / "dataset_google.xlsx"
    if gfile.exists():
        df = pd.read_excel(gfile, sheet_name="GoogleMaps")
        df["plataforma"] = "googlemaps"
        frames.append(df)
    # Twitter
    tfile = DATA / "dataset_twitter.xlsx"
    if tfile.exists():
        df = pd.read_excel(tfile, sheet_name="Tweets")
        df["plataforma"] = "twitter"
        frames.append(df)
    # Reddit
    rfile = DATA / "dataset_reddit.xlsx"
    if rfile.exists():
        df = pd.read_excel(rfile, sheet_name="ExportComments.com")
        df = df.rename(columns={"Autor":"Author"})
        df["plataforma"] = "reddit"
        frames.append(df)
    if not frames:
        raise RuntimeError("No se encontraron datasets base en /data")
    df = pd.concat(frames, ignore_index=True)
    # Normalize text
    if "Comentario" in df.columns:
        df["Comentario"] = df["Comentario"].astype(str).map(normalize_text)
    # Filter out-of-context
    df = df[~df["Comentario"].map(looks_out_of_context)].copy()
    # Detect year
    if "Fecha" in df.columns:
        df["anio"] = pd.to_datetime(df["Fecha"], errors="coerce").dt.year
    else:
        df["anio"] = None
    # Empresa: respetar si viene de Google, si no, detectar por texto
    if "Empresa" not in df.columns:
        df["Empresa"] = None
    df["empresa"] = df.apply(lambda r: r["Empresa"] if isinstance(r.get("Empresa"), str) and r.get("Empresa").strip() else detect_brand(r.get("Comentario","")), axis=1)
    df["empresa"] = df["empresa"].replace({"vox":"Copaco","Vox":"Copaco"})
    return df

def main():
    OUT.mkdir(exist_ok=True, parents=True)
    nb_path = ART / "nb_tfidf_google_ratings.joblib"
    mlp_path = ART / "mlp_tfidf_google_ratings.joblib"
    if not nb_path.exists() or not mlp_path.exists():
        raise RuntimeError("No existen modelos entrenados. Ejecuta: python -m src.train_from_ratings")

    nb = joblib.load(nb_path)
    mlp = joblib.load(mlp_path)

    df = load_all_sources()
    X = df["Comentario"].fillna("")
    df["sentimiento_nb"] = nb.predict(X)
    df["sentimiento_mlp"] = mlp.predict(X)

    # Export processed with trained models
    cols = ["plataforma","Author","Fecha","Comentario","empresa","anio","sentimiento_nb","sentimiento_mlp"]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df[cols].to_excel(OUT / "dataset_anotado_con_modelos_entrenados.xlsx", index=False)

if __name__ == "__main__":
    main()

import os, pandas as pd
from pathlib import Path
import joblib
from .utils import detect_brand, normalize_text, looks_out_of_context
from .aspect_rules import detect_aspects
from .bert_utils import bert_predict_3, DEFAULT_BERT_MODEL

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
OUT  = BASE / "outputs"
ART  = BASE / "src" / "artifacts"

ANIOS_VALIDOS = set(range(2020, 2025))

def _log(msg):  # logging simple a consola y archivo
    print(msg)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "infer_log.txt", "a", encoding="utf-8") as f:
        f.write(str(msg) + "\n")

def load_all_sources():
    frames = []

    # GOOGLE MAPS
    gfile = DATA / "dataset_google.xlsx"
    if gfile.exists():
        df = pd.read_excel(gfile, sheet_name="GoogleMaps")
        df["plataforma"] = "googlemaps"
        frames.append(df)

    # TWITTER
    tfile = DATA / "dataset_twitter.xlsx"
    if tfile.exists():
        df = pd.read_excel(tfile, sheet_name="Tweets")
        df["plataforma"] = "twitter"
        frames.append(df)

    # REDDIT
    rfile = DATA / "dataset_reddit.xlsx"
    if rfile.exists():
        df = pd.read_excel(rfile, sheet_name="ExportComments.com").rename(columns={"Autor":"Author"})
        df["plataforma"] = "reddit"
        frames.append(df)

    if not frames:
        raise RuntimeError("No se encontraron datasets base en /data")

    df = pd.concat(frames, ignore_index=True)

    # Normalizar texto
    if "Comentario" not in df.columns:
        df["Comentario"] = ""
    df["Comentario"] = df["Comentario"].astype(str).map(normalize_text)

    # Filtros DUROS: no exportar NaN/blancos/ruido
    df = df[df["Comentario"].notna()]
    df = df[df["Comentario"].str.strip() != ""]
    df = df[df["Comentario"].str.len() >= 3]
    df = df[~df["Comentario"].map(looks_out_of_context)].copy()

    # Año (ajusta dayfirst si tus fechas son dd/mm/aaaa)
    if "Fecha" in df.columns:
        fechas = pd.to_datetime(df["Fecha"], errors="coerce", dayfirst=True)
        df["anio"] = fechas.dt.year
    else:
        df["anio"] = None
    df = df[df["anio"].isin(ANIOS_VALIDOS)]

    # Empresa: si viene de Google se respeta; si no, detectar (incluye @handles/hashtags y Vox->Copaco)
    if "Empresa" not in df.columns:
        df["Empresa"] = None
    df["empresa"] = df.apply(
        lambda r: r["Empresa"] if isinstance(r.get("Empresa"), str) and r.get("Empresa").strip()
                  else detect_brand(r.get("Comentario","")),
        axis=1
    )
    df["empresa"] = df["empresa"].replace({"vox":"Copaco","Vox":"Copaco"})

    # Aspectos / patrones
    df["patron_detectado"] = df["Comentario"].map(detect_aspects)

    return df

def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # Intentar cargar modelos NB/MLP (con fallback elegante)
    # Opción 1: modelos entrenados con ratings
    nb_path_r  = ART / "nb_tfidf_google_ratings.joblib"
    mlp_path_r = ART / "mlp_tfidf_google_ratings.joblib"
    # Opción 2: modelos re-entrenados con pseudo-gold de BERT (si los generaste)
    nb_path_p  = ART / "nb_tfidf_pseudogold_bert.joblib"
    mlp_path_p = ART / "mlp_tfidf_pseudogold_bert.joblib"

    _log(f"BASE={BASE}")
    _log(f"ART={ART}")
    _log(f"Buscando modelos:")
    _log(f"  ratings NB: {nb_path_r} -> {nb_path_r.exists()}")
    _log(f"  ratings MLP: {mlp_path_r} -> {mlp_path_r.exists()}")
    _log(f"  pseudo  NB: {nb_path_p} -> {nb_path_p.exists()}")
    _log(f"  pseudo  MLP: {mlp_path_p} -> {mlp_path_p.exists()}")

    nb_path = None
    mlp_path = None
    # priorizar pseudo-gold si existen (suelen alinearse mejor a BERT)
    if nb_path_p.exists() and mlp_path_p.exists():
        nb_path, mlp_path = nb_path_p, mlp_path_p
        _log("Usando modelos: pseudo-gold BERT")
    elif nb_path_r.exists() and mlp_path_r.exists():
        nb_path, mlp_path = nb_path_r, mlp_path_r
        _log("Usando modelos: entrenados con ratings")
    else:
        _log("No se encontraron modelos NB/MLP. Continuaré SOLO con BERT (fallback).")

    nb = joblib.load(nb_path) if nb_path else None
    mlp = joblib.load(mlp_path) if mlp_path else None

    # Datos
    df = load_all_sources()
    X  = df["Comentario"].fillna("")

    # Predicciones NB/MLP (opcionales)
    if nb is not None:
        try:
            df["sentimiento_nb"] = nb.predict(X)
        except Exception as e:
            _log(f"[WARN] NB falló al predecir: {e}")
            df["sentimiento_nb"] = None
    else:
        df["sentimiento_nb"] = None

    if mlp is not None:
        try:
            df["sentimiento_mlp"] = mlp.predict(X)
        except Exception as e:
            _log(f"[WARN] MLP falló al predecir: {e}")
            df["sentimiento_mlp"] = None
    else:
        df["sentimiento_mlp"] = None

    # BERT (siempre)
    try:
        df["sentimiento_bert"] = bert_predict_3(X.tolist(), model_name=DEFAULT_BERT_MODEL, batch_size=16)
    except Exception as e:
        _log(f"[WARN] BERT falló ({e}). Se rellena 'neutro'.")
        df["sentimiento_bert"] = "neutro"

    # Asegurar que no exportamos vacíos
    df = df[df["Comentario"].notna() & (df["Comentario"].str.strip()!="")]

    # Export: columnas finales
    cols = [
        "plataforma","Author","Fecha","Comentario","empresa","anio",
        "patron_detectado","sentimiento_nb","sentimiento_mlp","sentimiento_bert"
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = None

    out_file = OUT / "dataset_anotado_con_nb_mlp_bert.xlsx"
    df[cols].to_excel(out_file, index=False)
    _log(f"Exportado: {out_file}")

if __name__ == "__main__":
    main()


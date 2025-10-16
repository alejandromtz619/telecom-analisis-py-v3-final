
import pandas as pd

def map_rating_to_sentiment(x):
    try:
        v = float(x)
    except Exception:
        return None
    if v <= 2.0:
        return "negativo"
    if v >= 4.0:
        return "positivo"
    return "neutro"

def add_sentimiento_gold(df_unificado: pd.DataFrame) -> pd.DataFrame:
    df = df_unificado.copy()
    cand_cols = [c for c in df.columns if str(c).lower() in {"rating","estrellas","stars","puntuacion"}]
    if cand_cols:
        col = cand_cols[0]
        mask = df["plataforma"].eq("googlemaps")
        df.loc[mask, "sentimiento_gold"] = df.loc[mask, col].map(map_rating_to_sentiment)
    return df

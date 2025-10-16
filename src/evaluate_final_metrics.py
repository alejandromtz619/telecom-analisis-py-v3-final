import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

BASE = Path(__file__).resolve().parents[1]
OUT  = BASE / "outputs"

def safe_class_metrics(y_true, y_pred):
    """Devuelve dict con metrics y matriz, evitando errores si hay clases ausentes."""
    labels = ["negativo", "neutro", "positivo"]
    y_true = [x for x in y_true if x in labels]
    y_pred = [x if x in labels else "neutro" for x in y_pred]
    cr = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    df_cr = pd.DataFrame(cr).T
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    return df_cr, df_cm

def evaluate_and_export(df, group_field, out_writer):
    modelos = ["sentimiento_nb", "sentimiento_mlp", "sentimiento_bert"]
    grupos = sorted(df[group_field].dropna().unique())
    for g in grupos:
        sub = df[df[group_field] == g]
        if sub.empty: continue
        for modelo in modelos:
            y_true = sub["sentimiento_bert"]  # usamos BERT como referencia "gold"
            y_pred = sub[modelo]
            if len(set(y_true)) < 2: continue
            df_cr, df_cm = safe_class_metrics(y_true, y_pred)
            sheet_name = f"{group_field[:3]}_{g}_{modelo.replace('sentimiento_','')}"
            df_cr.to_excel(out_writer, sheet_name=sheet_name[:30])
            df_cm.to_excel(out_writer, sheet_name=(sheet_name[:27] + "_cm")[:31])

def main():
    df_path = OUT / "dataset_anotado_con_nb_mlp_bert.xlsx"
    if not df_path.exists():
        raise FileNotFoundError(f"No se encontró {df_path}")
    df = pd.read_excel(df_path)

    # Limpieza y filtrado
    df = df[df["Comentario"].notna() & (df["Comentario"].str.strip()!="")].copy()
    df = df[df["anio"].between(2020, 2024, inclusive="both")]

    out_xlsx = OUT / "metricas_finales_por_anio_y_plataforma.xlsx"
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        # Por año
        evaluate_and_export(df, "anio", writer)
        # Por plataforma
        evaluate_and_export(df, "plataforma", writer)

        # General (sin agrupar)
        modelos = ["sentimiento_nb", "sentimiento_mlp", "sentimiento_bert"]
        for modelo in modelos:
            y_true = df["sentimiento_bert"]  # referencia final
            y_pred = df[modelo]
            df_cr, df_cm = safe_class_metrics(y_true, y_pred)
            df_cr.to_excel(writer, sheet_name=f"general_{modelo.replace('sentimiento_','')}")
            df_cm.to_excel(writer, sheet_name=f"general_{modelo.replace('sentimiento_','')}_cm")

    print(f"✅ Métricas finales exportadas a: {out_xlsx}")

if __name__ == "__main__":
    main()

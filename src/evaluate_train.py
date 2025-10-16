
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def export_metrics(y_true, y_pred_dict, out_xlsx_path):
    with pd.ExcelWriter(out_xlsx_path, engine="openpyxl") as writer:
        for name, y_pred in y_pred_dict.items():
            cr = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            cm = confusion_matrix(y_true, y_pred, labels=["negativo","neutro","positivo"]).astype(int)
            df_cr = pd.DataFrame(cr).T
            df_cm = pd.DataFrame(cm, index=["negativo","neutro","positivo"], columns=["negativo","neutro","positivo"])
            df_cr.to_excel(writer, sheet_name=f"{name}_report")
            df_cm.to_excel(writer, sheet_name=f"{name}_confusion")

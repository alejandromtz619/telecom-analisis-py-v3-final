# ...existing code...
import os
import sys
import logging
import warnings
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from src.utils import normalize_text
from src.bert_utils import bert_predict_3

# --- Logging configuration (consola + archivo) ---
LOG_DIR = "test"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "run_verbose.log")

logger = logging.getLogger("test_models")
logger.setLevel(logging.INFO)

# Eliminar handlers previos para evitar duplicados en pytest múltiples runs
if logger.handlers:
    for h in logger.handlers[:]:
        logger.removeHandler(h)

fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
fh = logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")
fh.setFormatter(fmt)
logger.addHandler(fh)

ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(fmt)
logger.addHandler(ch)

# Reducir warnings ruidosos de librerías externas en el log
warnings.filterwarnings("default")

DATA_FILES = {
    "rating": "data/rating_google.xlsx",
    "google": "data/dataset_google.xlsx",
    "twitter": "data/dataset_twitter.xlsx",
    "reddit": "data/dataset_reddit.xlsx",
}


def _log_header(test_name: str):
    logger.info("========== START TEST: %s ==========", test_name)


def _log_footer(test_name: str):
    logger.info("=========== END TEST: %s ===========\n", test_name)


def test_load_data():
    test_name = "test_load_data"
    _log_header(test_name)
    missing = []
    for name, path in DATA_FILES.items():
        logger.info("Comprobando archivo '%s' -> %s", name, path)
        if not os.path.exists(path):
            logger.warning("FALTA: %s (se saltará la comprobación de contenido).", path)
            missing.append(path)
            continue
        try:
            df = pd.read_excel(path)
        except Exception as e:
            logger.error("ERROR al leer %s: %s", path, e)
            raise
        logger.info("Leído %s: shape=%s, columnas=%s", path, df.shape, list(df.columns)[:10])
        # Loguear una fila de ejemplo si existe
        if not df.empty:
            sample = df.iloc[0].to_dict()
            logger.info("Fila de ejemplo (primer registro): %s", {k: str(v)[:200] for k, v in sample.items()})
        assert not df.empty, f"El dataset {name} ({path}) está vacío."
    if missing:
        logger.info("Nota: algunos archivos faltaron y se imprimieron advertencias: %s", missing)
    _log_footer(test_name)


def test_normalize_text():
    test_name = "test_normalize_text"
    _log_header(test_name)
    inp = "  Este es un Texto   Con   Espacios  "
    expected = "este es un texto con espacios"
    logger.info("Entrada raw: %r", inp)
    out = normalize_text(inp)
    logger.info("Salida normalize_text: %r", out)
    logger.info("Salida esperada: %r", expected)
    assert out == expected, f"Se esperaba: '{expected}', pero se obtuvo: '{out}'"
    _log_footer(test_name)


def _is_valid_bert_label(pred):
    if isinstance(pred, (int, np.integer)):
        return int(pred) in {0, 1, 2}
    if isinstance(pred, str):
        p = pred.lower()
        return any(k in p for k in ("neg", "neut", "neu", "pos", "positivo", "negativo", "positive", "negative"))
    return False


def test_bert_inference():
    test_name = "test_bert_inference"
    _log_header(test_name)
    text = "Excelente servicio, muy conforme con la atención."
    logger.info("Texto para inferencia BERT: %r", text)
    try:
        preds = bert_predict_3([text])
    except Exception as e:
        logger.exception("ERROR: bert_predict_3 lanzó excepción")
        raise
    logger.info("bert_predict_3 retornó (raw): %s", preds)
    assert isinstance(preds, (list, tuple, np.ndarray)), "bert_predict_3 debe devolver una colección"
    assert len(preds) >= 1, "bert_predict_3 devolvió colección vacía"
    pred0 = preds[0]
    valid = _is_valid_bert_label(pred0)
    logger.info("Predicción 0: %s (válida=%s)", pred0, valid)
    assert valid, f"Predicción fuera de rango / formato inesperado: {pred0}"
    _log_footer(test_name)


def test_training_models_with_bert():
    test_name = "test_training_models_with_bert"
    _log_header(test_name)
    logger.info("Construyendo toy-set de probabilidades (3 clases) para entrenar modelos numéricos.")
    X = np.array([
        [0.9, 0.05, 0.05],  # clase 0
        [0.05, 0.9, 0.05],  # clase 1
        [0.05, 0.05, 0.9],  # clase 2
    ])
    y = np.array([0, 1, 2])
    logger.info("X:\n%s", X)
    logger.info("y: %s", y.tolist())

    # GaussianNB
    logger.info("Entrenando GaussianNB...")
    gnb = GaussianNB()
    gnb.fit(X, y)
    pred_gnb = gnb.predict(X)
    acc_gnb = accuracy_score(y, pred_gnb)
    logger.info("Predicciones GNB: %s, accuracy=%s", pred_gnb.tolist(), acc_gnb)
    assert acc_gnb == 1.0, "GaussianNB no alcanzó accuracy=1.0 en toy-set (esperado sobreajuste)"

    # MLP
    logger.info("Entrenando MLPClassifier (toy)...")
    mlp = MLPClassifier(hidden_layer_sizes=(5,), max_iter=500, random_state=0)
    mlp.fit(X, y)
    pred_mlp = mlp.predict(X)
    acc_mlp = accuracy_score(y, pred_mlp)
    logger.info("Predicciones MLP: %s, accuracy=%s", pred_mlp.tolist(), acc_mlp)
    assert acc_mlp == 1.0, "MLP no alcanzó accuracy=1.0 en toy-set (esperado sobreajuste)"
    _log_footer(test_name)


# Si se ejecuta el módulo directamente (útil para debug fuera de pytest)
if __name__ == "__main__":
    logger.info("Ejecutando test/test_models.py directamente (no pytest). Los asserts fallarán si algo está mal.")
    test_load_data()
    test_normalize_text()
    test_bert_inference()
    test_training_models_with_bert()
    logger.info("Todas las pruebas de test/test_models.py se ejecutaron (ver %s).", LOG_PATH)
# ...existing code...
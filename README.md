# tfg_sentimientos_paraguay

Análisis de sentimientos 2020–2024 sobre servicios de conectividad en Paraguay (Google Maps, Twitter/X y Reddit), comparando **BERT**, **MLP** y **Naive Bayes**. Incluye entrenamiento, inferencia sobre datasets completos y consolidación de métricas.

## Estructura del proyecto

tfg_sentimientos_paraguay/
├─ .venv/ (entorno virtual) # (ignorado por Git)
├─ requirements.txt # dependencias (transformers, torch, scikit-learn, etc.)
├─ data/
│ ├─ rating_google.xlsx # dataset etiquetado para entrenamiento (BERT/NB/MLP)
│ ├─ dataset_google.xlsx # dataset completo Google
│ ├─ dataset_twitter.xlsx # dataset completo Twitter
│ └─ dataset_reddit.xlsx # dataset completo Reddit
├─ src/
│ ├─ train_eval_with_bert.py # entrena y evalúa NB/MLP/BERT con rating_google.xlsx
│ ├─ infer_full_with_bert.py # aplica NB/MLP/BERT al dataset completo
│ ├─ evaluate_final_metrics.py # consolida y reporta métricas finales
│ ├─ retrain_nb_mlp_from_bert.py # reentrena NB/MLP usando salidas de BERT
│ ├─ bert_utils.py # utilidades (mapeo 5→3 clases, etc.)
│ ├─ aspect_rules.py # reglas de filtrado por aspecto (si se usan)
│ ├─ models_nb_mlp.py, utils.py, ...
│ └─ artifacts/
│ ├─ nb_tfidf_google_ratings.joblib
│ └─ mlp_tfidf_google_ratings.joblib
└─ outputs/
├─ metricas_entrenamiento_desde_rating_google.xlsx
├─ holdout_predicciones_desde_rating_google.xlsx
├─ dataset_anotado_con_modelos_entrenados.xlsx
├─ datasets_metricas.xlsx
├─ metricas_finales.xlsx
└─ datasets_procesados.xlsx


> **Nota:** Los archivos `.xlsx` de `data/` y `outputs/` suelen ser grandes y se ignoran por defecto en Git. Si necesitás versionarlos, quitá las reglas correspondientes del `.gitignore`.

---

## Requisitos

- **Python 3.10+** (recomendado)
- **pip** actualizado
- (Opcional) **CUDA/cuDNN** si vas a usar GPU con `torch`.

---

## Instalación

### 1) Clonar o descargar el proyecto
Si ya tenés la carpeta local, podés saltar este paso. Si no:
```bash
git clone https://github.com/<tu-usuario>/tfg_sentimientos_paraguay.git
cd tfg_sentimientos_paraguay
```

### 2) Crear y activar entorno virtual

Windows (PowerShell):

python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip wheel
pip install -r requirements.txt


Si PowerShell bloquea la activación, ejecutá PowerShell como Administrador y:

Set-ExecutionPolicy -Scope CurrentUser RemoteSigned


Linux / macOS:

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
pip install -r requirements.txt


(Opcional) PyTorch con CUDA:
Si querés GPU, instalá la build de torch según tu versión de CUDA desde la guía oficial de PyTorch. Luego volvé a pip install -r requirements.txt si hace falta.

### 3) Uso

Todos los scripts pueden ejecutarse con -h/--help para ver opciones (si están disponibles).
Por defecto, usan las rutas de data/ y escriben salidas en outputs/.

#### A) Entrenamiento + evaluación (BERT, NB y MLP)
python -m src.train_eval_with_bert


Salidas esperadas en outputs/:

metricas_entrenamiento_desde_rating_google.xlsx

holdout_predicciones_desde_rating_google.xlsx

Y artefactos/actualizaciones en src/artifacts/ (según el flujo de tu script).

#### B) Inferencia sobre datasets completos (Google/Twitter/Reddit)
python -m src.infer_full_with_bert


Salida principal:

outputs/dataset_anotado_con_modelos_entrenados.xlsx
(incluye columnas sentimiento_bert, sentimiento_mlp, sentimiento_nb)

#### C) Reentrenar NB/MLP desde salidas de BERT (opcional)
python -m src.retrain_nb_mlp_from_bert

#### D) Consolidar métricas finales y reportes
python -m src.evaluate_final_metrics


#### Salidas esperadas:

outputs/datasets_metricas.xlsx

outputs/metricas_finales.xlsx
(y/o archivos .txt con precision/recall/F1 por modelo si así está implementado)

#### E) Reglas por aspecto (opcional)

Si usás filtrado/agrupación temática:

python -m src.aspect_rules

Buenas prácticas con datos

No subas datasets crudos a GitHub si son pesados o sensibles.

Podés dejar un muestra (ej. data/sample_rating_google.xlsx) y documentación de cómo reproducir el dataset completo.

Si querés versionar binarios grandes, evaluá Git LFS.

#### Solución de problemas comunes

-m : no se reconoce en PowerShell → Estás intentando ejecutar -m pip sin python. Usá:
python -m pip install -U pip

Permisos al crear .venv → Cerrá procesos que estén usando .venv, borrá .venv/ y volvé a crear.

ImportError/MemoryError → Confirmá versiones de torch/transformers, memoria disponible y usa batch sizes más pequeños si tu script lo permite.

### Créditos / contexto

TFG FPUNE – Percepción sobre servicios de conectividad doméstica y móvil en Paraguay con lenguaje natural y aprendizaje automático (2020–2024).

Autores: Alejandro Martinez y Matias Romero (Ing. de Sistemas 10mo semestre 2025)

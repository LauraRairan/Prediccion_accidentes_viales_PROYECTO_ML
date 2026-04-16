# Predicción de Severidad de Accidentes Viales — India

**Proyecto Final — Máquina de Aprendizaje 1 | Universidad de La Sabana**

Modelos de clasificación multiclase que predicen si un accidente vial será **minor**, **major** o **fatal** a partir de condiciones del entorno, la vía y el momento del siniestro.

---

## Descripción

Este proyecto aplica la metodología **CRISP-DM** sobre el *Indian Road Accident Dataset (2022–2025)* — 20,000 registros con 24 variables. Se entrenan y comparan tres modelos supervisados: **Árbol de Decisión**, **Random Forest** y **Gradient Boosting**, con optimización de hiperparámetros mediante GridSearchCV y validación cruzada estratificada.

Los resultados se comunican a través de una aplicación interactiva en **Streamlit**.

---

## Resultados

| Modelo | Accuracy | F1 ponderado | Recomendado |
|--------|---------|-------------|-------------|
| **Árbol de Decisión** | 0.630 | **0.615** | ✅ |
| Random Forest | 0.666 | 0.608 | |
| Gradient Boosting | 0.670 | 0.605 | |

> El Árbol de Decisión es el modelo recomendado por tener el mejor F1 ponderado e interpretabilidad total mediante reglas exportables.

---

## Estructura del proyecto

```
PROYECTO_FINAL_ML/
├── app/
│   └── app.py                          ← Dashboard Streamlit
├── data/
│   └── indian_roads_dataset.csv        ← Dataset original (no incluido en el repo)
├── models/
│   ├── trained/                        ← PKLs entrenados (no incluidos en el repo)
│   ├── 01_preparacion_modelo.ipynb
│   ├── 02_arbol_decision.ipynb
│   ├── 03_random_forest.ipynb
│   └── 04_gradient_boosting.ipynb
├── notebooks/
│   ├── 01_comprension_negocio.ipynb
│   ├── 02_comprension_datos.ipynb
│   ├── 03_preparacion_datos.ipynb
│   ├── 04_modelado.ipynb
│   ├── 05_evaluacion.ipynb
│   └── 06_conclusiones.ipynb
├── reports/
│   ├── figures/                        ← Gráficas generadas por los notebooks
│   └── tables/                         ← Métricas en JSON y CSV
├── src/
│   ├── __init__.py
│   ├── data_loader.py                  ← Carga el CSV
│   ├── preprocessing.py                ← Limpieza y feature engineering
│   ├── model_loader.py                 ← Carga PKLs para el notebook 05
│   └── predict.py                      ← Carga PKLs para la app
└── requirements.txt
```

---

## Arquitectura `src/`

Cada archivo tiene una única responsabilidad:

| Archivo | Responsabilidad | Usado por |
|---------|----------------|-----------|
| `data_loader.py` | Cargar el CSV | Todos los notebooks |
| `preprocessing.py` | Imputación, feature engineering, encoding | `model_loader.py`, `predict.py` |
| `model_loader.py` | Cargar los 3 PKLs para comparación | `notebooks/05_evaluacion.ipynb` |
| `predict.py` | Cargar un PKL y predecir | `app/app.py` |

---

## Orden de ejecución

```
1. models/02_arbol_decision.ipynb         (~5 min)
2. models/03_random_forest.ipynb          (~15 min)
3. models/04_gradient_boosting.ipynb      (~20 min)
4. notebooks/05_evaluacion.ipynb
5. notebooks/06_conclusiones.ipynb
6. streamlit run app/app.py
```

> ⚠️ El notebook `05_evaluacion` carga los PKLs desde `models/trained/` — debe correrse después de los 3 notebooks de modelos.

---

## Instalación

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/proyecto-final-ml.git
cd proyecto-final-ml

# Crear entorno virtual
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux

# Instalar dependencias
pip install -r requirements.txt
```

---

## Dataset

El dataset **no está incluido en el repositorio** por su tamaño. Colócalo manualmente en:

```
data/indian_roads_dataset.csv
```

---

## Nota sobre `risk_score`

La variable `risk_score` proviene directamente del CSV del sistema de registro vial de India. Se usa el valor original porque la fórmula propia produce F1=0.42 vs F1=0.61 del CSV. El data leakage (+0.20 en 99.3% de fatales) está documentado en el Notebook 02.

En la app de Streamlit, el `risk_score` se estima con una fórmula aproximada para fines demostrativos — no reemplaza al valor real del CSV usado en el entrenamiento.

---

## Referencias

- WHO. (2023). *Global status report on road safety 2023*. https://www.who.int
- MoRTH / PIB. (2023). *Road Accidents in India – 2022*. https://www.pib.gov.in
- IBM. (s.f.). *CRISP-DM in IBM SPSS Modeler*. https://www.ibm.com/docs
- scikit-learn. (2025). *GridSearchCV, StratifiedKFold, classification_report*. https://scikit-learn.org

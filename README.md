# Predicción de Severidad de Accidentes Viales — India

**Proyecto Final — Máquina de Aprendizaje 1 | Universidad de La Sabana**
**Maria Fernanda Rodriguez Chaparro - Laura Valentina Rairan Gavilan**

Modelos de clasificación multiclase que predicen si un accidente vial será **menor**, **mayor** o **fatal** a partir de condiciones del entorno, la vía y el momento del siniestro.

---

## Descripción

En este proyecto se aplica la metodología **CRISP-DM** sobre el *Indian Road Accident Dataset (2022–2025)* — Que contiene 20,000 registros dividos con 24 variables. Tambien se entrenan y comparan tres modelos supervisados: **Árbol de Decisión**, **Random Forest** y **Gradient Boosting**, con optimización de hiperparámetros mediante GridSearchCV y validación cruzada estratificada.

Los resultados se comunican a través de una aplicación interactiva en **Streamlit**.

---
## Problema
En los escenarios de accidentes de carros donde se busca una respuesta concreta e identificar patrones, tener una clasificacion temprana de este puede ayudar a complementar una asignacion inicial de recursos en cuanto problemas viales.

Cuando esta clasificacion depende de una valoracion manual o preliminar de los hechos existe el riesgo de subestimar algunos casos o no saber dimensionar otros.

*Pregunta de negocio:*

*¿Es posible predecir si un accidente vial será menor, mayor o fatal a partir de las condiciones observadas en el entorno, la vía y el momento del siniestro. De manera que el resultado sirva como apoyo inicial para identificar una mejora vial?*

--- 

## Objetivos
### Objetivo General:
Construir modelos supervisados de clasificación multiclase que predigan la variable accident_severity (menor / mayor / fatal) a partir de variables contextuales del accidente, con el fin de aportar evidencia para análisis de seguridad vial.

## Objetivos especificos:
1. Realizar un análisis exploratorio del dataset para comprender la distribución de los datos, su calidad y las relaciones entre variables relevantes para la severidad de accidentes.

2. Preparar los datos mediante procesos de limpieza, imputación y transformación, asegurando su calidad para el modelado.

3. Aplicar feature engineering basado en condiciones relevantes de seguridad vial, como conducción nocturna, clima, visibilidad e interacciones entre variables.

4. Entrenar y comparar modelos de clasificación (Árbol de Decisión, Random Forest y Gradient Boosting) utilizando validación cruzada estratificada y ajuste de hiperparámetros.

5. Evaluar el desempeño de los modelos mediante métricas adecuadas para problemas multiclase con desbalance y comunicar los resultados a través de notebooks y una aplicación en Streamlit.

---
| Fase | Notebook | Contenido principal |
|------|----------------------|---------------------|
| **1. Business Understanding** | `01_comprension_negocio.ipynb` | Contexto, problema, objetivos y criterio de éxito |
| **2. Data Understanding** | `02_comprension_datos.ipynb` | Exploración inicial, calidad de datos y hallazgos descriptivos |
| **3. Data Preparation** | `03_preparacion_datos.ipynb` | Limpieza, imputación, codificación y preparación para modelado |
| **4. Modeling** | `04_modelado.ipynb` | Entrenamiento y ajuste de Árbol, Random Forest y Gradient Boosting |
| **5. Evaluation** | `05_evaluacion.ipynb` | Comparación de métricas, reportes de clasificación e interpretación |
| **6. Deployment / Comunicación / Conclusiones** | App en **Streamlit** + conclusiones | Visualización de resultados y cierre del proyecto |

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
1. notebooks/02_comprension_datos.ipynb
2. notebooks/03_preparacion_datos.ipynb
3. models/01_preparacion_modelo.ipynb
4. models/02_arbol_decision.ipynb          (~5 min)
5. models/03_random_forest.ipynb           (~15 min)
6. models/04_gradient_boosting.ipynb       (~20 min)
7. notebooks/04_modelado.ipynb
8. notebooks/05_evaluacion.ipynb
9. notebooks/06_conclusiones.ipynb
10. streamlit run app/app.py
```

> El notebook `05_evaluacion` carga los PKLs desde `models/trained/` — debe correrse después de los 3 notebooks de modelos.

---
## Resultados

| Modelo | Accuracy | F1 ponderado |
|--------|---------|-------------|
| **Árbol de Decisión** | 0.630 | **0.615** | 
| Random Forest | 0.666 | 0.608 |
| Gradient Boosting | 0.670 | 0.605 |

> El Árbol de Decisión es el modelo recomendado por tener el mejor F1 ponderado e interpretabilidad total mediante reglas exportables.

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
El dataset se puede descargar del siguiente link:
https://www.kaggle.com/datasets/sehaj1104/indian-road-accident-dataset-20222025
```
data/indian_roads_dataset.csv
```

---

## Referencias

- WHO. (2023). *Global status report on road safety 2023*. [https://www.who.int](https://www.who.int/teams/social-determinants-of-health/safety-and-mobility/global-status-report-on-road-safety-2023)
- MoRTH / PIB. (2023). *Road Accidents in India – 2022*. [https://www.pib.gov.in](https://www.pib.gov.in/PressReleaseIframePage.aspx?PRID=1973295&reg=3&lang=2)
- IBM. (s.f.). *CRISP-DM in IBM SPSS Modeler*. [https://www.ibm.com/docs](https://www.ibm.com/docs/en/spss-modeler/18.6.0?topic=overview-crisp-dm-in-spss-modeler)
- scikit-learn. (2025). *GridSearchCV, StratifiedKFold, classification_report*. [https://scikit-learn.org](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
- Dataset: https://www.kaggle.com/datasets/sehaj1104/indian-road-accident-dataset-20222025

import sys
import os

APP_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)
sys.path.insert(0, ROOT_DIR)

# ── Imports de la app ─────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image

# ── Importar la clase Predictor de src/predict.py ────────────────────────────
from src.predict import Predictor

TRAINED_DIR = os.path.join(ROOT_DIR, "models", "trained")
FIGURES_DIR = os.path.join(ROOT_DIR, "reports", "figures")
TABLES_DIR  = os.path.join(ROOT_DIR, "reports", "tables")

st.set_page_config(
    page_title="Prediccion de Accidentes Viales - India",
    page_icon="car",
    layout="wide"
)

# ── Verificar que los modelos existen ─────────────────────────────────────────
if not os.path.exists(os.path.join(TRAINED_DIR, "arbol.pkl")):
    st.warning("Modelos no encontrados en models/trained/.")
    st.info("Corre primero los notebooks de models/ en orden.")
    st.stop()

# ── Cargar los 3 predictores usando src/predict.py ───────────────────────────
@st.cache_resource
def cargar_predictores():
    return {
        "Arbol de Decision" : Predictor("arbol"),
        "Random Forest"     : Predictor("random_forest"),
        "Gradient Boosting" : Predictor("gradient_boosting"),
    }

predictores = cargar_predictores()

# =============================================================================
# SIDEBAR — Inputs del usuario
# =============================================================================
st.sidebar.header("Condiciones del accidente")

st.sidebar.subheader("Tiempo")
hora       = st.sidebar.slider("Hora del dia (0-23)", 0, 23, 14)
dia_semana = st.sidebar.selectbox("Dia de la semana",
    ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
es_finde  = st.sidebar.checkbox("Es fin de semana?",
                                 value=(dia_semana in ["Saturday","Sunday"]))
hora_pico = st.sidebar.checkbox("Es hora pico? (7-9h o 17-19h)",
                                 value=(7<=hora<=9 or 17<=hora<=19))

st.sidebar.subheader("Condiciones de la via")
tipo_via = st.sidebar.selectbox("Tipo de via", ["highway","urban","rural"])
carriles = st.sidebar.slider("Numero de carriles", 1, 6, 2)
semaforo = st.sidebar.checkbox("Hay semaforo?", value=True)
densidad = st.sidebar.selectbox("Densidad de trafico", ["low","medium","high"])

st.sidebar.subheader("Condiciones climaticas")
clima       = st.sidebar.selectbox("Clima", ["clear","fog","rain"])
visibilidad = st.sidebar.selectbox("Visibilidad", ["high","medium","low"])
temperatura = st.sidebar.slider("Temperatura (C)", 5, 45, 28)

st.sidebar.subheader("Causa y contexto")
causa     = st.sidebar.selectbox("Causa del accidente",
    ["distraction","drunk driving","overspeeding","poor road","weather"])
vehiculos = st.sidebar.slider("Vehiculos involucrados", 1, 10, 2)

st.sidebar.subheader("Modelo")
modelo_tipo = st.sidebar.radio("Modelo a usar",
    ["Arbol de Decision","Random Forest","Gradient Boosting"], index=0)

# =============================================================================
# ESTIMACIÓN DEL RISK_SCORE para la interfaz
# =============================================================================
def estimar_risk_score(hora, clima, visibilidad, densidad, causa, tipo_via, vehiculos):  
    score = (0.10
             + int(clima in ["fog","rain"])                        * 0.15
             + int(visibilidad=="low")                             * 0.20
             + int(densidad=="high")                               * 0.20
             + int(7<=hora<=9 or 17<=hora<=19)                    * 0.15
             + int(hora>=20 or hora<=5)*int(clima in ["fog","rain"])* 0.10
             + int(causa in ["drunk driving","overspeeding"])      * 0.10
             + int(tipo_via=="highway")                            * 0.05
             + ((vehiculos-1)/9)                                   * 0.05)
    return round(min(score, 1.0), 2)

# =============================================================================
# CONSTRUCCIÓN EL DATAFRAME DE INPUT
# =============================================================================
def construir_dataframe_input(hora, dia_semana, es_finde, hora_pico, tipo_via,
                               carriles, semaforo, densidad, clima, visibilidad,
                               temperatura, causa, vehiculos, rs_estimado):
    return pd.DataFrame([{
        "hour"            : hora,
        "is_weekend"      : int(es_finde),
        "is_peak_hour"    : int(hora_pico),
        "lanes"           : carriles,
        "temperature"     : temperatura,
        "traffic_signal"  : int(semaforo),
        "risk_score"      : rs_estimado,     
        "road_type"       : tipo_via,
        "weather"         : clima,
        "visibility"      : visibilidad,
        "traffic_density" : densidad,
        "cause"           : causa,
        "day_of_week"     : dia_semana,
        "vehicles_involved": vehiculos,
        "casualties"      : 1,                
        "accident_severity": "minor",         
    }])

# =============================================================================
# LAYOUT PRINCIPAL
# =============================================================================
st.title("Prediccion de Severidad de Accidentes Viales — India")
st.markdown("""
Predice **Menor**, **Mayor** o **Fatal** a partir de condiciones del entorno.
El `risk_score` mostrado es una estimación aproximada para la interfaz —
el modelo fue entrenado con el valor original del dataset (sistema de registro vial de India).
""")
st.divider()

tab1, tab2, tab3 = st.tabs(["Prediccion", "Metricas del modelo", "Visualizaciones"])

explicaciones_graficas = {
    "Distribucion del target": {
        "que_muestra": """
Esta grafica deja ver como esta repartida la severidad de los accidentes en el dataset. 
Se observa que la clase **minor** aparece con mucha mas frecuencia, mientras que **major** y sobre todo **fatal** tienen menos casos.
""",
        "interpretacion": """
En palabras sencillas, esto significa que el modelo aprende con mas ejemplos de accidentes leves que de accidentes graves. 
Por eso no basta con mirar solo el accuracy; tambien toca revisar metricas como **F1** y **recall**, porque nos interesa saber que tan bien reconoce los casos realmente importantes.
"""
    },
    "Variables categoricas vs severidad": {
        "que_muestra": """
Aqui se comparan variables como tipo de via, clima, visibilidad, densidad del trafico, causa y dia de la semana frente a la severidad del accidente.
""",
        "interpretacion": """
Lo interesante es que ninguna categoria, por si sola, separa totalmente las clases. 
Aun asi, si se alcanzan a notar pequeñas diferencias: por ejemplo, en condiciones mas riesgosas la severidad tiende a subir. 
Eso sugiere que el modelo no decide por una sola variable, sino por la combinacion de varias al mismo tiempo.
"""
    },
    "Distribucion numericas": {
        "que_muestra": """
Esta grafica muestra como se distribuyen las variables numericas, como la hora, la temperatura, los carriles, los vehiculos involucrados, los lesionados y el `risk_score`.
""",
        "interpretacion": """
Sirve para entender si los datos estan balanceados o si hay concentraciones raras. 
Por ejemplo, variables como `casualties` y `risk_score` ayudan bastante porque muestran mejor diferencia entre accidentes leves y accidentes fatales.
"""
    },
    "Boxplots por clase": {
        "que_muestra": """
Los boxplots comparan las variables numericas entre las tres clases: **minor**, **major** y **fatal**.
""",
        "interpretacion": """
Aqui se ve mas claro en que variables si hay diferencia real entre clases. 
En varias variables el cambio no es tan grande, pero en `casualties` y sobre todo en `risk_score` la separacion es mucho mas evidente. 
Eso confirma que esas variables pesan bastante en la prediccion.
"""
    },
    "Mapa de correlacion": {
        "que_muestra": """
Este mapa enseña la relacion lineal entre las variables numericas usando correlacion de Pearson.
""",
        "interpretacion": """
La mayoria de relaciones son bajas, o sea que las variables no estan diciendo exactamente lo mismo. 
La relacion mas visible es entre `vehicles_involved` y `casualties`, lo cual tiene sentido: entre mas vehiculos participan, suele haber mas lesionados. 
Esto es bueno porque evita demasiada redundancia en el modelo.
"""
    },
    "Mapa de nulos": {
        "que_muestra": """
Esta visualizacion permite revisar si hay valores faltantes dentro del dataset.
""",
        "interpretacion": """
En este caso casi no se observan nulos, lo cual es una ventaja porque no hizo falta imputar datos ni eliminar muchas filas. 
Eso ayuda a que el analisis sea mas limpio y que el entrenamiento del modelo sea mas estable.
"""
    },
    "Variables engineered vs target": {
        "que_muestra": """
Aqui aparecen variables creadas a partir de otras, como `night_drive`, `rush_hour`, `bad_weather`, `low_vis` o `high_traffic`, comparadas con la severidad.
""",
        "interpretacion": """
Estas variables sintetizan mejor situaciones de riesgo reales. 
Por ejemplo, manejar de noche, con mala visibilidad o con trafico alto tiene mas sentido cuando se evalua en conjunto que por separado. 
Por eso esta parte ayuda mucho a enriquecer el modelo.
"""
    },
    "risk_score por clase": {
        "que_muestra": """
Esta grafica compara el comportamiento del `risk_score` entre las clases **minor**, **major** y **fatal**.
""",
        "interpretacion": """
Aqui se nota una separacion bastante clara: los accidentes fatales tienden a tener puntajes mas altos. 
Eso explica por que `risk_score` termina siendo tan importante en los modelos. 
Tambien es una señal de que esta variable esta muy ligada al objetivo, asi que toca mencionarla con cuidado al sustentar.
"""
    },
    "Accidentes por hora": {
        "que_muestra": """
Se muestra como cambia la cantidad de accidentes segun la hora del dia y la severidad.
""",
        "interpretacion": """
Esto ayuda a ver en que momentos del dia se concentran mas eventos. 
Se alcanzan a notar aumentos en ciertas franjas, como horas de mayor movimiento o momentos nocturnos. 
No significa que la hora por si sola determine la severidad, pero si aporta contexto importante.
"""
    },
    "Comparacion de metricas": {
        "que_muestra": """
Aqui se comparan metricas globales como **accuracy**, **F1 ponderado** y **recall ponderado** entre los modelos.
""",
        "interpretacion": """
La idea de esta grafica es ver cual modelo logra un mejor equilibrio general. 
Si las diferencias son pequeñas, conviene no quedarse solo con el numero mas alto, sino tambien pensar en cual modelo es mas facil de explicar. 
Por eso el arbol suele ser una muy buena opcion cuando el rendimiento es parecido.
"""
    },
    "Matrices de confusion": {
        "que_muestra": """
Las matrices de confusion muestran en que clases acierta cada modelo y en cuales se equivoca.
""",
        "interpretacion": """
Esta es de las graficas mas utiles porque permite ver el error real. 
Normalmente la clase **major** es la que mas se confunde con **minor**, mientras que **fatal** suele quedar mejor identificada. 
Eso ayuda a explicar que el problema mas dificil no siempre es detectar los extremos, sino la clase intermedia.
"""
    },
    "Importancia features (Arbol y RF)": {
        "que_muestra": """
Aqui se ve cuales variables fueron las mas importantes para el **Arbol de Decision** y el **Random Forest**.
""",
        "interpretacion": """
En ambos casos resalta muchisimo `risk_score`, y despues aparecen variables como visibilidad, trafico, temperatura u hora. 
En otras palabras, el modelo se apoya bastante en una mezcla de condiciones del entorno. 
Esta grafica sirve mucho para justificar por que ciertas variables tienen mas peso en la decision final.
"""
    },
    "Importancia features (GB)": {
        "que_muestra": """
Esta grafica muestra las variables que mas peso tuvieron dentro del modelo **Gradient Boosting**.
""",
        "interpretacion": """
De nuevo aparece `risk_score` como la variable dominante, seguida por temperatura y combinaciones como `vis_trafico`. 
Eso indica que Gradient Boosting tambien esta captando patrones complejos, no solo relaciones directas. 
Es una buena forma de mostrar que el modelo aprende interacciones entre factores.
"""
    },
    "Arbol de decision": {
        "que_muestra": """
Aqui se visualizan las primeras reglas que sigue el arbol para separar las clases.
""",
        "interpretacion": """
Esta grafica es muy valiosa porque vuelve el modelo entendible. 
Uno puede leer las ramas como si fueran reglas: si el riesgo supera cierto punto, o si hay baja visibilidad y ciertas condiciones, la prediccion cambia. 
Eso hace que el arbol sea mucho mas facil de defender que otros modelos mas cerrados.
"""
    },
    "Curvas ROC": {
        "que_muestra": """
Las curvas ROC comparan la capacidad de cada modelo para distinguir una clase frente a las otras.
""",
        "interpretacion": """
Entre mas cerca este la curva de la esquina superior izquierda, mejor separa esa clase. 
Si una clase tiene AUC muy alto, significa que el modelo la distingue bastante bien. 
Esta grafica sirve para mostrar que no todas las clases tienen el mismo nivel de dificultad.
"""
    },
    "Curvas de aprendizaje": {
        "que_muestra": """
Estas curvas comparan el rendimiento en entrenamiento y validacion a medida que aumenta el tamaño de los datos.
""",
        "interpretacion": """
Sirven para revisar si el modelo esta aprendiendo bien o si se esta sobreajustando. 
Si la curva de entrenamiento queda muy arriba y la de validacion muy abajo, hay overfitting. 
Si ambas se mantienen cercanas, el modelo generaliza mejor.
"""
    },
    "Efecto del umbral": {
        "que_muestra": """
Esta grafica enseña como cambian precision, recall y F1 cuando se mueve el umbral de decision para la clase `fatal`.
""",
        "interpretacion": """
Es util porque deja ver el equilibrio entre detectar mas casos graves y evitar falsas alarmas. 
Un umbral muy bajo captura mas accidentes fatales, pero tambien puede equivocarse mas. 
Uno muy alto hace lo contrario. 
Por eso esta grafica ayuda a justificar cual umbral conviene usar segun el objetivo del sistema.
"""
    }
}

# =============================================================================
# TAB 1 — PREDICCION
# =============================================================================
with tab1:
    col_left, col_right = st.columns([1, 1])

    # Estimar risk_score para mostrar al usuario
    rs = estimar_risk_score(hora, clima, visibilidad, densidad, causa, tipo_via, vehiculos)

    # Construir DataFrame de input — el Predictor hará el preprocessing
    df_input = construir_dataframe_input(
        hora, dia_semana, es_finde, hora_pico, tipo_via, carriles,
        semaforo, densidad, clima, visibilidad, temperatura, causa, vehiculos, rs)

    # ── Predecir usando src/predict.py ───────────────────────────────────────
    predictor      = predictores[modelo_tipo]
    prediccion     = predictor.predecir(df_input)[0]
    probabilidades = predictor.predecir_proba(df_input)[0]
    clases         = predictor.modelo.classes_
    prob_fatal     = probabilidades[list(clases).index("fatal")]

    # ── Columna izquierda ─────────────────────────────────────────────────────
    with col_left:
        st.subheader("Condiciones ingresadas")

        if prediccion == "fatal":
            st.error(f"Nivel de riesgo MUY ALTO — Prediccion: FATAL | risk_score estimado: {rs:.2f}")
        elif prediccion == "Mayor":
            st.warning(f"Nivel de riesgo ALTO — Prediccion: MAYOR | risk_score estimado: {rs:.2f}")
        else:
            st.success(f"Nivel de riesgo BAJO — Prediccion: MENOR | risk_score estimado: {rs:.2f}")

        resumen = pd.DataFrame({
            "Variable": ["Hora","Dia","Tipo de via","Clima","Visibilidad",
                         "Densidad trafico","Causa","Temperatura",
                         "Vehiculos involucrados","risk_score (estimado)"],
            "Valor"   : [f"{hora}:00", dia_semana, tipo_via, clima, visibilidad,
                         densidad, causa, f"{temperatura}C", vehiculos, f"{rs:.2f}"]
        })
        st.dataframe(resumen, hide_index=True, use_container_width=True)

        st.markdown("**Variables derivadas:**")
        col_a, col_b = st.columns(2)
        col_a.metric("Conduccion nocturna", "Si" if (hora>=20 or hora<=5) else "No")
        col_a.metric("Hora pico",            "Si" if hora_pico else "No")
        col_a.metric("Mal clima",             "Si" if clima in ["fog","rain"] else "No")
        col_b.metric("Visibilidad baja",      "Si" if visibilidad=="low" else "No")
        col_b.metric("Trafico alto",           "Si" if densidad=="high" else "No")
        col_b.metric("Causa alto riesgo",
                     "Si" if causa in ["drunk driving","overspeeding"] else "No")

    # ── Columna derecha ───────────────────────────────────────────────────────
    with col_right:
        st.subheader("Resultado de la prediccion")

        if prediccion == "fatal":
            st.error("### Severidad predicha: FATAL")
        elif prediccion == "major":
            st.warning("### Severidad predicha: MAYOR")
        else:
            st.success("### Severidad predicha: MENOR")

        st.markdown(f"**Modelo:** {modelo_tipo} | **risk_score estimado:** `{rs:.2f}`")
        st.divider()

        st.markdown("**Probabilidades por clase:**")
        for clase, prob in sorted(zip(clases, probabilidades), key=lambda x: -x[1]):
            cn, cb = st.columns([1, 3])
            cn.markdown(f"`{clase}`")
            cb.progress(float(prob), text=f"{prob*100:.1f}%")

        if prob_fatal >= 0.60:
            st.error("Alta probabilidad de accidente fatal. Respuesta de emergencia prioritaria.")
        elif prob_fatal >= 0.35:
            st.warning("Probabilidad moderada de fatalidad. Monitorear de cerca.")

# =============================================================================
# TAB 2 — METRICAS
# =============================================================================
with tab2:
    st.subheader("Metricas de evaluacion")
    ruta_metricas = os.path.join(TABLES_DIR, "metricas_finales.csv")
    if os.path.exists(ruta_metricas):
        df_m = pd.read_csv(ruta_metricas)
        st.markdown("#### Modelo recomendado — Arbol de Decision")
        with st.container(border=True):
            col_just, col_kpi = st.columns([2, 1])
            with col_just:
                st.markdown("""
**Por que el Arbol de Decision?**

| Criterio | Arbol | Random Forest | Gradient Boosting |
|----------|-------|---------------|-------------------|
| **F1 ponderado** | Mejor | Similar | Similar |
| Interpretabilidad | Reglas visibles | Caja negra | Caja negra |
| Recall clase `fatal` | Mejor balance | Similar | Similar |
| Escalado requerido | No | No | No |
| Reglas exportables | Si | No | No |
                """)
            with col_kpi:
                arbol_row = df_m[df_m["Modelo"].str.contains("rbol", case=False)]
                if not arbol_row.empty:
                    fila = arbol_row.iloc[0]
                    st.metric("Accuracy",     f"{fila['Accuracy']:.3f}")
                    st.metric("F1 ponderado", f"{fila['F1 ponderado']:.3f}")
                    st.metric("Recall",       f"{fila['Recall ponderado']:.3f}")
        st.divider()
        st.dataframe(df_m, hide_index=True, use_container_width=True)
    else:
        st.warning("No se encontro metricas_finales.csv.")
        st.info("Corre el notebook 05_evaluacion.ipynb primero.")

# =============================================================================
# TAB 3 — VISUALIZACIONES
# =============================================================================
with tab3:
    st.subheader("Visualizaciones del modelo")
    imagenes = {
        "Distribucion del target"            : "distribucion_target.png",
        "Variables categoricas vs severidad" : "categoricas_vs_target.png",
        "Distribucion de variables numericas": "distribucion_numericas.png",
        "Boxplots por clase"                 : "boxplots_numericas.png",
        "Mapa de correlacion"                : "correlacion.png",
        "Mapa de valores nulos"              : "mapa_nulos.png",
        "Variables engineered vs target"     : "engineered_vs_target.png",
        "risk_score por clase"               : "boxplot_risk_score.png",
        "Accidentes por hora"                : "accidentes_por_hora.png",
        "Comparacion de metricas"            : "comparacion_metricas.png",
        "Matrices de confusion"              : "matrices_confusion.png",
        "Importancia features (Arbol y RF)"  : "feature_importance.png",
        "Importancia features (GB)"          : "feature_importance_gb.png",
        "Arbol de decision"                  : "arbol_decision.png",
        "Curvas ROC"                         : "curvas_roc.png",
        "Curvas de aprendizaje"              : "curvas_aprendizaje.png",
        "Efecto del umbral"                  : "efecto_umbral.png",
    }  

    for titulo, archivo in imagenes.items():
        ruta = os.path.join(FIGURES_DIR, archivo)
        if os.path.exists(ruta):
            st.markdown(f"#### {titulo}")
            st.image(Image.open(ruta), use_container_width=True)

            if titulo in explicaciones_graficas:
                with st.expander("Ver explicacion de esta grafica"):
                    st.markdown("**Que muestra**")
                    st.markdown(explicaciones_graficas[titulo]["que_muestra"].strip())
                    st.markdown("**Interpretacion**")
                    st.markdown(explicaciones_graficas[titulo]["interpretacion"].strip())
            st.divider()
        else:
            st.info(f"{archivo} — corre los notebooks primero.")



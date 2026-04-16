import joblib
import os
import pandas as pd
import numpy as np

from src.preprocessing import ProcesadorDatos

_SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR    = os.path.dirname(_SRC_DIR)
_TRAINED_DIR = os.path.join(_ROOT_DIR, "models", "trained")


class Predictor:    

    MODELOS = {
        "arbol"             : "arbol.pkl",
        "random_forest"     : "random_forest.pkl",
        "gradient_boosting" : "gradient_boosting.pkl",
    }

    def __init__(self, modelo_tipo: str = "arbol"):
        assert modelo_tipo in self.MODELOS, \
            f"modelo_tipo debe ser uno de: {list(self.MODELOS)}"
        self.modelo_tipo = modelo_tipo
        self.procesador  = ProcesadorDatos()
        self.modelo      = self._cargar(self.MODELOS[modelo_tipo])

    def _cargar(self, nombre):
        ruta = os.path.join(_TRAINED_DIR, nombre)
        return joblib.load(ruta)

    def predecir(self, df: pd.DataFrame) -> np.ndarray:
        X, _ = self.procesador.preparar_datos(df)
        cols  = self.modelo.feature_names_in_ \
                if hasattr(self.modelo, "feature_names_in_") else X.columns
        return self.modelo.predict(X.reindex(columns=cols, fill_value=0))

    def predecir_proba(self, df: pd.DataFrame) -> np.ndarray:
        X, _ = self.procesador.preparar_datos(df)
        cols  = self.modelo.feature_names_in_ \
                if hasattr(self.modelo, "feature_names_in_") else X.columns
        return self.modelo.predict_proba(X.reindex(columns=cols, fill_value=0))

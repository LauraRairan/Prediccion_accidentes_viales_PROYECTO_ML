import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, classification_report)

from src.preprocessing import ProcesadorDatos

_SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR    = os.path.dirname(_SRC_DIR)
_TRAINED_DIR = os.path.join(_ROOT_DIR, "models", "trained")


class ModeloAccidentes:   

    SEED      = 42
    TEST_SIZE = 0.20

    PKLS = {
        "arbol"             : "arbol.pkl",
        "random_forest"     : "random_forest.pkl",
        "gradient_boosting" : "gradient_boosting.pkl",
    }

    NOMBRES = {
        "arbol"             : "Arbol de Decision",
        "random_forest"     : "Random Forest",
        "gradient_boosting" : "Gradient Boosting",
    }

    def __init__(self):
        self.procesador = ProcesadorDatos()

    def _cargar_modelo(self, nombre_pkl: str):
        ruta = os.path.join(_TRAINED_DIR, nombre_pkl)
        if not os.path.exists(ruta):
            raise FileNotFoundError(
                f"No se encontró {ruta}. "
                f"Corre primero el notebook correspondiente en models/.")
        return joblib.load(ruta)

    def calcular_metricas(self, y_test, y_pred, nombre: str) -> dict:
        return {
            "modelo"               : nombre,
            "accuracy"             : round(accuracy_score(y_test, y_pred), 4),
            "f1_weighted"          : round(f1_score(y_test, y_pred, average="weighted"), 4),
            "recall_weighted"      : round(recall_score(y_test, y_pred, average="weighted"), 4),
            "precision_weighted"   : round(precision_score(y_test, y_pred, average="weighted"), 4),
            "reporte_clasificacion": classification_report(y_test, y_pred, output_dict=True),
        }

    def cargar_resultados(self, df: pd.DataFrame) -> dict:
        
        # Preparar datos y dividir igual que en el entrenamiento
        X, y = self.procesador.preparar_datos(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.TEST_SIZE,
            random_state=self.SEED, stratify=y)

        resultado = {}
        for key, nombre_pkl in self.PKLS.items():
            modelo   = self._cargar_modelo(nombre_pkl)
            y_pred   = modelo.predict(X_test)
            resultado[key] = {
                "modelo"  : modelo,
                "metricas": self.calcular_metricas(y_test, y_pred, self.NOMBRES[key]),
                "y_pred"  : y_pred,
                "y_proba" : modelo.predict_proba(X_test),
                "clases"  : modelo.classes_,
                "X_train" : X_train, "X_test": X_test,
                "y_train" : y_train, "y_test": y_test,
            }

        return resultado
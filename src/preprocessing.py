import pandas as pd
import numpy as np


class ProcesadorDatos:   

    CATEGORICAS = [
        "road_type", "weather", "visibility",
        "traffic_density", "cause", "day_of_week"
    ]

    FEATURES = [
        "hour", "is_weekend", "is_peak_hour", "lanes",
        "temperature", "traffic_signal", "risk_score",
        "night_drive", "rush_hour", "bad_weather",
        "low_vis", "high_traffic", "high_risk_cause", "is_highway",
        "hora_sin", "hora_cos",
        "noche_lluvia", "vis_trafico", "pico_trafico",
        "road_type", "weather", "visibility",
        "traffic_density", "cause", "day_of_week"
    ]

    TARGET = "accident_severity"

    def __init__(self):
        pass

    # -------------------------------------------------------------------------
    # IMPUTACION
    # -------------------------------------------------------------------------

    def imputar_nulos(self, df: pd.DataFrame) -> pd.DataFrame:
        """Imputa valores nulos. festival (99.4% nulos) → 'none'."""
        df = df.copy()
        if "festival" in df.columns:
            df["festival"] = df["festival"].fillna("none")
        return df

    # -------------------------------------------------------------------------
    # FEATURE ENGINEERING
    # -------------------------------------------------------------------------

    def crear_features(self, df: pd.DataFrame) -> pd.DataFrame:
        
        df = df.copy()

        df["night_drive"]     = ((df["hour"] >= 20) | (df["hour"] <= 5)).astype(int)
        df["rush_hour"]       = df["hour"].apply(
            lambda h: 1 if (7 <= h <= 9 or 17 <= h <= 19) else 0)
        df["bad_weather"]     = df["weather"].isin(["fog", "rain"]).astype(int)
        df["low_vis"]         = (df["visibility"] == "low").astype(int)
        df["high_traffic"]    = (df["traffic_density"] == "high").astype(int)
        df["high_risk_cause"] = df["cause"].isin(
            ["drunk driving", "overspeeding"]).astype(int)
        df["is_highway"]      = (df["road_type"] == "highway").astype(int)

        # Interacciones
        df["hora_sin"]     = np.sin(2 * np.pi * df["hour"] / 24)
        df["hora_cos"]     = np.cos(2 * np.pi * df["hour"] / 24)
        df["noche_lluvia"] = df["night_drive"] * df["bad_weather"]
        df["vis_trafico"]  = df["low_vis"]     * df["high_traffic"]
        df["pico_trafico"] = df["rush_hour"]   * df["high_traffic"]

        # risk_score: usar el del CSV si existe; calcular como fallback si no
        if "risk_score" not in df.columns:
            df["risk_score"] = (
                0.10
                + df["bad_weather"]  * 0.15
                + df["low_vis"]      * 0.20
                + df["high_traffic"] * 0.20
                + df["is_peak_hour"] * 0.15
            ).round(2)

        return df

    # -------------------------------------------------------------------------
    # CODIFICACION
    # -------------------------------------------------------------------------

    def codificar(self, df: pd.DataFrame, features: list) -> pd.DataFrame:
        X = df[features].copy()
        cats = [c for c in self.CATEGORICAS if c in X.columns]
        X = pd.get_dummies(X, columns=cats, drop_first=False)
        return X  

    # -------------------------------------------------------------------------
    # PIPELINE COMPLETO
    # -------------------------------------------------------------------------

    def preparar_datos(self, df: pd.DataFrame):
        
        df = self.imputar_nulos(df)
        df = self.crear_features(df)
        feats = [f for f in self.FEATURES if f in df.columns]
        X = self.codificar(df, feats)
        y = df[self.TARGET]
        return X, y

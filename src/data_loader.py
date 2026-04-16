import pandas as pd


class DataLoader:
    """Clase encargada de cargar los datos en formato DataFrame."""

    def __init__(self, ruta: str):
        self.ruta = ruta

    def cargar_datos(self) -> pd.DataFrame:
        return pd.read_csv(self.ruta)
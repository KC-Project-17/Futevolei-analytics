"""
Utilidades comunes a todo el paquete: metadatos del video y tramos de arrays booleanos.
"""
import numpy as np
import pandas as pd


def meta_to_dict(meta):
    """
    Normaliza los metadatos del video a un dict.

    Args:
        meta (dict | pd.Series | pd.DataFrame): metadatos (DataFrame de una fila, Series o dict).

    Returns:
        dict: {'fps': float, 'width': int, 'height': int, 'n_frames': int}.
    """
    if isinstance(meta, pd.DataFrame):
        meta = meta.iloc[0].to_dict()
    elif isinstance(meta, pd.Series):
        meta = meta.to_dict()
    return {"fps": float(meta["fps"]), "width": int(meta["width"]),
            "height": int(meta["height"]), "n_frames": int(meta["n_frames"])}


def find_runs(valid):
    """
    Tramos True consecutivos de un array booleano.

    Args:
        valid (np.ndarray): array booleano.

    Returns:
        list: [(inicio, fin_exclusivo)] de cada tramo.
    """
    v = np.concatenate([[0], valid.astype(int), [0]])
    d = np.diff(v)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))

from __future__ import annotations
import pandas as pd
import numpy as np

def read_idomain_csv(path: str, nrow: int, ncol: int) -> np.ndarray:
    arr = pd.read_csv(path, header=None).values
    if arr.shape != (nrow, ncol):
        raise ValueError(f"idomain shape {arr.shape} != ({nrow},{ncol})")
    return arr.astype(int)

def read_boundary_csv(path: str | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    return pd.read_csv(path)

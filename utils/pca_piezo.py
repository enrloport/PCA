# pca_piezo.py
# Implementación de PCA para señales de un sensor piezoeléctrico

import numpy as np
from sklearn.decomposition import PCA

def apply_pca_piezo(data, n_components=2):
    """
    Aplica PCA a datos de un sensor piezoeléctrico.
    Args:
        data (np.ndarray): Matriz de datos piezo (muestras x características).
        n_components (int): Número de componentes principales.
    Returns:
        np.ndarray: Datos transformados por PCA.
        PCA: Objeto PCA ajustado.
    """
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(data)
    return transformed, pca

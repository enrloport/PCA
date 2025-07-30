# pca_imu.py
# Implementación de PCA para señales de una IMU

import numpy as np
from sklearn.decomposition import PCA

def apply_pca_imu(data, n_components=2):
    """
    Aplica PCA a datos de una IMU.
    Args:
        data (np.ndarray): Matriz de datos IMU (muestras x características).
        n_components (int): Número de componentes principales.
    Returns:
        np.ndarray: Datos transformados por PCA.
        PCA: Objeto PCA ajustado.
    """
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(data)
    return transformed, pca

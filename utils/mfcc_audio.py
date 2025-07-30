# mfcc_audio.py
# Implementación de extracción de MFCC para archivos de audio

import numpy as np
import librosa

def extract_mfcc(audio_path, n_mfcc=13, sr=22050):
    """
    Extrae coeficientes MFCC de un archivo de audio.
    Args:
        audio_path (str): Ruta al archivo de audio.
        n_mfcc (int): Número de coeficientes MFCC a extraer.
        sr (int): Frecuencia de muestreo para cargar el audio.
    Returns:
        np.ndarray: Matriz de coeficientes MFCC.
    """
    y, sr = librosa.load(audio_path, sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfccs

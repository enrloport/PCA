import numpy as np
import pytest
from utils.mfcc_audio import extract_mfcc
from utils.pca_imu import apply_pca_imu
from utils.pca_piezo import apply_pca_piezo

# Suprime warnings de deprecación globalmente en los tests
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

def test_extract_mfcc(tmp_path):
    # Crea un archivo de audio sintético
    import soundfile as sf
    sr = 22050
    duration = 1.0  # 1 segundo
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 220 * t)
    audio_path = tmp_path / "test.wav"
    sf.write(audio_path, y, sr)
    print("\n" + "="*40)
    print("[MFCC AUDIO]")
    print(f"Señal de audio original (primeros 10):\n{y[:10]}")
    mfccs = extract_mfcc(str(audio_path))
    print(f"MFCCs (forma {mfccs.shape}, primeros 10 frames):\n{mfccs[:, :10]}")
    print("="*40 + "\n")
    assert mfccs.shape[0] == 13  # 13 MFCCs por defecto
    assert mfccs.shape[1] > 0

def test_apply_pca_imu():
    data = np.random.rand(100, 6)  # 100 muestras, 6 ejes IMU
    print("\n" + "="*40)
    print("[PCA IMU]")
    print(f"Señal IMU original (primeros 2 muestras):\n{data[:2]}")
    transformed, pca = apply_pca_imu(data, n_components=3)
    print(f"PCA IMU (primeros 2 muestras):\n{transformed[:2]}")
    print("="*40 + "\n")
    assert transformed.shape == (100, 3)
    assert hasattr(pca, 'components_')

def test_apply_pca_piezo():
    data = np.random.rand(50, 10)  # 50 muestras, 10 sensores piezo
    print("\n" + "="*40)
    print("[PCA PIEZO]")
    print(f"Señal piezo original (primeros 2 muestras):\n{data[:2]}")
    transformed, pca = apply_pca_piezo(data, n_components=2)
    print(f"PCA piezo (primeros 2 muestras):\n{transformed[:2]}")
    print("="*40 + "\n")
    assert transformed.shape == (50, 2)
    assert hasattr(pca, 'components_')

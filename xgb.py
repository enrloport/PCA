
import numpy as np
from utils.mfcc_audio import extract_mfcc
from utils.pca_imu import apply_pca_imu
from utils.pca_piezo import apply_pca_piezo
import xgboost as xgb

# Generar señal de audio sintética
def generar_audio(sr=22050, duration=1.0):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 220 * t)
    return y, sr

# Generar señal IMU sintética
def generar_imu():
    return np.random.rand(100, 6)

# Generar señal piezo sintética
def generar_piezo():
    return np.random.rand(50, 10)

if __name__ == "__main__":
    # Audio
    y, sr = generar_audio()
    print("\n" + "="*40)
    print("[SEÑAL AUDIO ORIGINAL] (primeros 20 samples):\n", y[:20])
    import librosa
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    print("[EMBEDDING MFCC] (forma {}):\n".format(mfccs.shape), mfccs[:, :10])
    mfccs_flat = mfccs.flatten()[:100]

    # IMU
    imu = generar_imu()
    print("\n" + "="*40)
    print("[SEÑAL IMU ORIGINAL] (primeras 2 muestras):\n", imu[:2])
    imu_pca, _ = apply_pca_imu(imu, n_components=5)
    print("[EMBEDDING PCA IMU] (primeras 2 muestras):\n", imu_pca[:2])
    imu_flat = imu_pca.flatten()[:100]

    # Piezo
    piezo = generar_piezo()
    print("\n" + "="*40)
    print("[SEÑAL PIEZO ORIGINAL] (primeras 2 muestras):\n", piezo[:2])
    piezo_pca, _ = apply_pca_piezo(piezo, n_components=5)
    print("[EMBEDDING PCA PIEZO] (primeras 2 muestras):\n", piezo_pca[:2])
    piezo_flat = piezo_pca.flatten()[:100]

    # Concatenar todos los embeddings
    X = np.stack([mfccs_flat, imu_flat, piezo_flat], axis=0)
    y_labels = np.array([0, 1, 2])  # etiquetas dummy

    print("\n" + "="*40)
    print("[EMBEDDINGS FINALES PARA XGBOOST] (cada fila es un embedding):\n", X)
    print("Etiquetas:", y_labels)

    # Pipeline XGBoost usando la API de scikit-learn
    clf = xgb.XGBClassifier(objective="multi:softmax", num_class=3, n_estimators=2, eval_metric='mlogloss')
    clf.fit(X, y_labels)
    preds = clf.predict(X)
    print("\n" + "="*40)
    print("[PREDICCIONES XGBOOST]:", preds)
    print("="*40 + "\n")

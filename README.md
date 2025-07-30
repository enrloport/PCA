
# Proyecto PCA + XGBoost para Clasificación de Eventos

Este proyecto implementa un pipeline de procesamiento y clasificación de señales provenientes de sensores instalados en un dispositivo (micrófono, IMU y piezoeléctrico) para la detección de eventos usando técnicas de extracción de características (MFCC, PCA) y un clasificador XGBoost.

## Estructura del proyecto

- `xgb.py`: Script principal que simula la generación de señales, aplica los embeddings y ejecuta el pipeline de clasificación con XGBoost.
- `utils/`: Módulos de utilidades para extracción de MFCC y PCA:
  - `mfcc_audio.py`: Extracción de MFCC de señales de audio.
  - `pca_imu.py`: PCA para señales de IMU.
  - `pca_piezo.py`: PCA para señales de piezoeléctrico.
- `test/`: Tests unitarios para las funciones de embedding.
  - `test_algorithms.py`: Tests de MFCC y PCA con datos sintéticos.

## Pipeline de procesamiento

1. **Generación de señales sintéticas**: Se simulan señales de audio, IMU y piezoeléctrico.
2. **Extracción de características**:
   - Audio: MFCC (Mel Frequency Cepstral Coefficients)
   - IMU y Piezo: PCA (Análisis de Componentes Principales)
3. **Concatenación de embeddings**: Los vectores de características de cada sensor se combinan en un único array.
4. **Clasificación con XGBoost**: El array combinado se utiliza como input para un modelo XGBoost que predice la clase del evento.

## Ejecución

### Ejecutar el pipeline principal
Desde la raíz del proyecto:
```bash
python xgb.py
```

### Ejecutar los tests
Desde la raíz del proyecto:
```bash
pytest
```
O bien:
```bash
python -m test.test_algorithms
```

## Requisitos
- Python 3.8+
- numpy, scikit-learn, librosa, soundfile, xgboost, pytest

Instala los requisitos en tu entorno (por ejemplo, con conda):
```bash
conda install numpy scikit-learn librosa soundfile xgboost pytest
```

## Notas
- El pipeline actual usa datos sintéticos y etiquetas dummy solo para probar la integración.
- Para un uso real, se debe alimentar el pipeline con datos reales y etiquetas representativas de los eventos a clasificar.

# AlphabetNet

Modelo RNN para predecir símbolos válidos después de un prefijo en autómatas finitos deterministas (DFA).

## Descripción

AlphabetNet es un modelo de clasificación multi-etiqueta que, dado un prefijo de caracteres, predice qué símbolos del alfabeto (A-L) son válidos como siguiente carácter según las reglas de un autómata finito.

## Arquitectura

- **Entrada**: Prefijo de caracteres (índices, padded) y opcionalmente ID de autómata
- **Embedding + RNN**: Embedding de caracteres + RNN (GRU o LSTM) → estado oculto del último carácter no-PAD
- **Salida**: Capa lineal → |Σ| = 12 logits (uno por cada símbolo A-L)
- **Activación**: Sigmoid en inferencia (logits en entrenamiento)

## Función de Pérdida y Métricas

### Función de Pérdida

Se utiliza **BCEWithLogitsLoss** con `pos_weight` a nivel símbolo para balancear la pérdida entre clases positivas y negativas.

### Métricas de Validación

#### Métrica Principal: auPRC Macro

**auPRC macro** (Average Precision macro): promedio simple del Average Precision (AP) por símbolo.

**Definición de "macro"**: 
- Se calcula el Average Precision (AP) para cada símbolo individualmente usando `sklearn.metrics.average_precision_score`
- El auPRC macro es el **promedio simple (media aritmética)** de estos APs
- Solo se incluyen en el promedio los símbolos que tienen al menos un ejemplo positivo (AP definido)
- Si un símbolo no tiene positivos en el conjunto de validación, su AP es `NaN` y se excluye del promedio

**Fórmula**:
```
macro_auPRC = (1 / N) * Σ(AP_i)
```
donde `N` es el número de símbolos con AP definido (no NaN) y `AP_i` es el Average Precision del símbolo i.

#### Métricas Adicionales

1. **micro-auPRC**: Average Precision calculada agregando todas las predicciones (tratando cada símbolo-observación como una predicción binaria independiente)

2. **F1@threshold**: F1 score usando un threshold (por defecto 0.5) para binarizar las probabilidades

3. **Coverage**: Porcentaje de símbolos con AP definido (es decir, símbolos que tienen al menos un ejemplo positivo en el conjunto de validación)

## Uso

### Ejemplo de evaluación de métricas

```python
import numpy as np
from metrics import evaluate_metrics

# Ejemplo: 100 muestras, 12 símbolos
y_true = np.random.randint(0, 2, size=(100, 12))  # Etiquetas binarias
y_scores = np.random.rand(100, 12)  # Scores/probabilidades

alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']

metrics = evaluate_metrics(y_true, y_scores, alphabet, threshold=0.5)

print(f"Macro auPRC: {metrics['macro_auprc']:.4f}")
print(f"Micro auPRC: {metrics['micro_auprc']:.4f}")
print(f"F1@0.5: {metrics['f1_at_threshold']:.4f}")
print(f"Coverage: {metrics['coverage']:.2f}%")
```

### Función de pérdida con pos_weight

```python
import torch
import torch.nn as nn
from metrics import compute_pos_weight

# Calcular pos_weight para cada símbolo
# pos_weight[i] = num_negativos[i] / num_positivos[i]
pos_weight = compute_pos_weight(y_train)  # y_train: (n_samples, n_symbols)

# Crear función de pérdida
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Uso en entrenamiento
logits = model(prefix_indices, lengths)
loss = criterion(logits, y_true)
```

## Datos

Los datos están en formato "wide" (multi-hot) con columnas:
- `dfa_id`: ID del autómata
- `prefix`: Prefijo de caracteres
- `y`: Lista de 12 enteros (0 o 1) indicando si cada símbolo es válido
- `support_pos`: Lista de 12 enteros con el soporte de cada símbolo positivo

## Configuración

Ver `hparams.json` para hiperparámetros del modelo y entrenamiento.

## Archivos

- `model.py`: Definición del modelo AlphabetNet
- `metrics.py`: Cálculo de métricas de evaluación
- `hparams.json`: Configuración de hiperparámetros
- `scripts/`: Scripts para procesamiento de datos y generación de splits


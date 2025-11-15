# Model Card: AlphabetNet

## Información General

**Modelo**: AlphabetNet  
**Versión**: 1.0  
**Tipo**: Red Neuronal Recurrente (RNN) para Clasificación Multi-etiqueta  
**Propósito**: Predecir qué símbolos del alfabeto (A-L) son válidos después de un prefijo en autómatas finitos deterministas (DFA)

### Arquitectura

- **Embedding Layer**: Embedding de caracteres (vocab_size=14, emb_dim=96)
- **RNN Layer**: GRU o LSTM unidireccional (hidden_dim=192, num_layers=1)
- **Conditioning Layer** (opcional): Embedding de ID de autómata (automata_emb_dim=16)
- **Output Layer**: Capa lineal (hidden_dim → alphabet_size=12)
- **Activación**: Sigmoid para inferencia, logits para entrenamiento

### Datos de Entrenamiento

- **Conjunto de entrenamiento**: `data/alphabet/train_wide.parquet`
- **Conjunto de validación**: `data/alphabet/val_wide.parquet`
- **Conjunto de prueba**: `data/alphabet/test_wide.parquet`
- **Alfabeto**: A, B, C, D, E, F, G, H, I, J, K, L (12 símbolos)
- **Tokens especiales**: PAD (índice 0), <EPS> (índice 1)
- **Longitud máxima de prefijo**: 64 caracteres

## Entrada/Salida

### Entrada

El modelo acepta los siguientes inputs:

#### 1. `prefix_indices` (requerido)
- **Tipo**: `torch.LongTensor` o `numpy.ndarray` (int64)
- **Forma**: `(batch_size, max_seq_len)` donde `max_seq_len = 64`
- **Descripción**: Índices de caracteres del prefijo, con padding (PAD=0)
- **Valores válidos**: 
  - `0` = PAD (padding)
  - `1` = <EPS> (prefijo vacío)
  - `2-13` = Caracteres A-L (2=A, 3=B, ..., 13=L)

**Ejemplo**:
```python
# Prefijo "ABC" (con padding)
prefix_indices = torch.tensor([[2, 3, 4, 0, 0, ...]])  # A=2, B=3, C=4, resto PAD=0
```

#### 2. `lengths` (requerido)
- **Tipo**: `torch.LongTensor` o `numpy.ndarray` (int64)
- **Forma**: `(batch_size,)`
- **Descripción**: Longitud real de cada prefijo (sin contar padding)
- **Valores válidos**: Enteros en rango [1, max_seq_len]

**Ejemplo**:
```python
# Para prefijo "ABC" de longitud 3
lengths = torch.tensor([3])
```

#### 3. `automata_ids` (opcional)
- **Tipo**: `torch.LongTensor` o `numpy.ndarray` (int64)
- **Forma**: `(batch_size,)`
- **Descripción**: IDs de autómatas para conditioning (solo si `use_automata_conditioning=True`)
- **Valores válidos**: Enteros en rango [0, num_automata)

**Ejemplo**:
```python
# Para autómata con ID 42
automata_ids = torch.tensor([42])
```

### Salida

#### `logits` o `probabilities`
- **Tipo**: `torch.FloatTensor` o `numpy.ndarray` (float32)
- **Forma**: `(batch_size, alphabet_size)` donde `alphabet_size = 12`
- **Descripción**: 
  - Si `return_logits=True`: Logits para cada símbolo A-L (valores no normalizados)
  - Si `return_logits=False`: Probabilidades para cada símbolo A-L (aplicando sigmoid, valores en [0, 1])
- **Orden de símbolos**: A (índice 0), B (índice 1), C (índice 2), ..., L (índice 11)

**Ejemplo**:
```python
# Salida con logits
logits = model(prefix_indices, lengths, return_logits=True)
# Shape: (batch_size, 12)
# logits[0, 0] = logit para símbolo 'A'
# logits[0, 1] = logit para símbolo 'B'
# ...

# Salida con probabilidades
probs = model(prefix_indices, lengths, return_logits=False)
# Shape: (batch_size, 12)
# probs[0, 0] = probabilidad para símbolo 'A' (en [0, 1])
# probs[0, 1] = probabilidad para símbolo 'B' (en [0, 1])
# ...
```

## Métricas de Rendimiento

### Métrica Principal: auPRC Macro

- **Métrica**: Average Precision macro (promedio simple de AP por símbolo)
- **Rango**: [0, 1] donde 1 es el mejor
- **Interpretación**: Mide la capacidad del modelo para predecir correctamente símbolos válidos, promediando sobre todos los símbolos

### Métricas Adicionales

- **auPRC Micro**: Average Precision agregando todas las predicciones
- **F1@threshold**: F1 score usando threshold=0.5 para binarizar probabilidades
- **Coverage**: Porcentaje de símbolos con AP definido (con al menos un ejemplo positivo)

### Rendimiento Esperado

*Nota: Los valores exactos dependen del entrenamiento específico. Consultar reportes generados en `checkpoints/A2_report.md` para métricas del modelo entrenado.*

## Uso del Modelo

### Cargar Modelo desde Checkpoint

```python
import torch
from model import AlphabetNet
import json

# Cargar hiperparámetros
with open('hparams.json', 'r') as f:
    hparams = json.load(f)

# Crear modelo
model = AlphabetNet(
    vocab_size=hparams['model']['vocab_size'],
    alphabet_size=hparams['model']['alphabet_size'],
    emb_dim=hparams['model']['emb_dim'],
    hidden_dim=hparams['model']['hidden_dim'],
    rnn_type=hparams['model']['rnn_type'],
    num_layers=hparams['model']['num_layers'],
    dropout=hparams['model']['dropout'],
    padding_idx=hparams['model']['padding_idx'],
    use_automata_conditioning=hparams['model']['use_automata_conditioning'],
    num_automata=hparams['model'].get('num_automata'),
    automata_emb_dim=hparams['model']['automata_emb_dim']
)

# Cargar pesos
checkpoint = torch.load('checkpoints/best.pt', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### Inferencia Rápida

```bash
# Usando script de inferencia
python infer.py --checkpoint checkpoints/best.pt --prefix "ABC" --top_k 5
```

### Exportación ONNX

```bash
# Exportar a ONNX
python export_model.py --checkpoint checkpoints/best.pt --output alphabetnet.onnx
```

## Límites Conocidos

### Limitaciones del Modelo

1. **Alfabeto Fijo**: El modelo solo maneja los 12 símbolos A-L. No puede predecir símbolos fuera de este alfabeto.

2. **Longitud Máxima**: Los prefijos están limitados a 64 caracteres. Prefijos más largos se truncan.

3. **Vocabulario Fijo**: El modelo está entrenado con un vocabulario específico (vocab_size=14). No puede manejar caracteres fuera del vocabulario.

4. **Autómatas Específicos**: El modelo está entrenado en un conjunto específico de autómatas. Su rendimiento puede degradarse en autómatas muy diferentes al conjunto de entrenamiento.

5. **Precisión Numérica**: El modelo usa float32. Puede haber pequeñas variaciones numéricas en diferentes plataformas o versiones de PyTorch.

### Casos Edge

1. **Prefijo Vacío**: El prefijo vacío se representa como `<EPS>` (índice 1).

2. **Símbolos sin Ejemplos**: Si un símbolo no tiene ejemplos positivos en validación, su AP es `NaN` y se excluye del auPRC macro.

3. **Batch Size Variable**: El modelo soporta batch size variable, pero el padding debe ser correcto.

4. **Conditioning Opcional**: Si `use_automata_conditioning=False`, el parámetro `automata_ids` se ignora.

### Requisitos de Sistema

- **PyTorch**: >= 2.0.0
- **Python**: >= 3.8
- **Memoria**: Depende del batch size, aproximadamente 500MB-2GB para inferencia
- **GPU**: Opcional pero recomendado para entrenamiento

## Reproducibilidad

El modelo usa seeds fijos para reproducibilidad:
- Python random seed: 42
- NumPy seed: 42
- PyTorch seed: 42
- CUDA seed: 42

Para reproducir resultados exactos, configurar `torch.backends.cudnn.deterministic = True` (puede reducir rendimiento en GPU).

## Archivos del Modelo

- **`checkpoints/best.pt`**: Mejor checkpoint según auPRC macro en validación
- **`checkpoints/last.pt`**: Último checkpoint guardado
- **`alphabetnet.onnx`**: Modelo exportado en formato ONNX (generado con `export_model.py`)

## Referencias

- Ver `README.md` para información general del proyecto
- Ver `checkpoints/A2_report.md` para reportes de validación y métricas detalladas
- Ver `hparams.json` para hiperparámetros del modelo

## Licencia

*Especificar licencia según corresponda*

## Contacto

*Información de contacto para preguntas o reportes de problemas*


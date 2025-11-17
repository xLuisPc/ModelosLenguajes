# AlphabetNet - Standalone Inference

Esta carpeta contiene todos los archivos necesarios para ejecutar inferencia con el modelo AlphabetNet de forma independiente.

## Estructura de archivos

```
standalone_inference/
├── test_model.py          # Script principal para inferencia
├── hparams.json           # Hiperparámetros del modelo
├── src/                   # Código fuente del modelo
│   ├── __init__.py
│   ├── model.py           # Definición del modelo AlphabetNet
│   ├── train.py           # Utilidades (ALPHABET, regex_to_indices, etc.)
│   └── metrics.py         # Métricas (no usado directamente en inferencia)
└── novTest/               # Modelo y thresholds
    ├── alphabetnet.pt     # Checkpoint del modelo entrenado
    └── thresholds.json    # Thresholds óptimos por símbolo
```

## Uso

### Requisitos

```bash
pip install torch pandas numpy scikit-learn
```

### Ejemplos de uso

1. **Predicción de un solo regex:**
```bash
python test_model.py --checkpoint novTest/alphabetnet.pt --thresholds novTest/thresholds.json --regex "((A+B+((C.D)+E)*)"
```

2. **Modo interactivo:**
```bash
python test_model.py --checkpoint novTest/alphabetnet.pt --thresholds novTest/thresholds.json
```

3. **Procesar CSV completo:**
```bash
python test_model.py --checkpoint novTest/alphabetnet.pt --thresholds novTest/thresholds.json --csv dataset.csv --output predictions.csv
```

## Archivos necesarios

Todos los archivos en esta carpeta son necesarios para ejecutar el comando:

- `test_model.py` - Script principal
- `src/model.py` - Modelo AlphabetNet
- `src/train.py` - Funciones de utilidad (ALPHABET, MAX_PREFIX_LEN, regex_to_indices)
- `src/__init__.py` - Inicialización del paquete
- `src/metrics.py` - Métricas (importado por train.py)
- `hparams.json` - Hiperparámetros del modelo
- `novTest/alphabetnet.pt` - Checkpoint del modelo
- `novTest/thresholds.json` - Thresholds por símbolo

## Notas

- El script busca archivos relativos al directorio actual (`standalone_inference/`)
- Los paths en los argumentos deben ser relativos a este directorio
- El modelo funciona con CPU o GPU (se detecta automáticamente)


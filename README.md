# AlphabetNet

Modelo RNN para predecir el alfabeto de un autómata finito a partir de su expresión regular.

## Estructura del Proyecto

```
ModelosLenguajes/
├── src/                  # Código fuente principal
│   ├── model.py         # Arquitectura del modelo AlphabetNet
│   ├── train.py         # Script de entrenamiento
│   ├── metrics.py       # Métricas de evaluación
│   └── utils.py         # Utilidades auxiliares
│
├── tools/                # Herramientas y scripts de análisis
│   ├── infer.py         # Inferencia desde línea de comandos
│   ├── find_thresholds.py  # Búsqueda de umbrales óptimos
│   ├── export_model.py  # Exportación a ONNX
│   ├── generate_pr_curves.py  # Generación de curvas PR
│   └── ablation_study.py  # Estudios de ablación
│
├── scripts/              # Scripts de procesamiento de datos
│   ├── create_regex_sigma_dataset.py  # Generar dataset regex→alfabeto
│   └── ...
│
├── notebooks/            # Notebooks de Colab
│   ├── alphabetnet_colab_standalone.py  # Script standalone para Colab
│   ├── run_all_colab.py  # Pipeline completo para Colab
│   └── ...
│
├── demo/                 # Interfaz interactiva para probar modelos
│   ├── test_model.py    # CLI interactiva para probar modelos
│   └── README.md        # Documentación del demo
│
├── data/                 # Datos
│   ├── dataset_regex_sigma.csv  # Dataset principal (regex → alfabeto)
│   ├── dataset3000.csv  # Dataset original
│   └── ...
│
├── checkpoints/          # Modelos entrenados
│   ├── best.pt          # Mejor modelo
│   ├── last.pt          # Último checkpoint
│   └── train_log.csv    # Log de entrenamiento
│
├── reports/              # Reportes y análisis
│   └── ...
│
├── docs/                 # Documentación
│   ├── README.md        # Este archivo
│   ├── MODEL_CARD.md    # Tarjeta del modelo
│   └── ...
│
├── hparams.json          # Hiperparámetros del modelo
└── requirements.txt      # Dependencias Python
```

## Inicio Rápido

### 1. Instalación

```bash
pip install -r requirements.txt
```

### 2. Entrenar un Modelo

```bash
python src/train.py \
  --train_data data/dataset_regex_sigma.csv \
  --val_data data/dataset_regex_sigma.csv \
  --checkpoint_dir checkpoints \
  --use_scheduler
```

### 3. Probar un Modelo (Interfaz Interactiva)

```bash
# Modo interactivo
python demo/test_model.py --checkpoint checkpoints/best.pt

# O predicción de un solo regex
python demo/test_model.py --checkpoint checkpoints/best.pt --regex "(AB)*C"
```

### 4. Buscar Umbrales Óptimos

```bash
python tools/find_thresholds.py \
  --checkpoint checkpoints/best.pt \
  --val_data data/dataset_regex_sigma.csv \
  --output_dir checkpoints
```

### 5. Inferencia (Línea de Comandos)

```bash
python tools/infer.py \
  --checkpoint checkpoints/best.pt \
  --regex "(AB)*C" \
  --thresholds checkpoints/thresholds.json
```

## Uso en Google Colab

Ver `notebooks/alphabetnet_colab_standalone.py` para un script standalone que incluye todo lo necesario para entrenar y probar en Colab.

## Tarea

El modelo predice el alfabeto de un autómata finito desde su expresión regular:
- **Entrada**: Expresión regular (regex)
- **Salida**: Conjunto de símbolos que pertenecen al alfabeto (A-L)

## Métricas

- **F1 Macro**: Promedio de F1 score por símbolo (objetivo: ≥ 0.92)
- **F1 Mínimo**: F1 score del símbolo con peor rendimiento (objetivo: ≥ 0.85)
- **ECE**: Error de calibración esperado (objetivo: ≤ 0.05)
- **Exactitud de Conjunto**: Porcentaje de predicciones exactas del conjunto completo (objetivo: ≥ 0.90)

## Referencias

Ver `docs/MODEL_CARD.md` para más detalles sobre el modelo y `docs/` para documentación adicional.

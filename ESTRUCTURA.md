# Estructura del Proyecto AlphabetNet

## 📁 Organización

```
ModelosLenguajes/
│
├── 📂 src/                      # Código fuente principal
│   ├── model.py                 # Arquitectura AlphabetNet
│   ├── train.py                 # Script de entrenamiento
│   ├── metrics.py               # Métricas de evaluación
│   ├── utils.py                 # Utilidades auxiliares
│   └── __init__.py              # Exports principales
│
├── 📂 tools/                     # Herramientas de análisis y utilidades
│   ├── infer.py                 # Inferencia desde CLI
│   ├── find_thresholds.py       # Búsqueda de umbrales óptimos
│   ├── export_model.py          # Exportación a ONNX
│   ├── generate_pr_curves.py    # Generación de curvas PR
│   └── ablation_study.py        # Estudios de ablación
│
├── 📂 scripts/                   # Scripts de procesamiento de datos
│   ├── create_regex_sigma_dataset.py  # Generar dataset regex→alfabeto
│   ├── create_splits.py         # Crear splits de datos
│   ├── generate_continuations.py  # Generar continuaciones
│   ├── process_dataset.py       # Procesar dataset
│   └── ...
│
├── 📂 notebooks/                 # Notebooks y scripts para Colab
│   ├── alphabetnet_colab_standalone.py  # Script standalone para Colab
│   ├── AlphabetNet_Colab.ipynb  # Notebook de Colab
│   ├── alphabetnet_colab.py     # Script de Colab
│   └── run_all_colab.py         # Pipeline completo para Colab
│
├── 📂 demo/                      # Interfaz interactiva para probar modelos
│   ├── test_model.py            # CLI interactiva para probar modelos
│   └── README.md                # Documentación del demo
│
├── 📂 data/                      # Datos
│   ├── dataset_regex_sigma.csv  # Dataset principal (regex → alfabeto)
│   ├── dataset3000.csv          # Dataset original
│   ├── alphabet/                # Datos de alfabeto (legacy)
│   └── ...
│
├── 📂 checkpoints/               # Modelos entrenados
│   ├── best.pt                  # Mejor modelo
│   ├── last.pt                  # Último checkpoint
│   ├── train_log.csv            # Log de entrenamiento
│   └── thresholds.json          # Umbrales óptimos (si existe)
│
├── 📂 reports/                   # Reportes y análisis
│   ├── figures/                 # Figuras y gráficos
│   └── ...
│
├── 📂 docs/                      # Documentación
│   ├── README.md                # Documentación principal
│   ├── MODEL_CARD.md            # Tarjeta del modelo
│   ├── COLAB_INSTRUCCIONES.md   # Instrucciones para Colab
│   └── ...
│
├── 📂 meta/                      # Metadatos
│   └── dataset_version.json     # Versión del dataset
│
├── hparams.json                  # Hiperparámetros del modelo
├── requirements.txt              # Dependencias Python
├── README.md                     # README principal
├── test.py                       # Script rápido para probar modelos
└── ESTRUCTURA.md                 # Este archivo
```

## 🚀 Uso Rápido

### Probar un Modelo (Interfaz Interactiva)

```bash
# Modo interactivo (recomendado)
python test.py

# O directamente
python demo/test_model.py --checkpoint checkpoints/best.pt

# Predicción de un solo regex
python test.py --regex "(AB)*C"
```

### Entrenar un Modelo

```bash
python src/train.py \
  --train_data data/dataset_regex_sigma.csv \
  --val_data data/dataset_regex_sigma.csv \
  --checkpoint_dir checkpoints \
  --use_scheduler
```

### Buscar Umbrales Óptimos

```bash
python tools/find_thresholds.py \
  --checkpoint checkpoints/best.pt \
  --val_data data/dataset_regex_sigma.csv \
  --output_dir checkpoints
```

### Inferencia (Línea de Comandos)

```bash
python tools/infer.py \
  --checkpoint checkpoints/best.pt \
  --regex "(AB)*C" \
  --thresholds checkpoints/thresholds.json
```

## 📝 Notas

- **src/**: Código fuente principal que puede ser importado como módulo
- **tools/**: Scripts independientes para análisis y utilidades
- **scripts/**: Scripts de procesamiento de datos
- **notebooks/**: Scripts específicos para Google Colab
- **demo/**: Interfaz interactiva para usuarios finales
- **data/**: Todos los datos del proyecto
- **checkpoints/**: Modelos entrenados y logs
- **docs/**: Documentación completa del proyecto

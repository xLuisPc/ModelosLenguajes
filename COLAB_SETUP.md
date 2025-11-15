# Guía para Ejecutar en Google Colab

## Archivos a Subir

### Archivos Python (obligatorios):
1. `model.py` - Definición del modelo
2. `metrics.py` - Cálculo de métricas
3. `train.py` - Script de entrenamiento
4. `generate_pr_curves.py` - Generación de curvas PR
5. `find_thresholds.py` - Búsqueda de umbrales
6. `export_model.py` - Exportación a ONNX ⭐
7. `infer.py` - Inferencia rápida
8. `utils.py` - Utilidades

### Archivos de configuración:
9. `hparams.json` - Hiperparámetros

### Datos:
10. `data/alphabet/train_wide.parquet` - Datos de entrenamiento
11. `data/alphabet/val_wide.parquet` - Datos de validación
12. `data/alphabet/test_wide.parquet` - Datos de prueba (opcional)

### Opcionales:
- `ablation_study.py` - Ablation study
- `MODEL_CARD.md` - Documentación
- `README.md` - Documentación

## Pasos en Colab

### 1. Subir archivos
```python
# Opción 1: Subir manualmente desde el panel izquierdo de Colab
# Opción 2: Usar este código para subir desde Google Drive
from google.colab import drive
drive.mount('/content/drive')
# Copiar archivos desde Drive a /content/
```

### 2. Instalar dependencias
```python
!pip install torch numpy pandas scikit-learn matplotlib pyarrow --quiet
```

### 3. Verificar estructura
```python
import os
print("Archivos Python:")
for f in ['model.py', 'metrics.py', 'train.py', 'generate_pr_curves.py', 
          'find_thresholds.py', 'export_model.py', 'infer.py', 'utils.py', 'hparams.json']:
    exists = os.path.exists(f)
    print(f"  {f}: {'✓' if exists else '✗'}")
    
print("\nArchivos de datos:")
for f in ['data/alphabet/train_wide.parquet', 'data/alphabet/val_wide.parquet']:
    exists = os.path.exists(f)
    print(f"  {f}: {'✓' if exists else '✗'}")
```

### 4. Entrenar modelo
```python
!python train.py --train_data data/alphabet/train_wide.parquet \
                 --val_data data/alphabet/val_wide.parquet \
                 --checkpoint_dir checkpoints
```

### 5. Generar curvas PR
```python
!python generate_pr_curves.py --checkpoint checkpoints/best.pt \
                            --val_data data/alphabet/val_wide.parquet \
                            --output_dir checkpoints
```

### 6. Buscar umbrales
```python
!python find_thresholds.py --checkpoint checkpoints/best.pt \
                           --val_data data/alphabet/val_wide.parquet \
                           --output_dir checkpoints
```

### 7. Exportar a ONNX ⭐
```python
!python export_model.py --checkpoint checkpoints/best.pt \
                        --output alphabetnet.onnx
```

### 8. Verificar exportación ONNX
```python
import onnx
import onnxruntime as ort

# Verificar que el archivo existe
import os
if os.path.exists('alphabetnet.onnx'):
    print("✓ Modelo ONNX exportado correctamente")
    
    # Cargar y verificar modelo ONNX
    model = onnx.load('alphabetnet.onnx')
    onnx.checker.check_model(model)
    print("✓ Modelo ONNX válido")
    
    # Ver inputs y outputs
    print("\nInputs:")
    for inp in model.graph.input:
        print(f"  {inp.name}: {[d.dim_value for d in inp.type.tensor_type.shape.dim]}")
    
    print("\nOutputs:")
    for out in model.graph.output:
        print(f"  {out.name}: {[d.dim_value for d in out.type.tensor_type.shape.dim]}")
else:
    print("✗ Modelo ONNX no encontrado")
```

### 9. Probar inferencia
```python
!python infer.py --checkpoint checkpoints/best.pt --prefix "ABC" --top_k 5
```

### 10. Descargar resultados
```python
# Descargar archivos importantes
from google.colab import files

# Modelo ONNX
files.download('alphabetnet.onnx')

# Mejor checkpoint
files.download('checkpoints/best.pt')

# Reportes
files.download('checkpoints/A2_report.md')
files.download('checkpoints/per_symbol_ap.csv')
files.download('checkpoints/thresholds.json')

# Gráficos
files.download('checkpoints/pr_macro.png')
files.download('checkpoints/pr_top10.png')
```

## Comando Todo-en-Uno (Script rápido)

Si quieres ejecutar todo de una vez:

```python
# Script completo
import subprocess
import sys

commands = [
    "python train.py --train_data data/alphabet/train_wide.parquet --val_data data/alphabet/val_wide.parquet --checkpoint_dir checkpoints",
    "python generate_pr_curves.py --checkpoint checkpoints/best.pt --val_data data/alphabet/val_wide.parquet --output_dir checkpoints",
    "python find_thresholds.py --checkpoint checkpoints/best.pt --val_data data/alphabet/val_wide.parquet --output_dir checkpoints",
    "python export_model.py --checkpoint checkpoints/best.pt --output alphabetnet.onnx"
]

for cmd in commands:
    print(f"\n{'='*60}")
    print(f"Ejecutando: {cmd}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd.split(), capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Errores:", result.stderr)
    if result.returncode != 0:
        print(f"Error en comando: {cmd}")
        break
```

## Estructura de Archivos en Colab

```
/content/
├── model.py
├── metrics.py
├── train.py
├── generate_pr_curves.py
├── find_thresholds.py
├── export_model.py          ⭐ Para exportar a ONNX
├── infer.py
├── utils.py
├── hparams.json
├── data/
│   └── alphabet/
│       ├── train_wide.parquet
│       ├── val_wide.parquet
│       └── test_wide.parquet
└── checkpoints/             (se crea automáticamente)
    ├── best.pt
    ├── last.pt
    ├── train_log.csv
    ├── pr_macro.png
    ├── pr_top10.png
    ├── per_symbol_ap.csv
    ├── A2_report.md
    ├── thresholds.json
    └── threshold_eval.csv
```

## Notas Importantes

1. **GPU**: Colab proporciona GPU gratuita. El entrenamiento será más rápido con GPU.

2. **Tiempo**: El entrenamiento puede tomar 30-60 minutos dependiendo del tamaño del dataset.

3. **Memoria**: Si tienes problemas de memoria, reduce el `batch_size` en `hparams.json`.

4. **Exportación ONNX**: Requiere que el modelo esté entrenado (`best.pt` debe existir).

5. **Verificación ONNX**: Puedes instalar `onnx` y `onnxruntime` para verificar:
   ```python
   !pip install onnx onnxruntime
   ```

## Solución de Problemas

### Error: "No module named 'torch'"
```python
!pip install torch numpy pandas scikit-learn matplotlib pyarrow
```

### Error: "Checkpoint no encontrado"
- Asegúrate de que el entrenamiento se completó
- Verifica que `checkpoints/best.pt` existe

### Error en exportación ONNX
- Verifica que el modelo se entrenó correctamente
- Asegúrate de tener la versión correcta de PyTorch (>=2.0.0)

### Problemas con datos
- Verifica que los archivos parquet estén en `data/alphabet/`
- Verifica que los archivos no estén corruptos


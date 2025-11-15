# Instrucciones para Google Colab

## 📁 Archivo a Subir

**Solo necesitas subir UN archivo:**
- `alphabetnet_colab.py` - Contiene todo el código necesario

## 📊 Datos a Subir en sample_data/

En Colab, crea una carpeta llamada `sample_data` y sube estos 2 archivos:

1. **`train_wide.parquet`** - Datos de entrenamiento
2. **`val_wide.parquet`** - Datos de validación

### Estructura en Colab:
```
/content/
├── alphabetnet_colab.py    (subir este archivo)
└── sample_data/            (crear esta carpeta)
    ├── train_wide.parquet  (subir aquí)
    └── val_wide.parquet    (subir aquí)
```

## 🚀 Pasos en Colab

### 1. Instalar dependencias
```python
!pip install torch numpy pandas scikit-learn matplotlib pyarrow --quiet
```

### 2. Subir archivos
- Sube `alphabetnet_colab.py` a la raíz de Colab
- Crea carpeta `sample_data` y sube los 2 archivos parquet

### 3. Ejecutar todo
```python
!python alphabetnet_colab.py
```

## ✅ Resultados Generados

Después de ejecutar, tendrás:

- `checkpoints/best.pt` - Mejor modelo PyTorch
- `alphabetnet.onnx` - Modelo ONNX exportado ⭐

## 📥 Descargar Resultados

```python
from google.colab import files

# Descargar modelo ONNX
files.download('alphabetnet.onnx')

# Descargar mejor checkpoint
files.download('checkpoints/best.pt')
```

## 📝 Notas

- El entrenamiento puede tomar 30-60 minutos
- Usa GPU en Colab para acelerar (Runtime > Change runtime type > GPU)
- Los archivos parquet deben estar en `sample_data/` (no en `data/alphabet/`)


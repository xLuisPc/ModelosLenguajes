# Demo - Probar Modelos AlphabetNet

Esta carpeta contiene herramientas interactivas para probar modelos AlphabetNet entrenados.

## test_model.py

Interfaz interactiva para probar modelos desde la línea de comandos.

### Uso Básico

```bash
# Modo interactivo (recomendado)
python demo/test_model.py --checkpoint checkpoints/best.pt

# Predicción de un solo regex
python demo/test_model.py --checkpoint checkpoints/best.pt --regex "(AB)*C"

# Con thresholds personalizados
python demo/test_model.py --checkpoint checkpoints/best.pt --thresholds checkpoints/thresholds.json --regex "A+B*"
```

### Modo Interactivo

El modo interactivo te permite probar múltiples regexes sin tener que ejecutar el script cada vez:

```bash
python demo/test_model.py --checkpoint checkpoints/best.pt
```

Luego puedes ingresar regexes y ver las predicciones:

```
Regex: (AB)*C
```

Para salir, escribe `quit` o `exit`.

### Procesar CSV Completo

Procesa un CSV completo (como `dataset3000.csv`) y genera predicciones para todas las filas:

```bash
# Procesar CSV completo
python demo/test_model.py --checkpoint checkpoints/best.pt --csv data/dataset3000.csv --output data/predictions.csv

# Sin especificar output (genera automáticamente dataset3000_predictions.csv)
python demo/test_model.py --checkpoint checkpoints/best.pt --csv data/dataset3000.csv
```

El CSV de salida contiene:
- `regex`: Expresión regular original
- `alfabeto_predicho`: Alfabeto predicho por el modelo (ej: "A, B, C")
- `alfabeto_real`: Alfabeto real extraído de la columna 'Alfabeto' del CSV (ej: "A B C")

### Opciones

- `--checkpoint`: Path al checkpoint del modelo (requerido)
- `--hparams`: Path al archivo de hiperparámetros (default: `hparams.json`)
- `--thresholds`: Path al archivo JSON con thresholds por símbolo (opcional)
- `--regex`: Regex para predecir (si no se especifica, modo interactivo)
- `--csv`: Path al CSV de entrada para procesar completo (ej: `data/dataset3000.csv`)
- `--output`: Path donde guardar el CSV de salida (solo con `--csv`)
- `--device`: Dispositivo: cpu, cuda, o auto (default: auto)

### Ejemplo de Salida

```
======================================================================
PREDICCIÓN DE ALFABETO DESDE REGEX
======================================================================
Regex: '(AB)*C'

Probabilidades por símbolo:
----------------------------------------------------------------------
Símbolo    Probabilidad    Threshold    Predicción
----------------------------------------------------------------------
A          0.999500        0.5000       ✓ SÍ
B          0.999400        0.5000       ✓ SÍ
C          0.997700        0.5000       ✓ SÍ
D          0.026700        0.5000         NO
...
----------------------------------------------------------------------

Alfabeto predicho (sigma_hat): A, B, C
======================================================================
```

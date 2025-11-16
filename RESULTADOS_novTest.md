# Resultados del Modelo novTest

## Resumen Ejecutivo

El modelo novTest fue entrenado y evaluado con diferentes thresholds para optimizar la exactitud de conjunto (set accuracy).

## Métricas del Modelo Entrenado

- **F1 Macro**: 1.000000
- **F1 Min**: 1.000000  
- **ECE**: 0.5993
- **Época**: 5

## Comparación de Thresholds

### Thresholds Originales (0.92-0.98)
- **Exactitud de conjunto**: 49.93% (1,498/3,000 correctas)
- **Problema**: Thresholds demasiado altos causan que el modelo sea muy conservador
- **Resultado**: Todos los errores son **falsos negativos** (no predice símbolos que sí debería)

### Thresholds Sugeridos (0.87-0.93)
- **Exactitud de conjunto**: 86.03% (2,581/3,000 correctas)
- **Mejora**: +36.10 puntos porcentuales
- **Resultado**: Mejor balance entre precision y recall

## Análisis de Errores

### Con Thresholds Originales
- **Total de errores**: 1,502 (50.07%)
- **Falsos Positivos**: 0 (0%)
- **Falsos Negativos**: 1,502 (100% de errores)
- **Problema**: El modelo NO predice símbolos que SÍ debería predecir

### Con Thresholds Sugeridos
- **Total de errores**: 419 (13.97%)
- **Mejora**: -72% de errores
- **Resultado**: Predicciones mucho más balanceadas

## Thresholds Recomendados

Los thresholds sugeridos están guardados en:
- `checkpoints/thresholds_novTest_suggested.json`

Valores recomendados:
- A: 0.8765 (original: 0.9227)
- B: 0.9381 (original: 0.9875)
- C: 0.9275 (original: 0.9763)
- D: 0.9335 (original: 0.9826)
- E: 0.9295 (original: 0.9784)
- F: 0.9350 (original: 0.9842)
- G: 0.9273 (original: 0.9761)
- H: 0.9362 (original: 0.9855)
- I: 0.9336 (original: 0.9828)
- J: 0.9316 (original: 0.9807)
- K: 0.9323 (original: 0.9814)
- L: 0.9344 (original: 0.9835)

## Uso del Modelo

### Con thresholds originales
```bash
python demo/test_model.py \
  --checkpoint checkpoints/best_novTest.pt \
  --thresholds checkpoints/thresholds_novTest.json \
  --regex "(AB)*C"
```

### Con thresholds sugeridos (recomendado)
```bash
python demo/test_model.py \
  --checkpoint checkpoints/best_novTest.pt \
  --thresholds checkpoints/thresholds_novTest_suggested.json \
  --regex "(AB)*C"
```

### Evaluación completa
```bash
python demo/test_model.py \
  --checkpoint checkpoints/best_novTest.pt \
  --thresholds checkpoints/thresholds_novTest_suggested.json \
  --csv data/dataset3000.csv \
  --output data/predictions_suggested.csv
```

## Conclusiones

1. ✅ **El modelo tiene muy buen rendimiento** (F1 macro = 1.0, F1 min = 1.0)
2. ⚠️ **Los thresholds originales eran demasiado altos**, causando que el modelo fuera muy conservador
3. ✅ **Los thresholds sugeridos mejoran la exactitud de conjunto de 49.93% a 86.03%**
4. 📊 **El modelo ahora predice correctamente 8.6 de cada 10 alfabetos**

## Archivos Generados

- `checkpoints/best_novTest.pt` - Modelo entrenado
- `checkpoints/thresholds_novTest.json` - Thresholds originales
- `checkpoints/thresholds_novTest_suggested.json` - Thresholds sugeridos (recomendado)
- `data/dataset3000_predictions_novTest.csv` - Predicciones con thresholds originales
- `data/predictions_suggested.csv` - Predicciones con thresholds sugeridos


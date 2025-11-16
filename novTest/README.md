# Modelo novTest - Thresholds Optimizados

## 📊 Resultados

### Thresholds Originales
- **Exactitud**: 49.93% (1,498/3,000 correctas)
- **Problema**: Thresholds demasiado altos (0.92-0.98)

### Thresholds Optimizados ✅
- **Exactitud**: 86.03% (2,581/3,000 correctas)
- **Mejora**: +36.10 puntos porcentuales
- **Resultado**: 8.6 de cada 10 alfabetos predicen correctamente

## 📁 Archivos

- `best (1).pt` - Modelo entrenado (mejor checkpoint)
- `last.pt` - Último checkpoint
- `train_log.csv` - Log completo de entrenamiento
- `thresholds.json` - **Thresholds optimizados** (actualizado)
- `thresholds_optimized.json` - Copia de los thresholds optimizados

## 🚀 Uso del Modelo

### Probar con un regex específico

```bash
# Usar modelo con thresholds optimizados
python demo/test_model.py \
  --checkpoint novTest/best\ \(1\).pt \
  --thresholds novTest/thresholds.json \
  --regex "(AB)*C"
```

### Procesar CSV completo

```bash
# Evaluar todas las regex del dataset
python demo/test_model.py \
  --checkpoint novTest/best\ \(1\).pt \
  --thresholds novTest/thresholds.json \
  --csv data/dataset3000.csv \
  --output data/predictions_novTest_optimized.csv
```

### Modo interactivo

```bash
# Modo interactivo para probar múltiples regex
python demo/test_model.py \
  --checkpoint novTest/best\ \(1\).pt \
  --thresholds novTest/thresholds.json
```

## 📈 Métricas del Modelo

- **F1 Macro**: 1.000000
- **F1 Min**: 1.000000
- **ECE**: 0.5993
- **Época**: 5
- **Set Accuracy (con thresholds optimizados)**: 86.03%

## 🔧 Thresholds Actuales (Optimizados)

| Símbolo | Threshold | Cambio vs Original |
|---------|-----------|-------------------|
| A | 0.8765 | ↓ -0.0461 |
| B | 0.9381 | ↓ -0.0494 |
| C | 0.9275 | ↓ -0.0488 |
| D | 0.9335 | ↓ -0.0491 |
| E | 0.9295 | ↓ -0.0489 |
| F | 0.9350 | ↓ -0.0492 |
| G | 0.9273 | ↓ -0.0488 |
| H | 0.9362 | ↓ -0.0493 |
| I | 0.9336 | ↓ -0.0491 |
| J | 0.9316 | ↓ -0.0490 |
| K | 0.9323 | ↓ -0.0491 |
| L | 0.9344 | ↓ -0.0492 |

## ✅ Recomendaciones

1. **Usar siempre los thresholds optimizados** (`thresholds.json` actualizado)
2. Los thresholds optimizados reducen los falsos negativos
3. El modelo ahora predice correctamente **86.03%** de los alfabetos
4. La mejora principal fue bajar los thresholds de ~0.98 a ~0.93

## 📝 Notas

- El archivo `thresholds.json` ha sido actualizado con los valores optimizados
- Los thresholds originales están guardados en el checkpoint si necesitas revertir
- Para más análisis, ver `analyze_novTest.py` en el directorio raíz


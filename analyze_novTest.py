"""
Script para analizar los errores del modelo novTest y sugerir ajustes de thresholds.
"""

import pandas as pd
import json
from pathlib import Path
from collections import Counter

# Cargar predicciones
print("="*70)
print("ANÁLISIS DE ERRORES - MODELO novTest")
print("="*70)

df_predictions = pd.read_csv('data/dataset3000_predictions_novTest.csv')
print(f"\nTotal de predicciones: {len(df_predictions):,}")

# Cargar thresholds actuales
with open('checkpoints/thresholds_novTest.json', 'r') as f:
    thresholds_data = json.load(f)
    current_thresholds = thresholds_data['per_symbol']

print(f"\nThresholds actuales:")
for symbol, thresh in current_thresholds.items():
    print(f"  {symbol}: {thresh:.4f}")

# Analizar predicciones correctas vs incorrectas
def parse_alphabet(alphabet_str):
    """Parsea string de alfabeto a conjunto."""
    if alphabet_str == '(ninguno)' or alphabet_str == '(no disponible)':
        return set()
    return set(s.strip() for s in alphabet_str.replace(', ', ',').split(',') if s.strip())

df_predictions['pred_set'] = df_predictions['alfabeto_predicho'].apply(parse_alphabet)
df_predictions['real_set'] = df_predictions['alfabeto_real'].apply(parse_alphabet)
df_predictions['correct'] = df_predictions['pred_set'] == df_predictions['real_set']

# Estadísticas generales
total = len(df_predictions)
correct = df_predictions['correct'].sum()
incorrect = total - correct
accuracy = (correct / total) * 100.0

print(f"\n{'='*70}")
print("ESTADÍSTICAS GENERALES")
print("="*70)
print(f"Total: {total:,}")
print(f"Correctas: {correct:,} ({accuracy:.2f}%)")
print(f"Incorrectas: {incorrect:,} ({100.0 - accuracy:.2f}%)")

# Análisis de errores por tipo
print(f"\n{'='*70}")
print("ANÁLISIS DE ERRORES POR TIPO")
print("="*70)

errors_df = df_predictions[~df_predictions['correct']].copy()
errors_df['false_positives'] = errors_df.apply(lambda row: row['pred_set'] - row['real_set'], axis=1)
errors_df['false_negatives'] = errors_df.apply(lambda row: row['real_set'] - row['pred_set'], axis=1)
errors_df['num_fp'] = errors_df['false_positives'].apply(len)
errors_df['num_fn'] = errors_df['false_negatives'].apply(len)

# Contar falsos positivos y negativos por símbolo
fp_by_symbol = Counter()
fn_by_symbol = Counter()

for _, row in errors_df.iterrows():
    for symbol in row['false_positives']:
        fp_by_symbol[symbol] += 1
    for symbol in row['false_negatives']:
        fn_by_symbol[symbol] += 1

print(f"\nFalsos Positivos por símbolo (el modelo predijo pero NO está en el real):")
for symbol in sorted(fp_by_symbol.keys()):
    count = fp_by_symbol[symbol]
    percentage = (count / len(errors_df)) * 100.0
    current_thresh = current_thresholds.get(symbol, 0.5)
    print(f"  {symbol}: {count:,} veces ({percentage:.1f}%) - Threshold actual: {current_thresh:.4f}")

print(f"\nFalsos Negativos por símbolo (NO predijo pero SÍ está en el real):")
for symbol in sorted(fn_by_symbol.keys()):
    count = fn_by_symbol[symbol]
    percentage = (count / len(errors_df)) * 100.0
    current_thresh = current_thresholds.get(symbol, 0.5)
    print(f"  {symbol}: {count:,} veces ({percentage:.1f}%) - Threshold actual: {current_thresh:.4f}")

# Analizar distribución de errores
print(f"\n{'='*70}")
print("DISTRIBUCIÓN DE ERRORES")
print("="*70)

print(f"\nSolo falsos positivos (predijo símbolos que no están):")
only_fp = errors_df[errors_df['num_fp'] > 0][errors_df['num_fn'] == 0]
print(f"  Cantidad: {len(only_fp):,} ({len(only_fp)/len(errors_df)*100:.1f}% de errores)")

print(f"\nSolo falsos negativos (no predijo símbolos que sí están):")
only_fn = errors_df[errors_df['num_fn'] > 0][errors_df['num_fp'] == 0]
print(f"  Cantidad: {len(only_fn):,} ({len(only_fn)/len(errors_df)*100:.1f}% de errores)")

print(f"\nAmbos (falsos positivos Y negativos):")
both = errors_df[(errors_df['num_fp'] > 0) & (errors_df['num_fn'] > 0)]
print(f"  Cantidad: {len(both):,} ({len(both)/len(errors_df)*100:.1f}% de errores)")

# Ejemplos de errores más comunes
print(f"\n{'='*70}")
print("EJEMPLOS DE ERRORES MÁS COMUNES")
print("="*70)

print(f"\nTop 10 errores por falsos negativos (símbolos faltantes):")
fn_examples = errors_df.nlargest(10, 'num_fn')
for i, (idx, row) in enumerate(fn_examples.iterrows(), 1):
    print(f"\n  {i}. Regex: {row['regex']}")
    print(f"     Predicho: {', '.join(sorted(row['pred_set'])) if row['pred_set'] else '(ninguno)'}")
    print(f"     Real:     {', '.join(sorted(row['real_set'])) if row['real_set'] else '(ninguno)'}")
    print(f"     Faltan:   {', '.join(sorted(row['false_negatives']))}")
    print(f"     Sobre:    {', '.join(sorted(row['false_positives'])) if row['false_positives'] else '(ninguno)'}")

# Sugerencias de ajuste de thresholds
print(f"\n{'='*70}")
print("SUGERENCIAS DE AJUSTE DE THRESHOLDS")
print("="*70)

suggested_thresholds = current_thresholds.copy()

print(f"\nAnálisis:")
print(f"  - Falsos Positivos (FP): Símbolos predichos que NO están")
print(f"    → Thresholds muy BAJOS → Aumentar threshold")
print(f"  - Falsos Negativos (FN): Símbolos NO predichos que SÍ están")
print(f"    → Thresholds muy ALTOS → Disminuir threshold")

print(f"\nSugerencias:")
for symbol in sorted(set(list(fp_by_symbol.keys()) + list(fn_by_symbol.keys()))):
    current = current_thresholds.get(symbol, 0.5)
    fp_count = fp_by_symbol.get(symbol, 0)
    fn_count = fn_by_symbol.get(symbol, 0)
    
    suggestion = current
    reason = ""
    
    if fp_count > fn_count * 1.5:
        # Más falsos positivos que negativos → aumentar threshold
        suggestion = min(current * 1.05, 0.99)
        reason = f"Muchos FP ({fp_count}) vs FN ({fn_count})"
    elif fn_count > fp_count * 1.5:
        # Más falsos negativos que positivos → disminuir threshold
        suggestion = max(current * 0.95, 0.3)
        reason = f"Muchos FN ({fn_count}) vs FP ({fp_count})"
    
    if suggestion != current:
        suggested_thresholds[symbol] = suggestion
        print(f"  {symbol}: {current:.4f} → {suggestion:.4f} ({reason})")

# Guardar thresholds sugeridos
suggested_file = Path('checkpoints/thresholds_novTest_suggested.json')
with open(suggested_file, 'w') as f:
    json.dump({'per_symbol': suggested_thresholds}, f, indent=2)

print(f"\n✓ Thresholds sugeridos guardados en: {suggested_file}")

# Comparar thresholds
print(f"\n{'='*70}")
print("COMPARACIÓN DE THRESHOLDS")
print("="*70)
print("\nThresholds actuales vs sugeridos:")
for symbol in sorted(current_thresholds.keys()):
    current = current_thresholds[symbol]
    suggested = suggested_thresholds[symbol]
    diff = suggested - current
    change = "↑" if diff > 0 else "↓" if diff < 0 else "="
    print(f"  {symbol}: {current:.4f} {change} {suggested:.4f} (Δ{diff:+.4f})")

print(f"\n{'='*70}")
print("ANÁLISIS COMPLETADO")
print("="*70)
print("\nPara probar con thresholds sugeridos:")
print("  python demo/test_model.py --checkpoint checkpoints/best_novTest.pt --thresholds checkpoints/thresholds_novTest_suggested.json --csv data/dataset3000.csv --output data/predictions_suggested.csv")


"""
Script para analizar predicciones de regex complejos.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

import torch
import json
import numpy as np

from src.model import AlphabetNet
from src.train import ALPHABET, MAX_PREFIX_LEN, regex_to_indices

# Cargar modelo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint_path = Path('novTest/best (1).pt')
hparams_path = Path('hparams.json')

with open(hparams_path, 'r') as f:
    hparams = json.load(f)

checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

model = AlphabetNet(**hparams['model']).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Cargar thresholds
with open('novTest/thresholds.json', 'r') as f:
    thresholds_data = json.load(f)
    threshold_per_symbol = thresholds_data['per_symbol']

# Regex complejo
regex = "((A+B+((C.D)+E)*) . (F+(G.H+)*) )*  +  ( (I.J+K)* . ( (A+B.C+((D.E)+F)*)+ ) )  +  ( ( (G+H).I* ) . ( (J.K)+L* ) )+"

# Extraer símbolos esperados
expected_symbols = set([c for c in regex if c in ALPHABET])
print("="*70)
print("ANÁLISIS DE REGEX COMPLEJO")
print("="*70)
print(f"\nRegex: {regex}")
print(f"\nSímbolos esperados: {sorted(expected_symbols)} ({len(expected_symbols)} símbolos)")

# Predecir
regex_indices, length = regex_to_indices(regex, MAX_PREFIX_LEN)
regex_indices = regex_indices.unsqueeze(0).to(device)
lengths = torch.tensor([length], dtype=torch.long).to(device)

with torch.no_grad():
    logits = model(regex_indices, lengths, return_logits=True)
    probs = torch.sigmoid(logits).cpu().numpy().flatten()

# Analizar probabilidades
print(f"\n{'='*70}")
print("PROBABILIDADES Y PREDICCIONES")
print("="*70)
print(f"\n{'Símbolo':<10} {'Probabilidad':<15} {'Threshold':<12} {'Predicción':<12} {'Esperado':<10}")
print("-"*70)

predicted_symbols = []
for i, symbol in enumerate(ALPHABET):
    prob = float(probs[i])
    threshold = threshold_per_symbol.get(symbol, 0.5)
    is_predicted = prob >= threshold
    is_expected = symbol in expected_symbols
    
    if is_predicted:
        predicted_symbols.append(symbol)
    
    status = ""
    if is_predicted and is_expected:
        status = "✓ CORRECTO"
    elif is_predicted and not is_expected:
        status = "✗ FP"
    elif not is_predicted and is_expected:
        status = "✗ FN"
    else:
        status = "  OK"
    
    marker = "✓" if is_predicted else " "
    print(f"{marker} {symbol:<8} {prob:<15.6f} {threshold:<12.4f} {is_predicted!s:<12} {status:<10}")

print("-"*70)
print(f"\nAlfabeto predicho: {sorted(predicted_symbols)} ({len(predicted_symbols)} símbolos)")
print(f"Alfabeto esperado: {sorted(expected_symbols)} ({len(expected_symbols)} símbolos)")

# Calcular métricas
correct = len(set(predicted_symbols) & expected_symbols)
false_positives = len(set(predicted_symbols) - expected_symbols)
false_negatives = len(expected_symbols - set(predicted_symbols))

print(f"\n{'='*70}")
print("ANÁLISIS DE RESULTADO")
print("="*70)
print(f"Correctos: {correct}/{len(expected_symbols)}")
print(f"Falsos positivos: {false_positives} (predijo pero no debería)")
print(f"Falsos negativos: {false_negatives} (no predijo pero debería)")

if set(predicted_symbols) == expected_symbols:
    print("\n✅ PREDICCIÓN PERFECTA!")
else:
    print(f"\n⚠️  Predicción incompleta")
    if false_negatives > 0:
        missing = sorted(expected_symbols - set(predicted_symbols))
        print(f"   Faltan: {', '.join(missing)}")
    if false_positives > 0:
        extra = sorted(set(predicted_symbols) - expected_symbols)
        print(f"   Extra: {', '.join(extra)}")


"""
Comparar predicciones con regex simples vs complejos.
"""

import sys
from pathlib import Path
import torch
import json
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.model import AlphabetNet
from src.train import ALPHABET, MAX_PREFIX_LEN, regex_to_indices

# Cargar modelo
device = torch.device('cpu')
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

# Regex simple vs complejo
test_cases = [
    {
        'name': 'Regex Simple: (AB)*C',
        'regex': '(AB)*C',
        'expected': {'A', 'B', 'C'}
    },
    {
        'name': 'Regex Complejo',
        'regex': '((A+B+((C.D)+E)*) . (F+(G.H+)*) )*  +  ( (I.J+K)* . ( (A+B.C+((D.E)+F)*)+ ) )  +  ( ( (G+H).I* ) . ( (J.K)+L* ) )+',
        'expected': {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L'}
    }
]

print("="*70)
print("COMPARACIÓN: REGEX SIMPLE vs COMPLEJO")
print("="*70)

for test_case in test_cases:
    regex = test_case['regex']
    expected = test_case['expected']
    
    # Predecir
    regex_indices, length = regex_to_indices(regex, MAX_PREFIX_LEN)
    regex_indices = regex_indices.unsqueeze(0).to(device)
    lengths = torch.tensor([length], dtype=torch.long).to(device)
    
    with torch.no_grad():
        logits = model(regex_indices, lengths, return_logits=True)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
    
    # Aplicar thresholds
    predicted = []
    probs_dict = {}
    for i, symbol in enumerate(ALPHABET):
        prob = float(probs[i])
        thresh = threshold_per_symbol[symbol]
        probs_dict[symbol] = prob
        if prob >= thresh:
            predicted.append(symbol)
    
    print(f"\n{test_case['name']}")
    print("-"*70)
    print(f"Regex: {regex[:60]}..." if len(regex) > 60 else f"Regex: {regex}")
    print(f"\nEsperado: {sorted(expected)} ({len(expected)} símbolos)")
    print(f"Predicho: {sorted(predicted)} ({len(predicted)} símbolos)")
    
    correct = len(set(predicted) & expected)
    false_positives = len(set(predicted) - expected)
    false_negatives = len(expected - set(predicted))
    
    print(f"\nCorrectos: {correct}/{len(expected)}")
    print(f"Falsos positivos: {false_positives}")
    print(f"Falsos negativos: {false_negatives}")
    
    # Mostrar probabilidades de símbolos esperados
    print(f"\nProbabilidades de símbolos esperados:")
    for symbol in sorted(expected):
        prob = probs_dict[symbol]
        thresh = threshold_per_symbol[symbol]
        status = "✓" if prob >= thresh else "✗"
        diff = prob - thresh
        print(f"  {status} {symbol}: {prob:.6f} (threshold: {thresh:.4f}, diff: {diff:+.6f})")
    
    # Análisis
    if set(predicted) == expected:
        print("\n✅ PREDICCIÓN PERFECTA!")
    else:
        if false_negatives > 0:
            missing = sorted(expected - set(predicted))
            print(f"\n⚠️  Faltan {false_negatives} símbolos: {missing}")
            
            # Ver cuánto faltan
            print("\nSímbolos faltantes - cuánto falta para pasar threshold:")
            for symbol in missing:
                prob = probs_dict[symbol]
                thresh = threshold_per_symbol[symbol]
                diff = prob - thresh
                if diff > -0.1:  # Solo mostrar los que están cerca
                    print(f"  {symbol}: prob={prob:.6f}, thresh={thresh:.4f}, falta={abs(diff):.6f}")
    
    print()

# Conclusión
print("="*70)
print("CONCLUSIÓN")
print("="*70)
print("\nEl problema es una COMBINACIÓN de:")
print("1. Thresholds altos (0.87-0.93) - algunos símbolos están cerca pero no pasan")
print("2. Probabilidades bajas en regex complejo - el modelo no aprende bien patrones complejos")
print("\nPara el regex complejo:")
print("- G está muy cerca (0.9242 vs 0.9273) → threshold puede ayudar")
print("- D, J tienen probabilidades medias (0.75-0.84) → threshold ayudaría algo")
print("- A, B, C, E, F, H, L tienen probabilidades muy bajas (<0.2) → problema del modelo")
print("\nRecomendación:")
print("- Bajar thresholds para casos complejos NO es ideal (reduce precision)")
print("- El modelo necesita más entrenamiento con regex complejos")
print("- O usar thresholds más bajos SOLO si priorizas recall sobre precision")


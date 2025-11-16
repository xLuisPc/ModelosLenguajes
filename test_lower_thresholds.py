"""
Prueba con thresholds más bajos para el regex complejo.
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint_path = Path('novTest/best (1).pt')
hparams_path = Path('hparams.json')

with open(hparams_path, 'r') as f:
    hparams = json.load(f)

checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
model = AlphabetNet(**hparams['model']).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Regex complejo
regex = "((A+B+((C.D)+E)*) . (F+(G.H+)*) )*  +  ( (I.J+K)* . ( (A+B.C+((D.E)+F)*)+ ) )  +  ( ( (G+H).I* ) . ( (J.K)+L* ) )+"
expected_symbols = set([c for c in regex if c in ALPHABET])

# Predecir probabilidades
regex_indices, length = regex_to_indices(regex, MAX_PREFIX_LEN)
regex_indices = regex_indices.unsqueeze(0).to(device)
lengths = torch.tensor([length], dtype=torch.long).to(device)

with torch.no_grad():
    logits = model(regex_indices, lengths, return_logits=True)
    probs = torch.sigmoid(logits).cpu().numpy().flatten()

# Cargar thresholds actuales
with open('novTest/thresholds.json', 'r') as f:
    thresholds_data = json.load(f)
    current_thresholds = thresholds_data['per_symbol']

print("="*70)
print("PRUEBA CON THRESHOLDS MÁS BAJOS")
print("="*70)
print(f"\nRegex: {regex[:80]}...")
print(f"\nSímbolos esperados: {sorted(expected_symbols)} ({len(expected_symbols)} símbolos)")

# Probar con diferentes niveles de thresholds
threshold_levels = {
    'Actuales (0.87-0.93)': current_thresholds,
    'Bajos (0.80-0.90)': {k: v * 0.90 for k, v in current_thresholds.items()},
    'Muy bajos (0.70-0.85)': {k: v * 0.85 for k, v in current_thresholds.items()},
    'Mínimos (0.50)': {k: 0.5 for k in ALPHABET},
}

for level_name, thresholds in threshold_levels.items():
    print(f"\n{'='*70}")
    print(f"{level_name}")
    print("="*70)
    
    predicted = []
    for i, symbol in enumerate(ALPHABET):
        prob = float(probs[i])
        thresh = thresholds[symbol]
        if prob >= thresh:
            predicted.append(symbol)
    
    correct = len(set(predicted) & expected_symbols)
    false_positives = len(set(predicted) - expected_symbols)
    false_negatives = len(expected_symbols - set(predicted))
    
    print(f"\nThresholds: {min(thresholds.values()):.3f} - {max(thresholds.values()):.3f}")
    print(f"Predicho: {sorted(predicted)} ({len(predicted)} símbolos)")
    print(f"Esperado: {sorted(expected_symbols)} ({len(expected_symbols)} símbolos)")
    print(f"\nCorrectos: {correct}/{len(expected_symbols)}")
    print(f"Falsos positivos: {false_positives}")
    print(f"Falsos negativos: {false_negatives}")
    
    if set(predicted) == expected_symbols:
        print("\n✅ PREDICCIÓN PERFECTA!")
        break
    
    # Mostrar símbolos que casi pasan
    print("\nSímbolos cerca del threshold:")
    for i, symbol in enumerate(ALPHABET):
        prob = float(probs[i])
        thresh = thresholds[symbol]
        diff = prob - thresh
        if -0.05 <= diff < 0:
            print(f"  {symbol}: {prob:.4f} (threshold: {thresh:.4f}, falta: {abs(diff):.4f})")


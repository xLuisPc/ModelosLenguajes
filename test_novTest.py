"""
Script rápido para probar el modelo novTest con múltiples regex.
"""

import sys
from pathlib import Path

# Ejecutar demo/test_model.py con el modelo novTest
if __name__ == '__main__':
    import subprocess
    
    # Test con varios regex
    test_regexes = [
        "(AB)*C",
        "A+B*",
        "[ABCD]+",
        "[LCIG]+",
        "[GDIK]*",
        "H+C*J+BF|DL*",
        "I+B*|AA*",
        "[KBG]*",
        "[FEK]+"
    ]
    
    print("="*70)
    print("PROBANDO MODELO novTest CON MÚLTIPLES REGEX")
    print("="*70)
    print()
    
    for regex in test_regexes:
        cmd = [
            sys.executable,
            'demo/test_model.py',
            '--checkpoint', 'checkpoints/best_novTest.pt',
            '--thresholds', 'checkpoints/thresholds_novTest.json',
            '--regex', regex
        ]
        print(f"\n{'='*70}")
        print(f"Probando: {regex}")
        print('='*70)
        subprocess.run(cmd)
    
    print("\n" + "="*70)
    print("PRUEBAS COMPLETADAS")
    print("="*70)


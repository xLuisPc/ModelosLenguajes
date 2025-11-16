#!/usr/bin/env python
"""
Script rápido para probar modelos AlphabetNet.
Ejecuta demo/test_model.py con argumentos predeterminados.
"""

import sys
from pathlib import Path

# Ejecutar demo/test_model.py
if __name__ == '__main__':
    import subprocess
    
    # Obtener argumentos pasados
    args = sys.argv[1:]
    
    # Si no hay argumentos, mostrar ayuda
    if not args or '-h' in args or '--help' in args:
        print("Script rápido para probar modelos AlphabetNet\n")
        print("Uso:")
        print("  python test.py [opciones]")
        print("\nEjemplos:")
        print("  # Modo interactivo")
        print("  python test.py")
        print("\n  # Predicción de un regex")
        print('  python test.py --regex "(AB)*C"')
        print("\n  # Con thresholds personalizados")
        print("  python test.py --thresholds checkpoints/thresholds.json --regex 'A+B*'")
        print("\nPara más opciones, ejecuta:")
        print("  python demo/test_model.py --help")
        print()
        
        # Si no hay argumentos, ejecutar modo interactivo
        if not args or '-h' in args or '--help' in args:
            import sys
            sys.exit(0)
    
    # Ejecutar demo/test_model.py con los argumentos
    script_path = Path(__file__).parent / 'demo' / 'test_model.py'
    
    # Agregar checkpoint por defecto si no se especifica
    if '--checkpoint' not in args:
        default_checkpoint = Path('checkpoints/best.pt')
        if default_checkpoint.exists():
            args.insert(0, str(default_checkpoint))
            args.insert(0, '--checkpoint')
        else:
            print("❌ Error: No se encontró checkpoints/best.pt")
            print("   Por favor, especifica el checkpoint con --checkpoint")
            print("   Ejemplo: python test.py --checkpoint checkpoints/best.pt")
            sys.exit(1)
    
    # Ejecutar
    cmd = [sys.executable, str(script_path)] + args
    sys.exit(subprocess.call(cmd))

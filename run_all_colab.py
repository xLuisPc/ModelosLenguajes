"""
Script para ejecutar todo el pipeline en Google Colab.

Ejecuta en orden:
1. Entrenamiento
2. Curvas PR y reporte
3. Búsqueda de umbrales
4. Exportación a ONNX
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Ejecuta un comando y muestra el resultado."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}\n")
    print(f"Comando: {cmd}\n")
    
    result = subprocess.run(cmd.split(), capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print("Errores:", result.stderr, file=sys.stderr)
    
    if result.returncode != 0:
        print(f"\n❌ Error ejecutando: {description}")
        return False
    
    print(f"\n✓ {description} completado")
    return True

def main():
    """Ejecuta todo el pipeline."""
    
    # Verificar que los archivos necesarios existen
    required_files = [
        'model.py',
        'metrics.py',
        'train.py',
        'generate_pr_curves.py',
        'find_thresholds.py',
        'export_model.py',
        'hparams.json',
        'data/alphabet/train_wide.parquet',
        'data/alphabet/val_wide.parquet'
    ]
    
    print("Verificando archivos necesarios...")
    missing = []
    for f in required_files:
        if not Path(f).exists():
            missing.append(f)
    
    if missing:
        print("❌ Archivos faltantes:")
        for f in missing:
            print(f"  - {f}")
        print("\nPor favor, sube todos los archivos necesarios a Colab.")
        return
    
    print("✓ Todos los archivos necesarios están presentes\n")
    
    # Pipeline de ejecución
    pipeline = [
        (
            "python train.py --train_data data/alphabet/train_wide.parquet --val_data data/alphabet/val_wide.parquet --checkpoint_dir checkpoints",
            "1. Entrenamiento del modelo"
        ),
        (
            "python generate_pr_curves.py --checkpoint checkpoints/best.pt --val_data data/alphabet/val_wide.parquet --output_dir checkpoints",
            "2. Generación de curvas PR y reporte"
        ),
        (
            "python find_thresholds.py --checkpoint checkpoints/best.pt --val_data data/alphabet/val_wide.parquet --output_dir checkpoints",
            "3. Búsqueda de umbrales óptimos"
        ),
        (
            "python export_model.py --checkpoint checkpoints/best.pt --output alphabetnet.onnx",
            "4. Exportación a ONNX"
        )
    ]
    
    # Ejecutar pipeline
    for cmd, desc in pipeline:
        if not run_command(cmd, desc):
            print(f"\n❌ Pipeline detenido en: {desc}")
            return
    
    # Verificar archivos generados
    print(f"\n{'='*60}")
    print("VERIFICACIÓN DE ARCHIVOS GENERADOS")
    print(f"{'='*60}\n")
    
    expected_files = {
        'checkpoints/best.pt': 'Mejor checkpoint',
        'checkpoints/last.pt': 'Último checkpoint',
        'checkpoints/train_log.csv': 'Log de entrenamiento',
        'checkpoints/pr_macro.png': 'Curva PR macro',
        'checkpoints/pr_top10.png': 'Curvas PR top-10',
        'checkpoints/per_symbol_ap.csv': 'AP por símbolo',
        'checkpoints/A2_report.md': 'Reporte de validación',
        'checkpoints/thresholds.json': 'Umbrales óptimos',
        'checkpoints/threshold_eval.csv': 'Evaluación de umbrales',
        'alphabetnet.onnx': 'Modelo ONNX exportado'
    }
    
    for filepath, description in expected_files.items():
        path = Path(filepath)
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"✓ {description}: {filepath} ({size_mb:.2f} MB)")
        else:
            print(f"✗ {description}: {filepath} (no encontrado)")
    
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETADO")
    print(f"{'='*60}\n")
    print("Archivos listos para descargar:")
    print("  - alphabetnet.onnx (modelo ONNX)")
    print("  - checkpoints/best.pt (mejor modelo)")
    print("  - checkpoints/A2_report.md (reporte)")
    print("  - checkpoints/*.png (gráficos)")
    print("  - checkpoints/*.csv (métricas)")

if __name__ == '__main__':
    main()


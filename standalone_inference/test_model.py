"""
Interfaz interactiva para probar modelos AlphabetNet entrenados.

Características:
- Carga modelo desde checkpoint
- Interfaz CLI simple para probar regexes
- Muestra probabilidades y alfabeto predicho
- Soporte para thresholds personalizados
- Procesa CSV completo y genera predicciones
"""

import json
import sys
from pathlib import Path
from typing import Dict, Optional, List

import pandas as pd
import torch
import numpy as np

# Agregar src al path para imports
root = Path(__file__).parent  # Directorio actual (standalone_inference)
sys.path.insert(0, str(root / 'src'))

from model import AlphabetNet
from train import ALPHABET, MAX_PREFIX_LEN, regex_to_indices


def load_model(checkpoint_path: Path, hparams_path: Path, device: torch.device):
    """Carga el modelo desde checkpoint."""
    # Cargar hiperparámetros
    with open(hparams_path, 'r') as f:
        hparams = json.load(f)
    
    # Cargar checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Crear modelo
    model = AlphabetNet(
        vocab_size=hparams['model']['vocab_size'],
        alphabet_size=hparams['model']['alphabet_size'],
        emb_dim=hparams['model']['emb_dim'],
        hidden_dim=hparams['model']['hidden_dim'],
        rnn_type=hparams['model']['rnn_type'],
        num_layers=hparams['model']['num_layers'],
        dropout=hparams['model']['dropout'],
        padding_idx=hparams['model']['padding_idx'],
        use_automata_conditioning=hparams['model']['use_automata_conditioning'],
        num_automata=hparams['model'].get('num_automata'),
        automata_emb_dim=hparams['model']['automata_emb_dim']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Obtener métricas del checkpoint
    metrics = {
        'epoch': checkpoint.get('epoch', 'N/A'),
        'f1_macro': checkpoint.get('f1_macro', 'N/A'),
        'f1_min': checkpoint.get('f1_min', 'N/A'),
        'ece': checkpoint.get('ece', 'N/A')
    }
    
    return model, metrics


def load_thresholds(thresholds_path: Optional[Path]) -> Optional[Dict[str, float]]:
    """Carga thresholds desde archivo JSON."""
    if thresholds_path is None or not thresholds_path.exists():
        return None
    
    with open(thresholds_path, 'r') as f:
        data = json.load(f)
        return data.get('per_symbol', {})


def predict_alphabet(model: AlphabetNet, regex: str, device: torch.device,
                     threshold_per_symbol: Optional[Dict[str, float]] = None) -> Dict:
    """Predice alfabeto desde regex."""
    if threshold_per_symbol is None:
        threshold_per_symbol = {sym: 0.5 for sym in ALPHABET}
    
    # Convertir regex a índices
    regex_indices, length = regex_to_indices(regex, MAX_PREFIX_LEN)
    regex_indices = regex_indices.unsqueeze(0).to(device)
    lengths = torch.tensor([length], dtype=torch.long).to(device)
    
    # Forward pass
    with torch.no_grad():
        logits = model(regex_indices, lengths, return_logits=True)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
    
    # Construir sigma_hat
    sigma_hat = []
    p_sigma_dict = {}
    
    for i, symbol in enumerate(ALPHABET):
        prob = float(probs[i])
        threshold = threshold_per_symbol.get(symbol, 0.5)
        p_sigma_dict[symbol] = prob
        if prob >= threshold:
            sigma_hat.append(symbol)
    
    return {
        'p_sigma': p_sigma_dict,
        'sigma_hat': sigma_hat
    }


def format_result(regex: str, result: Dict, threshold_per_symbol: Dict[str, float]):
    """Formatea resultado para mostrar."""
    print("\n" + "="*70)
    print("PREDICCIÓN DE ALFABETO DESDE REGEX")
    print("="*70)
    print(f"Regex: '{regex}'")
    print("\nProbabilidades por símbolo:")
    print("-"*70)
    print(f"{'Símbolo':<10} {'Probabilidad':<15} {'Threshold':<12} {'Predicción':<12}")
    print("-"*70)
    
    for symbol in ALPHABET:
        prob = result['p_sigma'][symbol]
        threshold = threshold_per_symbol.get(symbol, 0.5)
        predicted = "✓ SÍ" if prob >= threshold else "  NO"
        print(f"{symbol:<10} {prob:<15.6f} {threshold:<12.4f} {predicted:<12}")
    
    print("-"*70)
    print(f"\nAlfabeto predicho (sigma_hat): {', '.join(result['sigma_hat']) if result['sigma_hat'] else '(ninguno)'}")
    print("="*70 + "\n")


def extract_alphabet_from_string(alphabet_str: str) -> List[str]:
    """
    Extrae el conjunto de símbolos del alfabeto desde un string.
    
    Args:
        alphabet_str: String con símbolos separados por espacios (ej: "A B C D")
    
    Returns:
        Lista ordenada de símbolos del alfabeto
    """
    # Dividir por espacios y filtrar símbolos válidos
    symbols = [s.strip() for s in str(alphabet_str).split() if s.strip() in ALPHABET]
    return sorted(set(symbols))  # Ordenar y eliminar duplicados


def process_csv(model: AlphabetNet, device: torch.device, 
                csv_path: Path, output_path: Path,
                threshold_per_symbol: Optional[Dict[str, float]] = None):
    """
    Procesa un CSV completo y genera predicciones.
    
    Args:
        model: Modelo AlphabetNet entrenado
        device: Dispositivo (CPU/GPU)
        csv_path: Path al CSV de entrada (dataset3000.csv)
        output_path: Path donde guardar el CSV de salida
        threshold_per_symbol: Thresholds por símbolo (opcional)
    """
    print("\n" + "="*70)
    print("PROCESANDO CSV COMPLETO")
    print("="*70)
    
    # Cargar CSV
    print(f"\n📂 Cargando CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✓ CSV cargado: {len(df):,} filas")
    print(f"   Columnas: {list(df.columns)}")
    
    # Verificar que tiene las columnas necesarias
    if 'Regex' not in df.columns:
        raise ValueError("El CSV debe tener una columna 'Regex'")
    
    # Procesar cada fila
    print("\n🔄 Procesando predicciones...")
    results = []
    
    for idx, row in df.iterrows():
        regex = str(row['Regex']).strip()
        
        # Predecir alfabeto con el modelo
        result = predict_alphabet(model, regex, device, threshold_per_symbol)
        alfabeto_predicho = sorted(result['sigma_hat'])
        
        # Extraer alfabeto real desde la columna 'Alfabeto' (si existe)
        alfabeto_real = []
        if 'Alfabeto' in df.columns:
            alphabet_str = str(row['Alfabeto']).strip()
            alfabeto_real = extract_alphabet_from_string(alphabet_str)
        
        # Guardar resultado
        results.append({
            'regex': regex,
            'alfabeto_predicho': ', '.join(alfabeto_predicho) if alfabeto_predicho else '(ninguno)',
            'alfabeto_real': ', '.join(alfabeto_real) if alfabeto_real else '(no disponible)'
        })
        
        # Mostrar progreso cada 100 filas
        if (idx + 1) % 100 == 0:
            print(f"   Procesadas {idx + 1:,}/{len(df):,} filas...")
    
    # Crear DataFrame de resultados
    df_results = pd.DataFrame(results)
    
    # Guardar CSV
    print(f"\n💾 Guardando resultados en: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(output_path, index=False, encoding='utf-8')
    print(f"✓ CSV guardado: {output_path}")
    print(f"   Total de filas: {len(df_results):,}")
    
    # Estadísticas
    print("\n📊 Estadísticas:")
    if 'Alfabeto' in df.columns:
        # Calcular exactitud de conjunto (exact match)
        exact_matches = 0
        for i, row in df_results.iterrows():
            pred_set = set(row['alfabeto_predicho'].replace('(ninguno)', '').replace(', ', ',').split(','))
            pred_set = {s.strip() for s in pred_set if s.strip() and s.strip() != '(ninguno)'}
            real_set = set(row['alfabeto_real'].replace('(no disponible)', '').replace(', ', ',').split(','))
            real_set = {s.strip() for s in real_set if s.strip() and s.strip() != '(no disponible)'}
            
            if pred_set == real_set:
                exact_matches += 1
        
        accuracy = (exact_matches / len(df_results)) * 100.0
        print(f"   Exactitud de conjunto: {exact_matches:,}/{len(df_results):,} ({accuracy:.2f}%)")
    
    print("="*70 + "\n")


def interactive_mode(model: AlphabetNet, device: torch.device, 
                     threshold_per_symbol: Dict[str, float]):
    """Modo interactivo."""
    print("\n" + "="*70)
    print("MODO INTERACTIVO - AlphabetNet")
    print("="*70)
    print("Ingresa regexes para predecir alfabetos.")
    print("Comandos:")
    print("  - Escribe un regex y presiona Enter para predecir")
    print("  - Escribe 'quit' o 'exit' para salir")
    print("  - Escribe 'clear' para limpiar pantalla")
    print("="*70 + "\n")
    
    while True:
        try:
            regex = input("Regex: ").strip()
            
            if regex.lower() in ['quit', 'exit', 'q']:
                print("¡Hasta luego!")
                break
            
            if regex.lower() == 'clear':
                import os
                os.system('cls' if os.name == 'nt' else 'clear')
                continue
            
            if not regex:
                continue
            
            result = predict_alphabet(model, regex, device, threshold_per_symbol)
            format_result(regex, result, threshold_per_symbol)
            
        except KeyboardInterrupt:
            print("\n\n¡Hasta luego!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


def main():
    """Función principal."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Interfaz interactiva para probar modelos AlphabetNet',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Modo interactivo
  python test_model.py --checkpoint novTest/alphabetnet.pt
  
  # Con thresholds personalizados
  python test_model.py --checkpoint novTest/alphabetnet.pt --thresholds novTest/thresholds.json
  
  # Predicción de un solo regex
  python test_model.py --checkpoint novTest/alphabetnet.pt --thresholds novTest/thresholds.json --regex "((A+B+((C.D)+E)*)"
  
  # Procesar CSV completo
  python test_model.py --checkpoint novTest/alphabetnet.pt --csv dataset.csv --output predictions.csv
        """
    )
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path al checkpoint del modelo (ej: checkpoints/best.pt)')
    parser.add_argument('--hparams', type=str, default='hparams.json',
                       help='Path al archivo de hiperparámetros (default: hparams.json)')
    parser.add_argument('--thresholds', type=str, default=None,
                       help='Path al archivo JSON con thresholds por símbolo')
    parser.add_argument('--regex', type=str, default=None,
                       help='Regex para predecir (si no se especifica, modo interactivo)')
    parser.add_argument('--csv', type=str, default=None,
                       help='Path al CSV de entrada (dataset3000.csv) para procesar completo')
    parser.add_argument('--output', type=str, default=None,
                       help='Path donde guardar el CSV de salida (solo con --csv)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Dispositivo: cpu, cuda, o auto (default: auto)')
    
    args = parser.parse_args()
    
    # Resolver paths
    root = Path(__file__).parent  # Directorio actual (standalone_inference)
    checkpoint_path = root / args.checkpoint
    hparams_path = root / args.hparams
    thresholds_path = root / args.thresholds if args.thresholds else None
    
    # Validar archivos
    if not checkpoint_path.exists():
        print(f"❌ Error: Checkpoint no encontrado: {checkpoint_path}")
        sys.exit(1)
    
    if not hparams_path.exists():
        print(f"❌ Error: Hiperparámetros no encontrados: {hparams_path}")
        sys.exit(1)
    
    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"📱 Usando dispositivo: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Cargar modelo
    print(f"\n📦 Cargando modelo desde: {checkpoint_path}")
    model, metrics = load_model(checkpoint_path, hparams_path, device)
    print("✓ Modelo cargado")
    print(f"   Época: {metrics['epoch']}")
    print(f"   F1 macro: {metrics['f1_macro']}")
    print(f"   F1 min: {metrics['f1_min']}")
    print(f"   ECE: {metrics['ece']}")
    
    # Cargar thresholds
    threshold_per_symbol = None
    if thresholds_path and thresholds_path.exists():
        print(f"\n📊 Cargando thresholds desde: {thresholds_path}")
        threshold_per_symbol = load_thresholds(thresholds_path)
        print("✓ Thresholds cargados")
        print(f"   Thresholds: {threshold_per_symbol}")
    else:
        print(f"\n⚠️  No se encontraron thresholds, usando 0.5 para todos los símbolos")
        threshold_per_symbol = {sym: 0.5 for sym in ALPHABET}
    
    print()
    
    # Ejecutar según modo
    if args.csv:
        # Modo procesar CSV completo
        csv_path = root / args.csv
        if not csv_path.exists():
            print(f"❌ Error: CSV no encontrado: {csv_path}")
            sys.exit(1)
        
        if args.output:
            output_path = root / args.output
        else:
            # Generar nombre de salida automático
            output_path = csv_path.parent / f"{csv_path.stem}_predictions.csv"
        
        process_csv(model, device, csv_path, output_path, threshold_per_symbol)
        
    elif args.regex:
        # Modo de un solo regex
        result = predict_alphabet(model, args.regex, device, threshold_per_symbol)
        format_result(args.regex, result, threshold_per_symbol)
    else:
        # Modo interactivo
        interactive_mode(model, device, threshold_per_symbol)


if __name__ == '__main__':
    main()

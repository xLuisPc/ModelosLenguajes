"""
Script de inferencia rápida para AlphabetNet.

Características:
- Carga modelo desde checkpoint
- Recibe automata_id (opcional) + prefix
- Devuelve top-k símbolos con probabilidades
- Útil para QA y pruebas rápidas
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from model import AlphabetNet
from train import ALPHABET, MAX_PREFIX_LEN, regex_to_indices, char_to_idx

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device, 
                                hparams: dict) -> AlphabetNet:
    """Carga el modelo desde un checkpoint."""
    logger.info(f"Cargando checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
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
    
    epoch = checkpoint.get('epoch', 'unknown')
    logger.info(f"✓ Modelo cargado (época {epoch})")
    
    return model


def infer_one_regex(model: AlphabetNet, regex: str, device: torch.device,
                    automata_id: Optional[int] = None) -> np.ndarray:
    """
    Infiere probabilidades de símbolos para un regex.
    
    Args:
        model: Modelo AlphabetNet
        regex: String del regex del autómata
        device: Dispositivo (CPU/GPU)
        automata_id: ID del autómata (opcional, solo si use_automata_conditioning=True)
    
    Returns:
        Array numpy de forma (alphabet_size,) con probabilidades p_sigma
    """
    # Convertir regex a índices
    regex_indices, length = regex_to_indices(regex, MAX_PREFIX_LEN)
    
    # Preparar tensores para batch de tamaño 1
    regex_indices = regex_indices.unsqueeze(0).to(device)  # (1, max_len)
    lengths = torch.tensor([length], dtype=torch.long).to(device)  # (1,)
    
    # Preparar automata_id si está disponible
    automata_ids = None
    if automata_id is not None and model.use_automata_conditioning:
        automata_ids = torch.tensor([automata_id], dtype=torch.long).to(device)  # (1,)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(regex_indices, lengths, automata_ids=automata_ids, return_logits=True)
        probs = torch.sigmoid(logits)  # (1, alphabet_size)
    
    # Convertir a numpy
    probs_np = probs.cpu().numpy().flatten()  # (alphabet_size,)
    
    return probs_np


def predict_alphabet_from_regex(regex: str, model: AlphabetNet, device: torch.device,
                                threshold_per_symbol: Optional[Dict[str, float]] = None,
                                alphabet: List[str] = ALPHABET) -> Dict[str, any]:
    """
    Predice el alfabeto del autómata desde un regex.
    
    Args:
        regex: String del regex del autómata
        model: Modelo AlphabetNet entrenado
        device: Dispositivo (CPU/GPU)
        threshold_per_symbol: Dict con umbrales por símbolo: {symbol: threshold}
                             Si None, usa 0.5 para todos
        alphabet: Lista de nombres de símbolos
    
    Returns:
        Dict con:
        {
            'p_sigma': Lista de probabilidades por símbolo [p_A, p_B, ..., p_L],
            'sigma_hat': Lista de símbolos con probabilidad >= threshold
        }
    """
    if threshold_per_symbol is None:
        threshold_per_symbol = {sym: 0.5 for sym in alphabet}
    
    # Inferir probabilidades
    p_sigma = infer_one_regex(model, regex, device)
    
    # Construir sigma_hat usando thresholds por símbolo
    sigma_hat = []
    for i, symbol in enumerate(alphabet):
        threshold = threshold_per_symbol.get(symbol, 0.5)
        if p_sigma[i] >= threshold:
            sigma_hat.append(symbol)
    
    return {
        'p_sigma': p_sigma.tolist(),
        'sigma_hat': sigma_hat
    }


def predict_top_k(model: AlphabetNet, prefix: str, device: torch.device,
                  automata_id: Optional[int] = None, top_k: int = 5,
                  alphabet: List[str] = ALPHABET) -> List[Tuple[str, float]]:
    """
    Predice top-k símbolos más probables después de un prefijo.
    (Función legacy para compatibilidad)
    
    Args:
        model: Modelo AlphabetNet
        prefix: Prefijo string (puede ser '<EPS>' o cadena de caracteres A-L)
        device: Dispositivo (CPU/GPU)
        automata_id: ID del autómata (opcional, solo si use_automata_conditioning=True)
        top_k: Número de símbolos top a retornar
        alphabet: Lista de nombres de símbolos
    
    Returns:
        Lista de tuplas (symbol, probability) ordenadas por probabilidad descendente
    """
    # Usar regex_to_indices (es un alias de prefix_to_indices)
    regex_indices, length = regex_to_indices(prefix, MAX_PREFIX_LEN)
    
    # Preparar tensores para batch de tamaño 1
    regex_indices = regex_indices.unsqueeze(0).to(device)  # (1, max_len)
    lengths = torch.tensor([length], dtype=torch.long).to(device)  # (1,)
    
    # Preparar automata_id si está disponible
    automata_ids = None
    if automata_id is not None and model.use_automata_conditioning:
        automata_ids = torch.tensor([automata_id], dtype=torch.long).to(device)  # (1,)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(regex_indices, lengths, automata_ids=automata_ids, return_logits=True)
        probs = torch.sigmoid(logits)  # (1, alphabet_size)
    
    # Convertir a numpy
    probs_np = probs.cpu().numpy().flatten()  # (alphabet_size,)
    
    # Obtener top-k índices
    top_k_idx = np.argsort(probs_np)[::-1][:top_k]  # Orden descendente
    
    # Crear lista de resultados
    results = [(alphabet[idx], float(probs_np[idx])) for idx in top_k_idx]
    
    return results


def format_output(results: List[Tuple[str, float]], prefix: str, 
                  automata_id: Optional[int] = None) -> str:
    """Formatea la salida de predicción."""
    output = []
    
    output.append("="*60)
    output.append("PREDICCIÓN")
    output.append("="*60)
    
    if automata_id is not None:
        output.append(f"Autómata ID: {automata_id}")
    output.append(f"Prefijo: '{prefix}'")
    output.append("")
    output.append("Top-k símbolos más probables:")
    output.append("-"*60)
    
    for i, (symbol, prob) in enumerate(results, 1):
        output.append(f"{i}. {symbol}: {prob:.6f} ({prob*100:.2f}%)")
    
    output.append("="*60)
    
    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(
        description='Inferencia rápida para AlphabetNet',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Predicción básica desde regex
  python infer.py --checkpoint checkpoints/best.pt --regex "(AB)*C"
  
  # Con thresholds personalizados
  python infer.py --checkpoint checkpoints/best.pt --regex "(AB)*C" --thresholds checkpoints/thresholds.json
  
  # Predicción legacy con prefijo (compatibilidad hacia atrás)
  python infer.py --checkpoint checkpoints/best.pt --prefix "ABC" --top_k 10
        """
    )
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path al checkpoint del modelo')
    parser.add_argument('--hparams', type=str, default='hparams.json',
                       help='Path al archivo de hiperparámetros')
    parser.add_argument('--regex', type=str, default=None,
                       help='Regex del autómata (nueva API)')
    parser.add_argument('--prefix', type=str, default=None,
                       help='Prefijo string (legacy, usar --regex en su lugar)')
    parser.add_argument('--thresholds', type=str, default=None,
                       help='Path al archivo JSON con thresholds por símbolo')
    parser.add_argument('--automata_id', type=int, default=None,
                       help='ID del autómata (opcional, solo si use_automata_conditioning=True)')
    parser.add_argument('--top_k', type=int, default=5,
                       help='Número de símbolos top a retornar (default: 5, solo para --prefix)')
    parser.add_argument('--output', type=str, default=None,
                       help='Archivo de salida (opcional, si no se especifica imprime a stdout)')
    
    args = parser.parse_args()
    
    # Cargar hiperparámetros
    hparams_path = Path(args.hparams)
    if not hparams_path.exists():
        raise FileNotFoundError(f"Archivo de hiperparámetros no encontrado: {hparams_path}")
    
    with open(hparams_path, 'r') as f:
        hparams = json.load(f)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Usando dispositivo: {device}")
    
    # Cargar modelo
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint no encontrado: {checkpoint_path}")
    
    model = load_model_from_checkpoint(checkpoint_path, device, hparams)
    
    # Cargar thresholds si se especifica
    threshold_per_symbol = None
    if args.thresholds:
        thresholds_path = Path(args.thresholds)
        if thresholds_path.exists():
            with open(thresholds_path, 'r') as f:
                thresholds_data = json.load(f)
                threshold_per_symbol = thresholds_data.get('per_symbol', {})
                logger.info(f"✓ Thresholds cargados desde: {thresholds_path}")
    
    # Determinar modo: regex (nuevo) o prefix (legacy)
    if args.regex:
        # Modo nuevo: regex → alfabeto
        regex = args.regex.strip()
        logger.info(f"Prediciendo alfabeto desde regex: '{regex}'")
        
        result = predict_alphabet_from_regex(
            regex,
            model,
            device,
            threshold_per_symbol=threshold_per_symbol,
            alphabet=ALPHABET
        )
        
        # Formatear salida
        output_lines = []
        output_lines.append("="*60)
        output_lines.append("PREDICCIÓN DE ALFABETO DESDE REGEX")
        output_lines.append("="*60)
        output_lines.append(f"Regex: '{regex}'")
        output_lines.append("")
        output_lines.append("Probabilidades por símbolo (p_sigma):")
        output_lines.append("-"*60)
        for i, symbol in enumerate(ALPHABET):
            prob = result['p_sigma'][i]
            threshold = threshold_per_symbol.get(symbol, 0.5) if threshold_per_symbol else 0.5
            marker = "✓" if prob >= threshold else " "
            output_lines.append(f"{marker} {symbol}: {prob:.6f} (threshold: {threshold:.4f})")
        output_lines.append("")
        output_lines.append(f"Símbolos predichos (sigma_hat): {', '.join(result['sigma_hat'])}")
        output_lines.append("="*60)
        
        output = "\n".join(output_lines)
        
    elif args.prefix:
        # Modo legacy: prefix → top-k símbolos
        prefix = args.prefix.strip()
        if prefix == '' or prefix == '<EPS>':
            prefix = '<EPS>'
        
        # Validar prefijo
        if prefix != '<EPS>':
            for char in prefix:
                if char not in ALPHABET:
                    raise ValueError(f"Carácter inválido en prefijo: '{char}'. Solo se permiten caracteres A-L o '<EPS>'.")
        
        logger.info(f"Prediciendo símbolos para prefijo: '{prefix}'")
        if args.automata_id is not None:
            logger.info(f"Usando autómata ID: {args.automata_id}")
        
        results = predict_top_k(
            model,
            prefix,
            device,
            automata_id=args.automata_id,
            top_k=args.top_k,
            alphabet=ALPHABET
        )
        
        # Formatear salida
        output = format_output(results, prefix, args.automata_id)
    else:
        raise ValueError("Debe especificar --regex o --prefix")
    
    # Guardar o imprimir
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(output)
        logger.info(f"✓ Resultados guardados en: {output_path}")
    else:
        print(output)


if __name__ == '__main__':
    main()


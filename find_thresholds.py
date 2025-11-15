"""
Script para encontrar umbrales óptimos de decisión por símbolo.

Características:
- Busca umbral por símbolo que maximiza F1 score
- Compara con umbral global (0.5)
- Evalúa impacto (ΔF1, Δcoverage)
- Exporta thresholds.json y threshold_eval.csv
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score

from model import AlphabetNet
from train import AlphabetDataset, collate_fn, ALPHABET, MAX_PREFIX_LEN

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device, hparams: dict) -> AlphabetNet:
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


def find_optimal_threshold_f1(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float]:
    """
    Encuentra el umbral que maximiza F1 score.
    
    Args:
        y_true: Array 1D con etiquetas binarias
        y_scores: Array 1D con scores/probabilidades
    
    Returns:
        Tuple (optimal_threshold, best_f1)
    """
    # Calcular curva PR
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    
    # Calcular F1 para cada threshold
    # F1 = 2 * (precision * recall) / (precision + recall)
    # Nota: precision y recall tienen un elemento más que thresholds (último punto con recall=0)
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)
    
    # Encontrar índice del máximo F1
    best_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    return optimal_threshold, best_f1


def find_thresholds_per_symbol(y_true: np.ndarray, y_scores: np.ndarray,
                                alphabet: list) -> Dict[str, float]:
    """
    Encuentra umbral óptimo por símbolo que maximiza F1.
    
    Args:
        y_true: Array de forma (n_samples, n_symbols) con etiquetas binarias
        y_scores: Array de forma (n_samples, n_symbols) con scores/probabilidades
        alphabet: Lista de nombres de símbolos
    
    Returns:
        Dict con umbral óptimo por símbolo: {symbol: threshold}
    """
    thresholds = {}
    
    for i, symbol in enumerate(alphabet):
        y_true_symbol = y_true[:, i]
        y_scores_symbol = y_scores[:, i]
        
        # Solo calcular si hay positivos
        if y_true_symbol.sum() > 0:
            try:
                optimal_threshold, best_f1 = find_optimal_threshold_f1(
                    y_true_symbol, y_scores_symbol
                )
                thresholds[symbol] = float(optimal_threshold)
                logger.debug(f"  {symbol}: threshold={optimal_threshold:.4f}, F1={best_f1:.4f}")
            except Exception as e:
                logger.warning(f"  {symbol}: Error al calcular umbral óptimo: {e}")
                # Usar umbral por defecto si hay error
                thresholds[symbol] = 0.5
        else:
            # No hay positivos, usar umbral por defecto
            thresholds[symbol] = 0.5
            logger.debug(f"  {symbol}: Sin positivos, usando threshold=0.5")
    
    return thresholds


def evaluate_with_thresholds(y_true: np.ndarray, y_scores: np.ndarray,
                              thresholds: Dict[str, float], alphabet: list,
                              threshold_type: str = 'per_symbol') -> Dict[str, Dict[str, float]]:
    """
    Evalúa métricas por símbolo usando umbrales específicos.
    
    Args:
        y_true: Array de forma (n_samples, n_symbols) con etiquetas binarias
        y_scores: Array de forma (n_samples, n_symbols) con scores/probabilidades
        thresholds: Dict con umbrales por símbolo o umbral global
        alphabet: Lista de nombres de símbolos
        threshold_type: 'per_symbol' o 'global'
    
    Returns:
        Dict con métricas por símbolo: {symbol: {'precision': ..., 'recall': ..., 'f1': ...}}
    """
    metrics_per_symbol = {}
    
    for i, symbol in enumerate(alphabet):
        y_true_symbol = y_true[:, i]
        y_scores_symbol = y_scores[:, i]
        
        # Obtener umbral
        if threshold_type == 'global':
            threshold = thresholds.get('global', 0.5)
        else:
            threshold = thresholds.get(symbol, 0.5)
        
        # Binarizar predicciones
        y_pred_symbol = (y_scores_symbol >= threshold).astype(int)
        
        # Calcular métricas
        if y_true_symbol.sum() > 0:
            precision = precision_score(y_true_symbol, y_pred_symbol, zero_division=0)
            recall = recall_score(y_true_symbol, y_pred_symbol, zero_division=0)
            f1 = f1_score(y_true_symbol, y_pred_symbol, zero_division=0)
        else:
            # No hay positivos
            precision = 0.0
            recall = 0.0
            f1 = 0.0
        
        metrics_per_symbol[symbol] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'threshold': float(threshold)
        }
    
    return metrics_per_symbol


def compute_coverage(metrics_per_symbol: Dict[str, Dict[str, float]]) -> float:
    """
    Calcula coverage: porcentaje de símbolos con F1 > 0 (es decir, con predicciones positivas).
    
    Args:
        metrics_per_symbol: Dict con métricas por símbolo
    
    Returns:
        Porcentaje de símbolos con F1 > 0
    """
    total_symbols = len(metrics_per_symbol)
    if total_symbols == 0:
        return 0.0
    
    symbols_with_predictions = sum(
        1 for m in metrics_per_symbol.values() if m.get('f1', 0) > 0 or m.get('recall', 0) > 0
    )
    
    return (symbols_with_predictions / total_symbols) * 100.0


def compute_average_f1(metrics_per_symbol: Dict[str, Dict[str, float]]) -> float:
    """
    Calcula F1 promedio (macro) sobre todos los símbolos.
    
    Args:
        metrics_per_symbol: Dict con métricas por símbolo
    
    Returns:
        F1 promedio (macro)
    """
    f1_scores = [m.get('f1', 0.0) for m in metrics_per_symbol.values()]
    
    if len(f1_scores) == 0:
        return 0.0
    
    return np.mean(f1_scores)


def save_thresholds_json(thresholds_per_symbol: Dict[str, float],
                         global_threshold: float,
                         output_path: Path):
    """Guarda umbrales en formato JSON."""
    thresholds_data = {
        'global': global_threshold,
        'per_symbol': thresholds_per_symbol
    }
    
    with open(output_path, 'w') as f:
        json.dump(thresholds_data, f, indent=2)
    
    logger.info(f"✓ Umbrales guardados: {output_path}")


def save_threshold_eval_csv(metrics_global: Dict[str, Dict[str, float]],
                            metrics_per_symbol: Dict[str, Dict[str, float]],
                            alphabet: list,
                            output_path: Path):
    """Guarda evaluación de umbrales en CSV."""
    rows = []
    
    for symbol in alphabet:
        metrics_g = metrics_global.get(symbol, {})
        metrics_ps = metrics_per_symbol.get(symbol, {})
        
        row = {
            'symbol': symbol,
            'threshold_global': metrics_g.get('threshold', 0.5),
            'f1_global': metrics_g.get('f1', 0.0),
            'precision_global': metrics_g.get('precision', 0.0),
            'recall_global': metrics_g.get('recall', 0.0),
            'threshold_per_symbol': metrics_ps.get('threshold', 0.5),
            'f1_per_symbol': metrics_ps.get('f1', 0.0),
            'precision_per_symbol': metrics_ps.get('precision', 0.0),
            'recall_per_symbol': metrics_ps.get('recall', 0.0),
            'delta_f1': metrics_ps.get('f1', 0.0) - metrics_g.get('f1', 0.0),
            'delta_precision': metrics_ps.get('precision', 0.0) - metrics_g.get('precision', 0.0),
            'delta_recall': metrics_ps.get('recall', 0.0) - metrics_g.get('recall', 0.0)
        }
        rows.append(row)
    
    # Agregar fila de promedios
    avg_row = {
        'symbol': 'AVERAGE',
        'threshold_global': np.mean([m.get('threshold', 0.5) for m in metrics_global.values()]),
        'f1_global': compute_average_f1(metrics_global),
        'precision_global': np.mean([m.get('precision', 0.0) for m in metrics_global.values()]),
        'recall_global': np.mean([m.get('recall', 0.0) for m in metrics_global.values()]),
        'threshold_per_symbol': np.mean([m.get('threshold', 0.5) for m in metrics_per_symbol.values()]),
        'f1_per_symbol': compute_average_f1(metrics_per_symbol),
        'precision_per_symbol': np.mean([m.get('precision', 0.0) for m in metrics_per_symbol.values()]),
        'recall_per_symbol': np.mean([m.get('recall', 0.0) for m in metrics_per_symbol.values()]),
        'delta_f1': compute_average_f1(metrics_per_symbol) - compute_average_f1(metrics_global),
        'delta_precision': np.mean([m.get('precision', 0.0) for m in metrics_per_symbol.values()]) - np.mean([m.get('precision', 0.0) for m in metrics_global.values()]),
        'delta_recall': np.mean([m.get('recall', 0.0) for m in metrics_per_symbol.values()]) - np.mean([m.get('recall', 0.0) for m in metrics_global.values()])
    }
    rows.append(avg_row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    
    logger.info(f"✓ Evaluación de umbrales guardada: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Encontrar umbrales óptimos de decisión')
    
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best.pt',
                       help='Path al checkpoint (default: checkpoints/best.pt)')
    parser.add_argument('--val_data', type=str, default='data/alphabet/val_wide.parquet',
                       help='Path al archivo de validación (parquet)')
    parser.add_argument('--hparams', type=str, default='hparams.json',
                       help='Path al archivo de hiperparámetros')
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                       help='Directorio de salida para umbrales y evaluación')
    parser.add_argument('--global_threshold', type=float, default=0.5,
                       help='Umbral global para comparación (default: 0.5)')
    
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
    
    # Cargar dataset de validación
    val_dataset = AlphabetDataset(Path(args.val_data), MAX_PREFIX_LEN)
    val_loader = DataLoader(
        val_dataset,
        batch_size=hparams['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    # Evaluar modelo
    logger.info("Evaluando modelo en conjunto de validación...")
    all_logits = []
    all_y_true = []
    
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            prefix_indices = batch['prefix_indices'].to(device)
            lengths = batch['lengths'].to(device)
            y_true = batch['y'].to(device)
            
            logits = model(prefix_indices, lengths, return_logits=True)
            
            all_logits.append(logits.cpu())
            all_y_true.append(y_true.cpu())
    
    # Concatenar todas las predicciones
    all_logits = torch.cat(all_logits, dim=0)
    all_y_true = torch.cat(all_y_true, dim=0)
    
    # Convertir logits a probabilidades
    probs = torch.sigmoid(all_logits)
    
    y_true_np = all_y_true.numpy()
    probs_np = probs.numpy()
    
    logger.info(f"Total de ejemplos: {len(y_true_np):,}")
    
    # Encontrar umbrales óptimos por símbolo
    logger.info("="*60)
    logger.info("BÚSQUEDA DE UMBRALES ÓPTIMOS POR SÍMBOLO")
    logger.info("="*60)
    logger.info("Buscando umbral que maximiza F1 para cada símbolo...")
    
    thresholds_per_symbol = find_thresholds_per_symbol(y_true_np, probs_np, ALPHABET)
    
    logger.info("✓ Umbrales óptimos encontrados")
    logger.info(f"  Rango de umbrales: [{min(thresholds_per_symbol.values()):.4f}, {max(thresholds_per_symbol.values()):.4f}]")
    logger.info(f"  Media de umbrales: {np.mean(list(thresholds_per_symbol.values())):.4f}")
    
    # Umbral global
    global_threshold = args.global_threshold
    thresholds_global = {'global': global_threshold}
    
    # Evaluar con umbral global
    logger.info("="*60)
    logger.info("EVALUACIÓN CON UMBRAL GLOBAL")
    logger.info("="*60)
    logger.info(f"Umbral global: {global_threshold:.4f}")
    
    metrics_global = evaluate_with_thresholds(
        y_true_np, probs_np, thresholds_global, ALPHABET, threshold_type='global'
    )
    
    f1_global_avg = compute_average_f1(metrics_global)
    coverage_global = compute_coverage(metrics_global)
    
    logger.info(f"  F1 promedio (macro): {f1_global_avg:.6f}")
    logger.info(f"  Coverage: {coverage_global:.2f}%")
    
    # Evaluar con umbrales por símbolo
    logger.info("="*60)
    logger.info("EVALUACIÓN CON UMBRALES POR SÍMBOLO")
    logger.info("="*60)
    
    metrics_per_symbol = evaluate_with_thresholds(
        y_true_np, probs_np, thresholds_per_symbol, ALPHABET, threshold_type='per_symbol'
    )
    
    f1_per_symbol_avg = compute_average_f1(metrics_per_symbol)
    coverage_per_symbol = compute_coverage(metrics_per_symbol)
    
    logger.info(f"  F1 promedio (macro): {f1_per_symbol_avg:.6f}")
    logger.info(f"  Coverage: {coverage_per_symbol:.2f}%")
    
    # Comparar resultados
    logger.info("="*60)
    logger.info("COMPARACIÓN DE RESULTADOS")
    logger.info("="*60)
    
    delta_f1 = f1_per_symbol_avg - f1_global_avg
    delta_coverage = coverage_per_symbol - coverage_global
    
    logger.info(f"  ΔF1 (per_symbol - global): {delta_f1:+.6f}")
    logger.info(f"  ΔCoverage (per_symbol - global): {delta_coverage:+.2f}%")
    
    if delta_f1 > 0:
        logger.info(f"  ✓ Umbrales por símbolo mejoran F1 en {delta_f1:.6f}")
    elif delta_f1 < 0:
        logger.info(f"  ⚠ Umbrales por símbolo reducen F1 en {abs(delta_f1):.6f}")
    else:
        logger.info(f"  = Umbrales por símbolo no cambian F1")
    
    # Crear directorio de salida
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Guardar umbrales
    save_thresholds_json(
        thresholds_per_symbol,
        global_threshold,
        output_dir / 'thresholds.json'
    )
    
    # Guardar evaluación
    save_threshold_eval_csv(
        metrics_global,
        metrics_per_symbol,
        ALPHABET,
        output_dir / 'threshold_eval.csv'
    )
    
    logger.info("="*60)
    logger.info("PROCESO COMPLETADO")
    logger.info("="*60)
    logger.info(f"Umbrales guardados: {output_dir / 'thresholds.json'}")
    logger.info(f"Evaluación guardada: {output_dir / 'threshold_eval.csv'}")
    logger.info("")
    logger.info("Resumen:")
    logger.info(f"  Umbral global: {global_threshold:.4f}")
    logger.info(f"    - F1 promedio: {f1_global_avg:.6f}")
    logger.info(f"    - Coverage: {coverage_global:.2f}%")
    logger.info(f"  Umbrales por símbolo:")
    logger.info(f"    - F1 promedio: {f1_per_symbol_avg:.6f}")
    logger.info(f"    - Coverage: {coverage_per_symbol:.2f}%")
    logger.info(f"  Mejora:")
    logger.info(f"    - ΔF1: {delta_f1:+.6f}")
    logger.info(f"    - ΔCoverage: {delta_coverage:+.2f}%")
    logger.info("="*60)


if __name__ == '__main__':
    main()


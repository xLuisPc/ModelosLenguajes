"""
Métricas de evaluación para AlphabetNet.

Calcula métricas de clasificación multi-etiqueta a nivel símbolo:
- auPRC macro (Average Precision promedio por símbolo)
- micro-auPRC (Average Precision agregada)
- F1@threshold
- Coverage (porcentaje de símbolos con AP definido)
"""

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, f1_score
from typing import Dict, Tuple, Optional, List


def compute_ap_per_symbol(y_true: np.ndarray, y_scores: np.ndarray, 
                          alphabet: List[str]) -> Dict[str, float]:
    """
    Calcula Average Precision (AP) por símbolo.
    
    Args:
        y_true: Array de forma (n_samples, n_symbols) con etiquetas binarias (0/1)
        y_scores: Array de forma (n_samples, n_symbols) con scores/probabilidades
        alphabet: Lista de nombres de símbolos (orden correspondiente a las columnas)
    
    Returns:
        Dict con AP por símbolo: {symbol: ap_score} o {symbol: np.nan} si no hay positivos
    """
    n_symbols = y_true.shape[1]
    if len(alphabet) != n_symbols:
        raise ValueError(f"Tamaño de alphabet ({len(alphabet)}) no coincide con número de símbolos ({n_symbols})")
    
    ap_per_symbol = {}
    
    for i, symbol in enumerate(alphabet):
        y_true_symbol = y_true[:, i]
        y_scores_symbol = y_scores[:, i]
        
        # Verificar si hay al menos un positivo
        if y_true_symbol.sum() == 0:
            # No hay positivos para este símbolo, AP no está definido
            ap_per_symbol[symbol] = np.nan
        else:
            # Calcular AP usando sklearn
            ap = average_precision_score(y_true_symbol, y_scores_symbol)
            ap_per_symbol[symbol] = ap
    
    return ap_per_symbol


def compute_macro_auprc(ap_per_symbol: Dict[str, float]) -> float:
    """
    Calcula auPRC macro como promedio simple de los APs por símbolo.
    
    Args:
        ap_per_symbol: Dict con AP por símbolo (puede contener np.nan)
    
    Returns:
        Promedio de los APs válidos (ignora np.nan)
    """
    valid_aps = [ap for ap in ap_per_symbol.values() if not np.isnan(ap)]
    
    if len(valid_aps) == 0:
        return np.nan
    
    return np.mean(valid_aps)


def compute_micro_auprc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Calcula auPRC micro (Average Precision agregando todas las clases).
    
    Args:
        y_true: Array de forma (n_samples, n_symbols) con etiquetas binarias
        y_scores: Array de forma (n_samples, n_symbols) con scores/probabilidades
    
    Returns:
        Average Precision micro
    """
    # Aplanar las matrices: tratar cada símbolo como una predicción independiente
    y_true_flat = y_true.flatten()
    y_scores_flat = y_scores.flatten()
    
    # Si no hay positivos en total, AP no está definido
    if y_true_flat.sum() == 0:
        return np.nan
    
    return average_precision_score(y_true_flat, y_scores_flat)


def compute_f1_at_threshold(y_true: np.ndarray, y_scores: np.ndarray, 
                            threshold: float = 0.5) -> float:
    """
    Calcula F1 score usando un threshold para binarizar las predicciones.
    
    Args:
        y_true: Array de forma (n_samples, n_symbols) con etiquetas binarias
        y_scores: Array de forma (n_samples, n_symbols) con scores/probabilidades
        threshold: Umbral para binarizar (default: 0.5)
    
    Returns:
        F1 score micro (agregando todas las predicciones)
    """
    # Binarizar predicciones
    y_pred = (y_scores >= threshold).astype(int)
    
    # Aplanar para calcular F1 micro (agregando todas las predicciones)
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    return f1_score(y_true_flat, y_pred_flat, average='micro', zero_division=0)


def compute_coverage(ap_per_symbol: Dict[str, float]) -> float:
    """
    Calcula coverage: porcentaje de símbolos con AP definido (no NaN).
    
    Args:
        ap_per_symbol: Dict con AP por símbolo
    
    Returns:
        Porcentaje de símbolos con AP definido (0-100)
    """
    total_symbols = len(ap_per_symbol)
    if total_symbols == 0:
        return 0.0
    
    valid_symbols = sum(1 for ap in ap_per_symbol.values() if not np.isnan(ap))
    
    return (valid_symbols / total_symbols) * 100.0


def compute_pos_weight(y_true: np.ndarray) -> torch.Tensor:
    """
    Calcula pos_weight para BCEWithLogitsLoss a nivel símbolo.
    
    pos_weight[i] = num_negativos[i] / num_positivos[i]
    
    Esto balancea la pérdida cuando hay desbalance entre clases positivas y negativas
    por símbolo. Para clases balanceadas, pos_weight ≈ 1.0.
    
    Args:
        y_true: Array de forma (n_samples, n_symbols) con etiquetas binarias (0/1)
    
    Returns:
        Tensor de forma (n_symbols,) con pos_weight para cada símbolo
    """
    # Convertir a numpy si es tensor de PyTorch
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    
    y_true = np.asarray(y_true)
    n_symbols = y_true.shape[1]
    
    pos_weight = np.ones(n_symbols, dtype=np.float32)
    
    for i in range(n_symbols):
        y_symbol = y_true[:, i]
        num_positives = y_symbol.sum()
        num_negatives = len(y_symbol) - num_positives
        
        if num_positives > 0:
            pos_weight[i] = num_negatives / num_positives
        else:
            # Si no hay positivos, usar 1.0 (no balancear)
            pos_weight[i] = 1.0
    
    return torch.tensor(pos_weight, dtype=torch.float32)


def evaluate_metrics(y_true: np.ndarray, y_scores: np.ndarray, 
                    alphabet: List[str], threshold: float = 0.5) -> Dict[str, float]:
    """
    Calcula todas las métricas de evaluación.
    
    Args:
        y_true: Array de forma (n_samples, n_symbols) con etiquetas binarias
        y_scores: Array de forma (n_samples, n_symbols) con scores/probabilidades
        alphabet: Lista de nombres de símbolos (orden correspondiente a las columnas)
        threshold: Umbral para F1@threshold (default: 0.5)
    
    Returns:
        Dict con todas las métricas:
        {
            'macro_auprc': float,
            'micro_auprc': float,
            'f1_at_threshold': float,
            'coverage': float,
            'ap_per_symbol': Dict[str, float]
        }
    """
    # Convertir a numpy si son tensores de PyTorch
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_scores, torch.Tensor):
        y_scores = y_scores.cpu().numpy()
    
    # Asegurar que son arrays numpy
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    
    # Verificar dimensiones
    if y_true.shape != y_scores.shape:
        raise ValueError(f"Formas no coinciden: y_true {y_true.shape} vs y_scores {y_scores.shape}")
    
    # Calcular AP por símbolo
    ap_per_symbol = compute_ap_per_symbol(y_true, y_scores, alphabet)
    
    # Calcular métricas agregadas
    macro_auprc = compute_macro_auprc(ap_per_symbol)
    micro_auprc = compute_micro_auprc(y_true, y_scores)
    f1_at_threshold = compute_f1_at_threshold(y_true, y_scores, threshold)
    coverage = compute_coverage(ap_per_symbol)
    
    return {
        'macro_auprc': macro_auprc,
        'micro_auprc': micro_auprc,
        'f1_at_threshold': f1_at_threshold,
        'coverage': coverage,
        'ap_per_symbol': ap_per_symbol
    }


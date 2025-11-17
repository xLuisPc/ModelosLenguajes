"""
Métricas de evaluación para AlphabetNet.

Calcula métricas de clasificación multi-etiqueta a nivel símbolo:
- auPRC macro (Average Precision promedio por símbolo)
- micro-auPRC (Average Precision agregada)
- F1@threshold
- Coverage (porcentaje de símbolos con AP definido)
- F1 macro por símbolo
- F1 mínimo por símbolo
- ECE (Expected Calibration Error)
- Exactitud de conjunto por autómata
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


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, 
                               n_bins: int = 10) -> float:
    """
    Calcula Expected Calibration Error (ECE).
    
    ECE mide qué tan bien calibradas están las probabilidades del modelo.
    Un ECE bajo indica que las probabilidades son confiables.
    
    Args:
        y_true: Array de forma (n_samples, n_symbols) o (n_samples,) con etiquetas binarias
        y_prob: Array de forma (n_samples, n_symbols) o (n_samples,) con probabilidades
        n_bins: Número de bins para la calibración (default: 10)
    
    Returns:
        ECE (Expected Calibration Error)
    """
    # Convertir a numpy si son tensores de PyTorch
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_prob, torch.Tensor):
        y_prob = y_prob.cpu().numpy()
    
    # Aplanar si es necesario
    y_true = np.asarray(y_true).flatten()
    y_prob = np.asarray(y_prob).flatten()
    
    # Crear bins
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        # Encontrar índices en este bin
        bin_lower = bins[i]
        bin_upper = bins[i + 1]
        
        # Caso especial para el último bin (incluye el límite superior)
        if i == n_bins - 1:
            idx = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        else:
            idx = (y_prob >= bin_lower) & (y_prob < bin_upper)
        
        if not np.any(idx):
            continue
        
        # Calcular accuracy y confianza en este bin
        acc = np.mean((y_prob[idx] >= 0.5) == y_true[idx])
        conf = np.mean(y_prob[idx])
        
        # Peso del bin (proporción de muestras en este bin)
        bin_weight = np.sum(idx) / len(y_prob)
        
        # Contribución al ECE
        ece += bin_weight * abs(acc - conf)
    
    return float(ece)


def compute_f1_per_symbol(y_true: np.ndarray, y_pred: np.ndarray, 
                          alphabet: List[str]) -> Dict[str, float]:
    """
    Calcula F1 score por símbolo.
    
    Args:
        y_true: Array de forma (n_samples, n_symbols) con etiquetas binarias
        y_pred: Array de forma (n_samples, n_symbols) con predicciones binarias
        alphabet: Lista de nombres de símbolos
    
    Returns:
        Dict con F1 por símbolo: {symbol: f1_score}
    """
    # Convertir a numpy si son tensores
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    f1_per_symbol = {}
    
    for i, symbol in enumerate(alphabet):
        y_true_symbol = y_true[:, i]
        y_pred_symbol = y_pred[:, i]
        
        # Calcular F1 para este símbolo
        f1 = f1_score(y_true_symbol, y_pred_symbol, zero_division=0)
        f1_per_symbol[symbol] = float(f1)
    
    return f1_per_symbol


def compute_f1_macro(y_true: np.ndarray, y_pred: np.ndarray, 
                     alphabet: List[str]) -> float:
    """
    Calcula F1 macro (promedio de F1 por símbolo).
    
    Args:
        y_true: Array de forma (n_samples, n_symbols) con etiquetas binarias
        y_pred: Array de forma (n_samples, n_symbols) con predicciones binarias
        alphabet: Lista de nombres de símbolos
    
    Returns:
        F1 macro (promedio de F1 por símbolo)
    """
    f1_per_symbol = compute_f1_per_symbol(y_true, y_pred, alphabet)
    f1_scores = list(f1_per_symbol.values())
    
    if len(f1_scores) == 0:
        return 0.0
    
    return float(np.mean(f1_scores))


def compute_f1_min(y_true: np.ndarray, y_pred: np.ndarray, 
                   alphabet: List[str]) -> float:
    """
    Calcula F1 mínimo por símbolo.
    
    Args:
        y_true: Array de forma (n_samples, n_symbols) con etiquetas binarias
        y_pred: Array de forma (n_samples, n_symbols) con predicciones binarias
        alphabet: Lista de nombres de símbolos
    
    Returns:
        F1 mínimo por símbolo
    """
    f1_per_symbol = compute_f1_per_symbol(y_true, y_pred, alphabet)
    f1_scores = list(f1_per_symbol.values())
    
    if len(f1_scores) == 0:
        return 0.0
    
    return float(np.min(f1_scores))


def set_accuracy_by_automata(df_test, model, tokenizer_func, threshold_per_symbol: Dict[str, float],
                             alphabet: List[str], device: str = 'cpu') -> float:
    """
    Calcula exactitud de conjunto por autómata.
    
    La exactitud de conjunto se define como: el conjunto predicho de símbolos
    debe ser exactamente igual al conjunto verdadero de símbolos.
    
    Args:
        df_test: DataFrame con columnas 'dfa_id', 'regex', y columnas A-L
        model: Modelo entrenado
        tokenizer_func: Función que tokeniza regex: regex -> (indices, length)
        threshold_per_symbol: Dict con umbrales por símbolo: {symbol: threshold}
        alphabet: Lista de nombres de símbolos
        device: Dispositivo ('cpu' o 'cuda')
    
    Returns:
        Exactitud de conjunto (proporción de autómatas con predicción exacta)
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for _, row in df_test.iterrows():
            regex = str(row['regex'])
            
            # Obtener conjunto verdadero de símbolos
            true_set = {sym for sym in alphabet if row[sym] == 1}
            
            # Tokenizar regex
            regex_indices, length = tokenizer_func(regex)
            regex_indices = regex_indices.unsqueeze(0).to(device)  # (1, max_len)
            lengths = torch.tensor([length], dtype=torch.long).to(device)  # (1,)
            
            # Inferencia
            logits = model(regex_indices, lengths, return_logits=True)
            probs = torch.sigmoid(logits).cpu().numpy()[0]  # (alphabet_size,)
            
            # Construir conjunto predicho usando thresholds por símbolo
            pred_set = set()
            for i, symbol in enumerate(alphabet):
                threshold = threshold_per_symbol.get(symbol, 0.5)
                if probs[i] >= threshold:
                    pred_set.add(symbol)
            
            # Verificar si el conjunto es exacto
            if pred_set == true_set:
                correct += 1
            total += 1
    
    return correct / total if total > 0 else 0.0


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
            'f1_macro': float,
            'f1_min': float,
            'ece': float,
            'ap_per_symbol': Dict[str, float],
            'f1_per_symbol': Dict[str, float]
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
    
    # Calcular ECE
    ece = expected_calibration_error(y_true, y_scores)
    
    # Binarizar predicciones para F1 macro y F1 min
    y_pred = (y_scores >= threshold).astype(int)
    f1_per_symbol = compute_f1_per_symbol(y_true, y_pred, alphabet)
    f1_macro = compute_f1_macro(y_true, y_pred, alphabet)
    f1_min = compute_f1_min(y_true, y_pred, alphabet)
    
    return {
        'macro_auprc': macro_auprc,
        'micro_auprc': micro_auprc,
        'f1_at_threshold': f1_at_threshold,
        'coverage': coverage,
        'f1_macro': f1_macro,
        'f1_min': f1_min,
        'ece': ece,
        'ap_per_symbol': ap_per_symbol,
        'f1_per_symbol': f1_per_symbol
    }


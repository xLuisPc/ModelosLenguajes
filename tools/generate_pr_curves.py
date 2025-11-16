"""
Script para generar curvas Precision-Recall y reporte de validación.

Características:
- Carga el mejor checkpoint
- Evalúa en conjunto de validación
- Genera curvas PR: macro agregada y top-10 símbolos más frecuentes
- Exporta CSV con AP por símbolo y soporte
- Genera reporte A2_report.md
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from model import AlphabetNet
from metrics import evaluate_metrics, compute_ap_per_symbol
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


def compute_support_per_symbol(y_true: np.ndarray, alphabet: List[str]) -> Dict[str, int]:
    """
    Calcula el soporte (número de positivos) por símbolo.
    
    Args:
        y_true: Array de forma (n_samples, n_symbols) con etiquetas binarias
        alphabet: Lista de nombres de símbolos
    
    Returns:
        Dict con soporte por símbolo: {symbol: count}
    """
    support = {}
    for i, symbol in enumerate(alphabet):
        support[symbol] = int(y_true[:, i].sum())
    return support


def compute_pr_curves_per_symbol(y_true: np.ndarray, y_scores: np.ndarray,
                                  alphabet: List[str]) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Calcula curvas Precision-Recall por símbolo.
    
    Args:
        y_true: Array de forma (n_samples, n_symbols) con etiquetas binarias
        y_scores: Array de forma (n_samples, n_symbols) con scores/probabilidades
        alphabet: Lista de nombres de símbolos
    
    Returns:
        Dict con curvas PR por símbolo: {symbol: (precision, recall, thresholds)}
    """
    pr_curves = {}
    
    for i, symbol in enumerate(alphabet):
        y_true_symbol = y_true[:, i]
        y_scores_symbol = y_scores[:, i]
        
        # Solo calcular si hay positivos
        if y_true_symbol.sum() > 0:
            precision, recall, thresholds = precision_recall_curve(
                y_true_symbol, y_scores_symbol
            )
            pr_curves[symbol] = (precision, recall, thresholds)
        else:
            pr_curves[symbol] = (None, None, None)
    
    return pr_curves


def compute_macro_pr_curve(pr_curves: Dict[str, Tuple]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcula curva PR macro agregada promediando precision y recall por símbolo.
    
    Args:
        pr_curves: Dict con curvas PR por símbolo
    
    Returns:
        Tuple (precision_macro, recall_macro) promediadas
    """
    # Recopilar todas las precision y recall válidas
    all_precisions = []
    all_recalls = []
    
    for symbol, (precision, recall, _) in pr_curves.items():
        if precision is not None and recall is not None:
            all_precisions.append(precision)
            all_recalls.append(recall)
    
    if len(all_precisions) == 0:
        return np.array([]), np.array([])
    
    # Interpolar todas las curvas a los mismos puntos de recall
    # Usamos recall común desde 0 a 1 con 100 puntos
    recall_common = np.linspace(0, 1, 101)
    precisions_interp = []
    
    for precision, recall in zip(all_precisions, all_recalls):
        # Interpolar precision a recall_common
        # sklearn retorna arrays ordenados de forma descendente en recall
        # Necesitamos revertir para interpolar correctamente
        if len(recall) > 1:
            # Asegurar que recall esté ordenado de menor a mayor para interp
            sorted_idx = np.argsort(recall)
            recall_sorted = recall[sorted_idx]
            precision_sorted = precision[sorted_idx]
            
            # Interpolar
            precision_interp = np.interp(recall_common, recall_sorted, precision_sorted)
            precisions_interp.append(precision_interp)
        else:
            # Si solo hay un punto, usar ese valor para todo
            precisions_interp.append(np.full_like(recall_common, precision[0]))
    
    # Promediar precisiones interpoladas
    if len(precisions_interp) > 0:
        precision_macro = np.mean(precisions_interp, axis=0)
    else:
        precision_macro = np.array([])
    
    return precision_macro, recall_common


def plot_macro_pr_curve(precision_macro: np.ndarray, recall_macro: np.ndarray,
                        macro_ap: float, output_path: Path):
    """Grafica curva PR macro agregada."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(recall_macro, precision_macro, linewidth=2, label=f'Macro PR (AP={macro_ap:.4f})')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Curva Precision-Recall Macro Agregada', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Gráfico macro PR guardado: {output_path}")


def plot_top10_pr_curves(pr_curves: Dict[str, Tuple], support: Dict[str, int],
                         ap_per_symbol: Dict[str, float], alphabet: List[str],
                         output_path: Path):
    """Grafica curvas PR para los top-10 símbolos más frecuentes."""
    # Ordenar símbolos por soporte (frecuencia)
    sorted_symbols = sorted(
        support.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]
    
    top10_symbols = [symbol for symbol, _ in sorted_symbols]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for i, symbol in enumerate(top10_symbols):
        if symbol in pr_curves:
            precision, recall, _ = pr_curves[symbol]
            if precision is not None and recall is not None:
                ap = ap_per_symbol.get(symbol, np.nan)
                if np.isnan(ap):
                    label = f'{symbol} (AP=N/A, support={support[symbol]})'
                else:
                    label = f'{symbol} (AP={ap:.4f}, support={support[symbol]})'
                ax.plot(recall, precision, linewidth=1.5, label=label, color=colors[i])
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Curvas Precision-Recall - Top-10 Símbolos Más Frecuentes', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Gráfico top-10 PR guardado: {output_path}")


def save_per_symbol_ap_csv(ap_per_symbol: Dict[str, float], support: Dict[str, int],
                           macro_ap: float, micro_ap: float, output_path: Path):
    """Guarda CSV con AP por símbolo y soporte."""
    rows = []
    for symbol in ALPHABET:
        ap = ap_per_symbol.get(symbol, np.nan)
        sup = support.get(symbol, 0)
        rows.append({
            'symbol': symbol,
            'AP': ap if not np.isnan(ap) else '',
            'support': sup
        })
    
    # Agregar filas de medias
    rows.append({
        'symbol': 'MACRO',
        'AP': macro_ap if not np.isnan(macro_ap) else '',
        'support': ''
    })
    rows.append({
        'symbol': 'MICRO',
        'AP': micro_ap if not np.isnan(micro_ap) else '',
        'support': ''
    })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    
    logger.info(f"✓ CSV guardado: {output_path}")


def generate_report(ap_per_symbol: Dict[str, float], support: Dict[str, int],
                   macro_ap: float, micro_ap: float, coverage: float,
                   checkpoint_path: Path, output_path: Path):
    """Genera reporte A2_report.md con interpretación."""
    
    # Calcular estadísticas adicionales
    valid_aps = [ap for ap in ap_per_symbol.values() if not np.isnan(ap)]
    min_ap = min(valid_aps) if valid_aps else np.nan
    max_ap = max(valid_aps) if valid_aps else np.nan
    
    # Top-5 símbolos por AP
    sorted_by_ap = sorted(
        [(s, ap) for s, ap in ap_per_symbol.items() if not np.isnan(ap)],
        key=lambda x: x[1],
        reverse=True
    )[:5]
    
    # Top-5 símbolos por soporte
    sorted_by_support = sorted(
        support.items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]
    
    report = f"""# Reporte de Validación - A2

## Resumen de Métricas

| Métrica | Valor |
|---------|-------|
| auPRC Macro | {macro_ap:.6f} |
| auPRC Micro | {micro_ap:.6f} |
| Coverage | {coverage:.2f}% |
| AP Mínimo | {min_ap:.6f} |
| AP Máximo | {max_ap:.6f} |

## Top-5 Símbolos por AP

| Símbolo | AP | Soporte |
|---------|-----|---------|
"""
    
    for symbol, ap in sorted_by_ap:
        sup = support[symbol]
        report += f"| {symbol} | {ap:.6f} | {sup} |\n"
    
    report += f"""
## Top-5 Símbolos por Soporte (Frecuencia)

| Símbolo | Soporte | AP |
|---------|---------|-----|
"""
    
    for symbol, sup in sorted_by_support:
        ap = ap_per_symbol.get(symbol, np.nan)
        ap_str = f"{ap:.6f}" if not np.isnan(ap) else "N/A"
        report += f"| {symbol} | {sup} | {ap_str} |\n"
    
    report += f"""
## Tabla Completa: AP por Símbolo

| Símbolo | AP | Soporte |
|---------|-----|---------|
"""
    
    for symbol in ALPHABET:
        ap = ap_per_symbol.get(symbol, np.nan)
        sup = support.get(symbol, 0)
        ap_str = f"{ap:.6f}" if not np.isnan(ap) else "N/A"
        report += f"| {symbol} | {ap_str} | {sup} |\n"
    
    report += f"""
## Interpretación

### Rendimiento General

El modelo alcanza un auPRC macro de **{macro_ap:.6f}** y un auPRC micro de **{micro_ap:.6f}**, lo que indica un rendimiento {'moderado' if macro_ap < 0.7 else 'bueno' if macro_ap < 0.9 else 'excelente'} en la predicción de símbolos válidos después de un prefijo. La diferencia entre macro y micro sugiere que {'hay desbalance entre símbolos, con algunos símbolos siendo más difíciles de predecir que otros' if abs(macro_ap - micro_ap) > 0.1 else 'el rendimiento es relativamente uniforme entre símbolos'}.

### Distribución de Rendimiento por Símbolo

El coverage de **{coverage:.2f}%** indica que {'todos' if coverage == 100.0 else 'la mayoría' if coverage >= 90.0 else 'algunos'} de los símbolos tienen ejemplos positivos en el conjunto de validación. Los símbolos con mayor AP (como {', '.join([s for s, _ in sorted_by_ap[:3]])}) muestran un rendimiento {'superior' if max_ap > 0.8 else 'moderado' if max_ap > 0.6 else 'limitado'}, mientras que los símbolos con menor AP presentan mayores desafíos para el modelo. La variabilidad en AP entre símbolos (rango: {min_ap:.6f} - {max_ap:.6f}) {'es considerable' if (max_ap - min_ap) > 0.3 else 'es moderada' if (max_ap - min_ap) > 0.15 else 'es baja'}.

### Análisis de Símbolos Frecuentes

Los símbolos más frecuentes (como {', '.join([s for s, _ in sorted_by_support[:3]])}) tienen {'un rendimiento ' if any(ap_per_symbol.get(s, np.nan) > 0.7 for s, _ in sorted_by_support[:3]) else 'rendimiento '}que {'confirma que el modelo aprende efectivamente patrones comunes' if all(ap_per_symbol.get(s, np.nan) > 0.6 for s, _ in sorted_by_support[:3]) else 'podría mejorarse, sugiriendo que aunque estos símbolos son comunes, su predicción sigue siendo desafiante'}. La relación entre frecuencia y AP {'muestra una correlación positiva' if all(ap_per_symbol.get(s, np.nan) > 0.6 for s, _ in sorted_by_support[:3]) else 'no es clara, indicando que la frecuencia por sí sola no garantiza un mejor rendimiento'}.

---

**Checkpoint evaluado:** {checkpoint_path}

**Generado:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"✓ Reporte guardado: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generar curvas PR y reporte de validación')
    
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best.pt',
                       help='Path al checkpoint (default: checkpoints/best.pt)')
    parser.add_argument('--val_data', type=str, default='data/alphabet/val_wide.parquet',
                       help='Path al archivo de validación (parquet)')
    parser.add_argument('--hparams', type=str, default='hparams.json',
                       help='Path al archivo de hiperparámetros')
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                       help='Directorio de salida para gráficos y reporte')
    
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
    
    # Calcular métricas
    logger.info("Calculando métricas...")
    metrics = evaluate_metrics(y_true_np, probs_np, ALPHABET, threshold=0.5)
    
    logger.info(f"✓ auPRC Macro: {metrics['macro_auprc']:.6f}")
    logger.info(f"✓ auPRC Micro: {metrics['micro_auprc']:.6f}")
    logger.info(f"✓ Coverage: {metrics['coverage']:.2f}%")
    
    # Calcular soporte por símbolo
    support = compute_support_per_symbol(y_true_np, ALPHABET)
    
    # Calcular curvas PR por símbolo
    logger.info("Calculando curvas PR por símbolo...")
    pr_curves = compute_pr_curves_per_symbol(y_true_np, probs_np, ALPHABET)
    
    # Calcular curva PR macro agregada
    logger.info("Calculando curva PR macro agregada...")
    precision_macro, recall_macro = compute_macro_pr_curve(pr_curves)
    
    # Crear directorio de salida
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generar gráficos
    logger.info("Generando gráficos...")
    plot_macro_pr_curve(
        precision_macro, recall_macro,
        metrics['macro_auprc'],
        output_dir / 'pr_macro.png'
    )
    
    plot_top10_pr_curves(
        pr_curves, support,
        metrics['ap_per_symbol'],
        ALPHABET,
        output_dir / 'pr_top10.png'
    )
    
    # Guardar CSV
    logger.info("Guardando CSV...")
    save_per_symbol_ap_csv(
        metrics['ap_per_symbol'],
        support,
        metrics['macro_auprc'],
        metrics['micro_auprc'],
        output_dir / 'per_symbol_ap.csv'
    )
    
    # Generar reporte
    logger.info("Generando reporte...")
    generate_report(
        metrics['ap_per_symbol'],
        support,
        metrics['macro_auprc'],
        metrics['micro_auprc'],
        metrics['coverage'],
        checkpoint_path,
        output_dir / 'A2_report.md'
    )
    
    logger.info("="*60)
    logger.info("PROCESO COMPLETADO")
    logger.info("="*60)
    logger.info(f"Gráficos guardados en: {output_dir}")
    logger.info(f"  - pr_macro.png")
    logger.info(f"  - pr_top10.png")
    logger.info(f"CSV guardado: {output_dir / 'per_symbol_ap.csv'}")
    logger.info(f"Reporte guardado: {output_dir / 'A2_report.md'}")
    logger.info("="*60)


if __name__ == '__main__':
    main()


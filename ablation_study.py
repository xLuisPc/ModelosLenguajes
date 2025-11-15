"""
Script de ablation study para AlphabetNet.

Experimentos controlados:
- GRU vs LSTM
- Con vs sin automata_id embedding
- Variaciones en emb_dim, hidden_dim, num_layers

Métricas comparadas:
- auPRC macro
- Número de parámetros
- Tiempo por época
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import AlphabetNet
from metrics import evaluate_metrics
from train import AlphabetDataset, collate_fn, ALPHABET, MAX_PREFIX_LEN, train_epoch, validate

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def count_parameters(model: nn.Module) -> int:
    """Cuenta el número total de parámetros del modelo."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model: nn.Module) -> float:
    """Calcula el tamaño del modelo en MB."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size = (param_size + buffer_size) / (1024 * 1024)  # Convertir a MB
    return total_size


def create_experiment_configs(base_hparams: dict) -> List[Dict]:
    """
    Crea configuraciones de experimentos para ablation study.
    
    Args:
        base_hparams: Hiperparámetros base
    
    Returns:
        Lista de configuraciones de experimentos
    """
    configs = []
    
    base_config = base_hparams['model'].copy()
    
    # Experimentos: GRU vs LSTM
    for rnn_type in ['GRU', 'LSTM']:
        config = base_config.copy()
        config['rnn_type'] = rnn_type
        config['name'] = f'rnn_{rnn_type.lower()}'
        configs.append(config)
    
    # Experimentos: Con vs sin automata_id embedding
    config_no_automata = base_config.copy()
    config_no_automata['use_automata_conditioning'] = False
    config_no_automata['name'] = 'no_automata_conditioning'
    configs.append(config_no_automata)
    
    config_with_automata = base_config.copy()
    config_with_automata['use_automata_conditioning'] = True
    # Necesitamos num_automata para este experimento
    # Usaremos un valor por defecto basado en el dataset
    config_with_automata['num_automata'] = 3000  # Aproximación basada en dataset3000
    config_with_automata['name'] = 'with_automata_conditioning'
    configs.append(config_with_automata)
    
    # Experimentos: Variaciones en emb_dim
    for emb_dim in [64, 96, 128]:
        if emb_dim != base_config['emb_dim']:
            config = base_config.copy()
            config['emb_dim'] = emb_dim
            config['name'] = f'emb_dim_{emb_dim}'
            configs.append(config)
    
    # Experimentos: Variaciones en hidden_dim
    for hidden_dim in [128, 192, 256]:
        if hidden_dim != base_config['hidden_dim']:
            config = base_config.copy()
            config['hidden_dim'] = hidden_dim
            config['name'] = f'hidden_dim_{hidden_dim}'
            configs.append(config)
    
    # Experimentos: Variaciones en num_layers
    for num_layers in [1, 2]:
        if num_layers != base_config['num_layers']:
            config = base_config.copy()
            config['num_layers'] = num_layers
            config['name'] = f'num_layers_{num_layers}'
            configs.append(config)
    
    return configs


def run_experiment(config: Dict, train_loader: DataLoader, val_loader: DataLoader,
                  device: torch.device, base_hparams: dict, num_epochs: int = 3,
                  alphabet: List[str] = ALPHABET) -> Dict:
    """
    Ejecuta un experimento con una configuración específica.
    
    Args:
        config: Configuración del modelo
        train_loader: DataLoader de entrenamiento
        val_loader: DataLoader de validación
        device: Dispositivo (CPU/GPU)
        base_hparams: Hiperparámetros base
        num_epochs: Número de épocas a entrenar (default: 3 para evaluación rápida)
        alphabet: Lista de nombres de símbolos
    
    Returns:
        Dict con resultados: {'config_name', 'auPRC_macro', 'params', 'size_mb', 'time_per_epoch'}
    """
    config_name = config.pop('name', 'unknown')
    
    logger.info("="*60)
    logger.info(f"Experimento: {config_name}")
    logger.info("="*60)
    logger.info(f"Configuración: {json.dumps(config, indent=2)}")
    
    # Crear modelo
    model = AlphabetNet(
        vocab_size=base_hparams['model']['vocab_size'],
        alphabet_size=base_hparams['model']['alphabet_size'],
        emb_dim=config.get('emb_dim', base_hparams['model']['emb_dim']),
        hidden_dim=config.get('hidden_dim', base_hparams['model']['hidden_dim']),
        rnn_type=config.get('rnn_type', base_hparams['model']['rnn_type']),
        num_layers=config.get('num_layers', base_hparams['model']['num_layers']),
        dropout=config.get('dropout', base_hparams['model']['dropout']),
        padding_idx=base_hparams['model']['padding_idx'],
        use_automata_conditioning=config.get('use_automata_conditioning', 
                                            base_hparams['model']['use_automata_conditioning']),
        num_automata=config.get('num_automata'),
        automata_emb_dim=config.get('automata_emb_dim', base_hparams['model']['automata_emb_dim'])
    ).to(device)
    
    # Contar parámetros
    num_params = count_parameters(model)
    size_mb = get_model_size_mb(model)
    
    logger.info(f"Parámetros: {num_params:,}")
    logger.info(f"Tamaño: {size_mb:.2f} MB")
    
    # Calcular pos_weight
    y_train_all = train_loader.dataset.labels
    from metrics import compute_pos_weight
    pos_weight = compute_pos_weight(y_train_all).to(device)
    
    # Función de pérdida
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Optimizador
    from torch.optim import Adam
    optimizer = Adam(
        model.parameters(),
        lr=base_hparams['training']['learning_rate'],
        weight_decay=base_hparams['training'].get('weight_decay', 1e-4)
    )
    
    # Entrenar por algunas épocas
    logger.info(f"Entrenando por {num_epochs} épocas...")
    times_per_epoch = []
    best_auprc_macro = -np.inf
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Entrenar
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device,
            gradient_clip=base_hparams['training'].get('gradient_clip', 1.0)
        )
        
        # Validar
        val_metrics = validate(model, val_loader, criterion, device, alphabet)
        auprc_macro = val_metrics.get('macro_auprc', np.nan)
        
        if not np.isnan(auprc_macro) and auprc_macro > best_auprc_macro:
            best_auprc_macro = auprc_macro
        
        epoch_time = time.time() - start_time
        times_per_epoch.append(epoch_time)
        
        logger.info(f"Época {epoch + 1}/{num_epochs}: "
                   f"loss={train_loss:.6f}, "
                   f"auPRC_macro={auprc_macro:.6f}, "
                   f"tiempo={epoch_time:.2f}s")
    
    avg_time_per_epoch = np.mean(times_per_epoch)
    
    logger.info(f"✓ Mejor auPRC macro: {best_auprc_macro:.6f}")
    logger.info(f"✓ Tiempo promedio por época: {avg_time_per_epoch:.2f}s")
    
    # Preparar resultado
    result = {
        'config_name': config_name,
        'rnn_type': config.get('rnn_type', base_hparams['model']['rnn_type']),
        'use_automata_conditioning': config.get('use_automata_conditioning', 
                                                base_hparams['model']['use_automata_conditioning']),
        'emb_dim': config.get('emb_dim', base_hparams['model']['emb_dim']),
        'hidden_dim': config.get('hidden_dim', base_hparams['model']['hidden_dim']),
        'num_layers': config.get('num_layers', base_hparams['model']['num_layers']),
        'auPRC_macro': best_auprc_macro if not np.isnan(best_auprc_macro) else 0.0,
        'params': num_params,
        'size_mb': size_mb,
        'time_per_epoch': avg_time_per_epoch
    }
    
    # Limpiar modelo de memoria
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return result


def save_ablation_csv(results: List[Dict], output_path: Path):
    """Guarda resultados del ablation study en CSV."""
    df = pd.DataFrame(results)
    
    # Ordenar columnas
    column_order = [
        'config_name',
        'rnn_type',
        'use_automata_conditioning',
        'emb_dim',
        'hidden_dim',
        'num_layers',
        'auPRC_macro',
        'params',
        'size_mb',
        'time_per_epoch'
    ]
    
    # Reordenar columnas (mantener solo las que existen)
    existing_columns = [col for col in column_order if col in df.columns]
    df = df[existing_columns]
    
    df.to_csv(output_path, index=False)
    logger.info(f"✓ Resultados guardados: {output_path}")


def generate_conclusions(results: List[Dict], output_path: Path, append_to_report: Optional[Path] = None):
    """
    Genera párrafo de conclusiones para A2_report.md.
    
    Args:
        results: Lista de resultados de experimentos
        output_path: Path donde guardar las conclusiones (archivo separado)
        append_to_report: Path al A2_report.md existente para agregar conclusiones (opcional)
    """
    
    if len(results) == 0:
        return
    
    df = pd.DataFrame(results)
    
    # Encontrar mejor configuración
    best_idx = df['auPRC_macro'].idxmax()
    best_config = df.loc[best_idx]
    
    # Análisis de RNN type
    gru_results = df[df['rnn_type'] == 'GRU']
    lstm_results = df[df['rnn_type'] == 'LSTM']
    
    gru_avg = gru_results['auPRC_macro'].mean() if len(gru_results) > 0 else 0
    lstm_avg = lstm_results['auPRC_macro'].mean() if len(lstm_results) > 0 else 0
    
    # Análisis de automata conditioning
    no_automata = df[df['use_automata_conditioning'] == False]
    with_automata = df[df['use_automata_conditioning'] == True]
    
    no_automata_avg = no_automata['auPRC_macro'].mean() if len(no_automata) > 0 else 0
    with_automata_avg = with_automata['auPRC_macro'].mean() if len(with_automata) > 0 else 0
    
    # Análisis de dimensiones
    emb_dims = df['emb_dim'].unique()
    hidden_dims = df['hidden_dim'].unique()
    
    if len(emb_dims) > 1:
        best_emb_idx = df.groupby('emb_dim')['auPRC_macro'].idxmax()
        best_emb_dim = df.loc[best_emb_idx, 'emb_dim'].iloc[0]
    else:
        best_emb_dim = emb_dims[0] if len(emb_dims) > 0 else df['emb_dim'].iloc[0]
    
    if len(hidden_dims) > 1:
        best_hidden_idx = df.groupby('hidden_dim')['auPRC_macro'].idxmax()
        best_hidden_dim = df.loc[best_hidden_idx, 'hidden_dim'].iloc[0]
    else:
        best_hidden_dim = hidden_dims[0] if len(hidden_dims) > 0 else df['hidden_dim'].iloc[0]
    
    # Generar conclusiones
    conclusions = f"""## Ablation Study - Conclusiones

### Configuración Óptima

La mejor configuración encontrada fue **{best_config['config_name']}** con un auPRC macro de **{best_config['auPRC_macro']:.6f}**. Esta configuración utiliza RNN tipo **{best_config['rnn_type']}**, {'con' if best_config['use_automata_conditioning'] else 'sin'} conditioning por autómata, embedding dimension de **{int(best_config['emb_dim'])}**, hidden dimension de **{int(best_config['hidden_dim'])}**, y **{int(best_config['num_layers'])}** capa(s). El modelo tiene **{best_config['params']:,}** parámetros, ocupa **{best_config['size_mb']:.2f} MB** en memoria, y requiere **{best_config['time_per_epoch']:.2f} segundos** por época en promedio.

### Análisis de Componentes

**Tipo de RNN**: {'GRU' if gru_avg >= lstm_avg else 'LSTM'} obtuvo mejor rendimiento promedio ({max(gru_avg, lstm_avg):.6f} vs {min(gru_avg, lstm_avg):.6f} para {'GRU' if gru_avg >= lstm_avg else 'LSTM'}), {'indicando que la arquitectura más simple es suficiente' if gru_avg >= lstm_avg else 'sugiriendo que la mayor capacidad de LSTM es beneficiosa'} para esta tarea.

**Conditioning por Autómata**: {'El uso de conditioning por autómata' if with_automata_avg >= no_automata_avg else 'Sin conditioning por autómata'} {'mejora' if with_automata_avg >= no_automata_avg else 'resulta en mejor'} el rendimiento ({max(with_automata_avg, no_automata_avg):.6f} vs {min(with_automata_avg, no_automata_avg):.6f}), {'confirmando que el embedding de autómata proporciona información útil' if with_automata_avg >= no_automata_avg else 'sugiriendo que el modelo puede generalizar sin información explícita del autómata'}.

**Dimensiones**: El análisis de variaciones en dimensiones muestra que embedding dimension de **{int(best_emb_dim)}** y hidden dimension de **{int(best_hidden_dim)}** ofrecen el mejor balance entre rendimiento y eficiencia. {'Aumentar las dimensiones' if max(emb_dims) > best_emb_dim or max(hidden_dims) > best_hidden_dim else 'Las dimensiones seleccionadas'} {'no proporciona ganancias significativas' if max(emb_dims) > best_emb_dim or max(hidden_dims) > best_hidden_dim else 'representan una configuración óptima'}.

### Compromisos Rendimiento-Eficiencia

El análisis muestra un compromiso claro entre rendimiento y eficiencia. La configuración óptima balancea {'rendimiento superior' if best_config['auPRC_macro'] > df['auPRC_macro'].mean() else 'rendimiento competitivo'} ({best_config['auPRC_macro']:.6f}) con {'eficiencia razonable' if best_config['time_per_epoch'] < df['time_per_epoch'].mean() else 'eficiencia adecuada'} ({best_config['time_per_epoch']:.2f}s/época) y tamaño de modelo moderado ({best_config['size_mb']:.2f} MB). {'Para aplicaciones con restricciones computacionales, se podrían considerar configuraciones más simples con menor rendimiento pero mayor eficiencia.' if best_config['time_per_epoch'] > df['time_per_epoch'].quantile(0.25) else 'La configuración óptima ya representa una solución eficiente.'}

"""
    
    # Guardar en archivo separado
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(conclusions)
    
    logger.info(f"✓ Conclusiones guardadas: {output_path}")
    
    # Agregar al reporte existente si se especifica
    if append_to_report and append_to_report.exists():
        with open(append_to_report, 'r', encoding='utf-8') as f:
            report_content = f.read()
        
        # Buscar si ya hay una sección de ablation
        if '## Ablation Study' in report_content:
            # Reemplazar sección existente
            import re
            pattern = r'## Ablation Study.*?(?=\n## |\Z)'
            report_content = re.sub(pattern, conclusions.strip(), report_content, flags=re.DOTALL)
        else:
            # Agregar al final
            report_content += '\n\n' + conclusions.strip() + '\n'
        
        with open(append_to_report, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"✓ Conclusiones agregadas a: {append_to_report}")


def main():
    parser = argparse.ArgumentParser(description='Ejecutar ablation study')
    
    parser.add_argument('--train_data', type=str, default='data/alphabet/train_wide.parquet',
                       help='Path al archivo de entrenamiento (parquet)')
    parser.add_argument('--val_data', type=str, default='data/alphabet/val_wide.parquet',
                       help='Path al archivo de validación (parquet)')
    parser.add_argument('--hparams', type=str, default='hparams.json',
                       help='Path al archivo de hiperparámetros')
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                       help='Directorio de salida para resultados')
    parser.add_argument('--num_epochs', type=int, default=3,
                       help='Número de épocas por experimento (default: 3)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (por defecto usa hparams)')
    parser.add_argument('--append_to_report', type=str, default=None,
                       help='Path al A2_report.md para agregar conclusiones (opcional)')
    
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
    
    # Cargar datasets
    train_dataset = AlphabetDataset(Path(args.train_data), MAX_PREFIX_LEN)
    val_dataset = AlphabetDataset(Path(args.val_data), MAX_PREFIX_LEN)
    
    batch_size = args.batch_size if args.batch_size else hparams['training']['batch_size']
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    # Crear configuraciones de experimentos
    logger.info("Creando configuraciones de experimentos...")
    configs = create_experiment_configs(hparams)
    logger.info(f"Total de experimentos: {len(configs)}")
    
    # Crear directorio de salida
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Ejecutar experimentos
    logger.info("="*60)
    logger.info("INICIANDO ABLATION STUDY")
    logger.info("="*60)
    
    results = []
    
    for i, config in enumerate(configs, 1):
        logger.info(f"\nExperimento {i}/{len(configs)}")
        
        try:
            result = run_experiment(
                config.copy(),
                train_loader,
                val_loader,
                device,
                hparams,
                num_epochs=args.num_epochs,
                alphabet=ALPHABET
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Error en experimento {config.get('name', 'unknown')}: {e}")
            continue
    
    # Guardar resultados
    logger.info("="*60)
    logger.info("GUARDANDO RESULTADOS")
    logger.info("="*60)
    
    save_ablation_csv(results, output_dir / 'ablation.csv')
    
    # Generar conclusiones
    append_to_report = Path(args.append_to_report) if args.append_to_report else None
    generate_conclusions(
        results, 
        output_dir / 'ablation_conclusions.md',
        append_to_report=append_to_report
    )
    
    logger.info("="*60)
    logger.info("ABLATION STUDY COMPLETADO")
    logger.info("="*60)
    logger.info(f"Resultados guardados: {output_dir / 'ablation.csv'}")
    logger.info(f"Conclusiones guardadas: {output_dir / 'ablation_conclusions.md'}")
    if append_to_report:
        logger.info(f"Conclusiones agregadas a: {append_to_report}")
    logger.info("="*60)


if __name__ == '__main__':
    main()


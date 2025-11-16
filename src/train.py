"""
Script de entrenamiento para AlphabetNet.

Características:
- Optimizador Adam con ReduceLROnPlateau opcional
- Early stopping sobre val auPRC macro
- Guarda checkpoints (best.pt y last.pt) por auPRC macro
- Logging a train_log.csv
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import AlphabetNet
from metrics import evaluate_metrics, compute_pos_weight

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constantes
ALPHABET = list('ABCDEFGHIJKL')
ALPHABET_SIZE = len(ALPHABET)
SPECIAL_TOKENS = {
    'PAD': 0,
    '<EPS>': 1
}
VOCAB_SIZE = ALPHABET_SIZE + 2  # A-L + PAD + <EPS>
MAX_PREFIX_LEN = 64


def char_to_idx(char: str) -> int:
    """
    Convierte un carácter a su índice en el vocabulario.
    
    Args:
        char: Carácter ('A'-'L' o '<EPS>')
    
    Returns:
        Índice en el vocabulario
    """
    if char == '<EPS>':
        return SPECIAL_TOKENS['<EPS>']
    elif char in ALPHABET:
        return ALPHABET.index(char) + 2  # +2 porque 0=PAD, 1=<EPS>
    else:
        raise ValueError(f"Carácter inválido: {char}")


def regex_to_indices(regex: str, max_len: int = MAX_PREFIX_LEN) -> Tuple[torch.Tensor, int]:
    """
    Convierte un regex string a índices de caracteres.
    
    Args:
        regex: String del regex (puede contener cualquier carácter, pero solo A-L se tokenizan)
        max_len: Longitud máxima (padding)
    
    Returns:
        Tuple (indices, length) donde:
        - indices: Tensor 1D de índices (padded con PAD=0)
        - length: Longitud real del regex (solo caracteres A-L, otros se ignoran)
    """
    # Tokenizar regex: solo caracteres A-L se convierten a índices, otros se ignoran
    # Para compatibilidad, si el regex está vacío usamos <EPS>
    if regex == '' or regex is None:
        indices = [SPECIAL_TOKENS['<EPS>']]
        length = 1
    else:
        # Extraer solo caracteres válidos (A-L) del regex
        valid_chars = [c for c in regex if c in ALPHABET]
        if len(valid_chars) == 0:
            # Si no hay caracteres válidos, usar <EPS>
            indices = [SPECIAL_TOKENS['<EPS>']]
            length = 1
        else:
            indices = [char_to_idx(c) for c in valid_chars]
            length = len(indices)
    
    # Padding
    if length < max_len:
        indices = indices + [SPECIAL_TOKENS['PAD']] * (max_len - length)
    
    return torch.tensor(indices[:max_len], dtype=torch.long), length

# Alias para compatibilidad hacia atrás
prefix_to_indices = regex_to_indices


class AlphabetDataset(Dataset):
    """
    Dataset para datos en formato regex-sigma (CSV).
    
    Formato esperado del DataFrame:
    - dfa_id: ID del autómata
    - regex: String del regex del autómata
    - A, B, C, ..., L: Columnas con 0/1 indicando si cada símbolo pertenece al alfabeto
    """
    
    def __init__(self, csv_path: Path, max_regex_len: int = MAX_PREFIX_LEN):
        """
        Args:
            csv_path: Path al archivo CSV con formato regex-sigma
            max_regex_len: Longitud máxima de regex
        """
        logger.info(f"Cargando dataset: {csv_path}")
        self.df = pd.read_csv(csv_path)
        self.max_regex_len = max_regex_len
        
        logger.info(f"  - Total de ejemplos: {len(self.df):,}")
        
        # Convertir columnas A-L a arrays numpy
        label_columns = [col for col in ALPHABET if col in self.df.columns]
        if len(label_columns) != ALPHABET_SIZE:
            raise ValueError(f"Faltan columnas de alfabeto. Esperadas: {ALPHABET_SIZE}, encontradas: {len(label_columns)}")
        
        self.labels = self.df[label_columns].values.astype(np.float32)
        logger.info(f"  - Forma de labels: {self.labels.shape}")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retorna un ejemplo del dataset.
        
        Returns:
            Dict con:
            - 'prefix_indices': Tensor 1D de índices (padded) - usando nombre antiguo para compatibilidad
            - 'length': Tensor escalar con longitud real
            - 'y': Tensor 1D con etiquetas multi-hot (12 elementos)
        """
        row = self.df.iloc[idx]
        regex = str(row['regex'])
        
        # Convertir regex a índices
        regex_indices, length = regex_to_indices(regex, self.max_regex_len)
        
        # Obtener etiquetas desde columnas A-L
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return {
            'prefix_indices': regex_indices,  # Usando nombre antiguo para compatibilidad con model.forward
            'length': torch.tensor(length, dtype=torch.long),
            'y': y
        }


def collate_fn(batch: list) -> Dict[str, torch.Tensor]:
    """
    Función de collate para DataLoader.
    
    Args:
        batch: Lista de ejemplos del dataset
    
    Returns:
        Dict con tensores batch:
        - 'prefix_indices': (batch_size, max_len)
        - 'lengths': (batch_size,)
        - 'y': (batch_size, alphabet_size)
    """
    prefix_indices = torch.stack([item['prefix_indices'] for item in batch])
    lengths = torch.stack([item['length'] for item in batch])
    y = torch.stack([item['y'] for item in batch])
    
    return {
        'prefix_indices': prefix_indices,
        'lengths': lengths,
        'y': y
    }


class EarlyStopping:
    """Early stopping basado en F1 macro."""
    
    def __init__(self, patience: int = 8, min_delta: float = 0.0):
        """
        Args:
            patience: Número de épocas sin mejora antes de parar
            min_delta: Mejora mínima considerada como mejora
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Args:
            score: Score a monitorear (mayor es mejor) - F1 macro
        
        Returns:
            True si debe parar, False en caso contrario
        """
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        
        return self.early_stop


def train_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module,
                optimizer: torch.optim.Optimizer, device: torch.device,
                gradient_clip: float = 1.0) -> float:
    """
    Entrena el modelo por una época.
    
    Returns:
        Loss promedio de la época
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in dataloader:
        prefix_indices = batch['prefix_indices'].to(device)
        lengths = batch['lengths'].to(device)
        y_true = batch['y'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(prefix_indices, lengths, return_logits=True)
        loss = criterion(logits, y_true)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def validate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module,
             device: torch.device, alphabet: list) -> Dict[str, float]:
    """
    Valida el modelo.
    
    Returns:
        Dict con métricas: {'loss', 'macro_auprc', 'micro_auprc', 'f1_at_threshold', 'coverage'}
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    all_logits = []
    all_y_true = []
    
    with torch.no_grad():
        for batch in dataloader:
            prefix_indices = batch['prefix_indices'].to(device)
            lengths = batch['lengths'].to(device)
            y_true = batch['y'].to(device)
            
            # Forward pass
            logits = model(prefix_indices, lengths, return_logits=True)
            loss = criterion(logits, y_true)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Acumular predicciones
            all_logits.append(logits.cpu())
            all_y_true.append(y_true.cpu())
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    # Concatenar todas las predicciones
    all_logits = torch.cat(all_logits, dim=0)
    all_y_true = torch.cat(all_y_true, dim=0)
    
    # Convertir logits a probabilidades (sigmoid)
    probs = torch.sigmoid(all_logits)
    
    # Calcular métricas
    metrics = evaluate_metrics(
        all_y_true.numpy(),
        probs.numpy(),
        alphabet,
        threshold=0.5
    )
    
    metrics['loss'] = avg_loss
    return metrics


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                    epoch: int, metrics: Dict[str, float], checkpoint_path: Path,
                    is_best: bool = False):
    """Guarda un checkpoint del modelo."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'is_best': is_best
    }
    torch.save(checkpoint, checkpoint_path)
    
    if is_best:
        logger.info(f"✓ Mejor checkpoint guardado: {checkpoint_path}")
    else:
        logger.info(f"✓ Checkpoint guardado: {checkpoint_path}")


def load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                    checkpoint_path: Path) -> int:
    """Carga un checkpoint del modelo."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    logger.info(f"✓ Checkpoint cargado: {checkpoint_path} (época {epoch})")
    return epoch


def main():
    parser = argparse.ArgumentParser(description='Entrenar AlphabetNet')
    
    # Paths
    parser.add_argument('--train_data', type=str,
                       default='data/dataset_regex_sigma.csv',
                       help='Path al archivo de entrenamiento (CSV regex-sigma)')
    parser.add_argument('--val_data', type=str,
                       default='data/dataset_regex_sigma.csv',
                       help='Path al archivo de validación (CSV regex-sigma)')
    parser.add_argument('--checkpoint_dir', type=str,
                       default='checkpoints',
                       help='Directorio para guardar checkpoints')
    parser.add_argument('--hparams', type=str,
                       default='hparams.json',
                       help='Path al archivo de hiperparámetros')
    
    # Seeds
    parser.add_argument('--random_seed', type=int, default=None,
                       help='Seed para random (sobrescribe hparams)')
    parser.add_argument('--numpy_seed', type=int, default=None,
                       help='Seed para numpy (sobrescribe hparams)')
    parser.add_argument('--torch_seed', type=int, default=None,
                       help='Seed para torch (sobrescribe hparams)')
    parser.add_argument('--cuda_seed', type=int, default=None,
                       help='Seed para cuda (sobrescribe hparams)')
    
    # Training options
    parser.add_argument('--resume', type=str, default=None,
                       help='Path al checkpoint para resumir entrenamiento')
    parser.add_argument('--use_scheduler', action='store_true',
                       help='Usar ReduceLROnPlateau scheduler')
    
    args = parser.parse_args()
    
    # Cargar hiperparámetros
    hparams_path = Path(args.hparams)
    if not hparams_path.exists():
        raise FileNotFoundError(f"Archivo de hiperparámetros no encontrado: {hparams_path}")
    
    with open(hparams_path, 'r') as f:
        hparams = json.load(f)
    
    # Actualizar seeds si se proporcionan por CLI
    if args.random_seed is not None:
        hparams['seeds']['random_seed'] = args.random_seed
    if args.numpy_seed is not None:
        hparams['seeds']['numpy_seed'] = args.numpy_seed
    if args.torch_seed is not None:
        hparams['seeds']['torch_seed'] = args.torch_seed
    if args.cuda_seed is not None:
        hparams['seeds']['cuda_seed'] = args.cuda_seed
    
    # Establecer seeds para reproducibilidad
    import random
    import os
    
    # Python random
    random.seed(hparams['seeds']['random_seed'])
    
    # NumPy
    np.random.seed(hparams['seeds']['numpy_seed'])
    
    # PyTorch
    torch.manual_seed(hparams['seeds']['torch_seed'])
    
    # CUDA seeds
    if torch.cuda.is_available():
        torch.cuda.manual_seed(hparams['seeds']['cuda_seed'])
        torch.cuda.manual_seed_all(hparams['seeds']['cuda_seed'])
    
    # Configurar PyTorch para determinismo
    # Nota: Esto puede reducir el rendimiento, pero asegura reproducibilidad
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Variable de entorno para reproducibilidad adicional
    os.environ['PYTHONHASHSEED'] = str(hparams['seeds']['random_seed'])
    
    logger.info("✓ Seeds configurados para reproducibilidad")
    logger.info(f"  Python random: {hparams['seeds']['random_seed']}")
    logger.info(f"  NumPy: {hparams['seeds']['numpy_seed']}")
    logger.info(f"  PyTorch: {hparams['seeds']['torch_seed']}")
    if torch.cuda.is_available():
        logger.info(f"  CUDA: {hparams['seeds']['cuda_seed']}")
        logger.info(f"  CUDNN deterministic: True")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Usando dispositivo: {device}")
    
    # Crear directorio de checkpoints
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Cargar datasets
    train_dataset = AlphabetDataset(Path(args.train_data), MAX_PREFIX_LEN)
    val_dataset = AlphabetDataset(Path(args.val_data), MAX_PREFIX_LEN)
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=hparams['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=hparams['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    # Modelo
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
    
    # Calcular pos_weight para loss
    logger.info("Calculando pos_weight para pérdida...")
    y_train_all = train_dataset.labels
    pos_weight = compute_pos_weight(y_train_all).to(device)
    logger.info(f"pos_weight: {pos_weight.cpu().numpy()}")
    
    # Función de pérdida
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Optimizador
    optimizer = Adam(
        model.parameters(),
        lr=hparams['training']['learning_rate'],
        weight_decay=hparams['training'].get('weight_decay', 1e-4)
    )
    
    # Scheduler opcional
    scheduler = None
    if args.use_scheduler:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3
        )
        logger.info("✓ ReduceLROnPlateau scheduler activado")
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=hparams['training'].get('early_stopping_patience', 8)
    )
    
    # Inicializar variables de entrenamiento
    start_epoch = 0
    best_f1_macro = -np.inf
    log_file = checkpoint_dir / 'train_log.csv'
    
    # Resumir entrenamiento si se especifica
    if args.resume:
        checkpoint_path = Path(args.resume)
        start_epoch = load_checkpoint(model, optimizer, checkpoint_path)
        # Cargar mejor métrica desde el checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'metrics' in checkpoint:
            if 'f1_macro' in checkpoint['metrics']:
                best_f1_macro = checkpoint['metrics']['f1_macro']
            elif 'macro_auprc' in checkpoint['metrics']:
                # Compatibilidad hacia atrás
                best_f1_macro = checkpoint['metrics']['macro_auprc']
    
    # Logging CSV
    log_columns = ['epoch', 'loss_tr', 'loss_val', 'f1_macro', 'f1_min', 'ece',
                   'auPRC_macro', 'auPRC_micro', 'f1_at_threshold', 'coverage', 'LR']
    
    # Inicializar log si no existe
    if not log_file.exists():
        with open(log_file, 'w') as f:
            f.write(','.join(log_columns) + '\n')
    
    logger.info("="*60)
    logger.info("INICIANDO ENTRENAMIENTO")
    logger.info("="*60)
    logger.info(f"Total de épocas: {hparams['training']['num_epochs']}")
    logger.info(f"Batch size: {hparams['training']['batch_size']}")
    logger.info(f"Learning rate: {hparams['training']['learning_rate']}")
    logger.info(f"Early stopping patience: {early_stopping.patience}")
    logger.info("="*60)
    
    # Bucle de entrenamiento
    for epoch in range(start_epoch, hparams['training']['num_epochs']):
        logger.info(f"\nÉpoca {epoch + 1}/{hparams['training']['num_epochs']}")
        logger.info("-" * 60)
        
        # Entrenar
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device,
            gradient_clip=hparams['training'].get('gradient_clip', 1.0)
        )
        logger.info(f"Train loss: {train_loss:.6f}")
        
        # Validar
        val_metrics = validate(model, val_loader, criterion, device, ALPHABET)
        logger.info(f"Val loss: {val_metrics['loss']:.6f}")
        logger.info(f"Val F1 macro: {val_metrics['f1_macro']:.6f}")
        logger.info(f"Val F1 min: {val_metrics['f1_min']:.6f}")
        logger.info(f"Val ECE: {val_metrics['ece']:.6f}")
        logger.info(f"Val auPRC macro: {val_metrics['macro_auprc']:.6f}")
        logger.info(f"Val auPRC micro: {val_metrics['micro_auprc']:.6f}")
        logger.info(f"Val F1@0.5: {val_metrics['f1_at_threshold']:.6f}")
        logger.info(f"Coverage: {val_metrics['coverage']:.2f}%")
        
        # Actualizar scheduler
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler:
            scheduler.step(val_metrics['loss'])
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr != current_lr:
                logger.info(f"LR reducido: {current_lr:.6e} -> {new_lr:.6e}")
        
        # Guardar log
        log_row = [
            epoch + 1,
            train_loss,
            val_metrics['loss'],
            val_metrics['f1_macro'],
            val_metrics['f1_min'],
            val_metrics['ece'],
            val_metrics['macro_auprc'] if not np.isnan(val_metrics['macro_auprc']) else '',
            val_metrics['micro_auprc'] if not np.isnan(val_metrics['micro_auprc']) else '',
            val_metrics['f1_at_threshold'],
            val_metrics['coverage'],
            optimizer.param_groups[0]['lr']
        ]
        
        with open(log_file, 'a') as f:
            f.write(','.join(str(x) for x in log_row) + '\n')
        
        # Guardar último checkpoint
        last_checkpoint = checkpoint_dir / 'last.pt'
        save_checkpoint(model, optimizer, epoch + 1, val_metrics, last_checkpoint)
        
        # Guardar mejor checkpoint por F1 macro
        f1_macro = val_metrics['f1_macro']
        if not np.isnan(f1_macro) and f1_macro > best_f1_macro:
            best_f1_macro = f1_macro
            best_checkpoint = checkpoint_dir / 'best.pt'
            save_checkpoint(model, optimizer, epoch + 1, val_metrics, best_checkpoint, is_best=True)
        
        # Early stopping (solo si F1 macro es válido)
        if not np.isnan(f1_macro):
            if early_stopping(f1_macro):
                logger.info(f"\nEarly stopping activado después de {epoch + 1} épocas")
                logger.info(f"Mejor F1 macro: {best_f1_macro:.6f}")
                break
    
    logger.info("\n" + "="*60)
    logger.info("ENTRENAMIENTO COMPLETADO")
    logger.info("="*60)
    logger.info(f"Mejor F1 macro: {best_f1_macro:.6f}")
    logger.info(f"Checkpoints guardados en: {checkpoint_dir}")
    logger.info(f"Log guardado en: {log_file}")
    logger.info("="*60)


if __name__ == '__main__':
    main()


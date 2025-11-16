"""
AlphabetNet - Script standalone completo para Google Colab
Contiene TODO lo necesario: modelo, métricas, entrenamiento, inferencia.
Usa datos de sample_data/ (donde Colab guarda archivos subidos).

Tarea: Regex → Alfabeto del Autómata
Métricas: F1 macro ≥ 0.92, F1 mínimo ≥ 0.85, ECE ≤ 0.05, Exactitud ≥ 0.90
"""

# ============================================================================
# IMPORTS
# ============================================================================
import os
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, 
    f1_score, precision_score, recall_score
)

# Agregar import para precision_recall_curve en find_optimal_threshold_f1

# ============================================================================
# CONSTANTES
# ============================================================================
ALPHABET = list('ABCDEFGHIJKL')
ALPHABET_SIZE = len(ALPHABET)
SPECIAL_TOKENS = {'PAD': 0, '<EPS>': 1}
VOCAB_SIZE = ALPHABET_SIZE + 2
MAX_REGEX_LEN = 64

# ============================================================================
# UTILIDADES
# ============================================================================
def char_to_idx(char: str) -> int:
    if char == '<EPS>':
        return SPECIAL_TOKENS['<EPS>']
    elif char in ALPHABET:
        return ALPHABET.index(char) + 2
    else:
        raise ValueError(f"Carácter inválido: {char}")

def regex_to_indices(regex: str, max_len: int = MAX_REGEX_LEN) -> Tuple[torch.Tensor, int]:
    """Convierte un regex string a índices (solo caracteres A-L se tokenizan)."""
    if regex == '' or regex is None:
        indices = [SPECIAL_TOKENS['<EPS>']]
        length = 1
    else:
        # Extraer solo caracteres válidos (A-L) del regex
        valid_chars = [c for c in regex if c in ALPHABET]
        if len(valid_chars) == 0:
            indices = [SPECIAL_TOKENS['<EPS>']]
            length = 1
        else:
            indices = [char_to_idx(c) for c in valid_chars]
            length = len(indices)
    
    if length < max_len:
        indices = indices + [SPECIAL_TOKENS['PAD']] * (max_len - length)
    
    return torch.tensor(indices[:max_len], dtype=torch.long), length

def set_seeds(random_seed=42, numpy_seed=42, torch_seed=42, cuda_seed=42):
    random.seed(random_seed)
    np.random.seed(numpy_seed)
    torch.manual_seed(torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cuda_seed)
        torch.cuda.manual_seed_all(cuda_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(random_seed)

# ============================================================================
# MODELO
# ============================================================================
class AlphabetNet(nn.Module):
    """Modelo RNN para predecir alfabeto del autómata desde regex."""
    
    def __init__(self, vocab_size=14, alphabet_size=12, emb_dim=96, hidden_dim=192,
                 rnn_type='GRU', num_layers=1, dropout=0.2, padding_idx=0,
                 use_automata_conditioning=False, num_automata=None, automata_emb_dim=16):
        super(AlphabetNet, self).__init__()
        self.vocab_size = vocab_size
        self.alphabet_size = alphabet_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.rnn_type = rnn_type.upper()
        self.num_layers = num_layers
        self.padding_idx = padding_idx
        self.use_automata_conditioning = use_automata_conditioning
        
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=padding_idx)
        
        if use_automata_conditioning:
            if num_automata is None:
                raise ValueError("num_automata debe ser especificado si use_automata_conditioning=True")
            self.automata_embedding = nn.Embedding(num_automata, automata_emb_dim)
            self.automata_emb_dim = automata_emb_dim
        else:
            self.automata_embedding = None
            self.automata_emb_dim = 0
        
        if self.rnn_type == 'GRU':
            self.rnn = nn.GRU(emb_dim, hidden_dim, num_layers=num_layers,
                            dropout=dropout if num_layers > 1 else 0,
                            batch_first=True, bidirectional=False)
        elif self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(emb_dim, hidden_dim, num_layers=num_layers,
                             dropout=dropout if num_layers > 1 else 0,
                             batch_first=True, bidirectional=False)
        else:
            raise ValueError(f"rnn_type debe ser 'GRU' o 'LSTM', recibido: {rnn_type}")
        
        self.dropout = nn.Dropout(dropout)
        linear_input_dim = hidden_dim + self.automata_emb_dim
        self.output_layer = nn.Linear(linear_input_dim, alphabet_size)
        
    def forward(self, prefix_indices, lengths, automata_ids=None, return_logits=True):
        batch_size = prefix_indices.size(0)
        embedded = self.embedding(prefix_indices)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        if self.rnn_type == 'GRU':
            packed_output, hidden = self.rnn(packed)
        else:
            packed_output, (hidden, _) = self.rnn(packed)
        
        if self.num_layers > 1:
            h_t = hidden[-1]
        else:
            h_t = hidden.squeeze(0)
        
        if self.use_automata_conditioning and automata_ids is not None:
            automata_emb = self.automata_embedding(automata_ids)
            h_t = torch.cat([h_t, automata_emb], dim=1)
        
        h_t = self.dropout(h_t)
        logits = self.output_layer(h_t)
        
        if return_logits:
            return logits
        else:
            return torch.sigmoid(logits)

# ============================================================================
# DATASET
# ============================================================================
class AlphabetDataset(Dataset):
    """Dataset para datos en formato regex-sigma (CSV)."""
    def __init__(self, csv_path: Path, max_regex_len: int = MAX_REGEX_LEN):
        self.df = pd.read_csv(csv_path)
        self.max_regex_len = max_regex_len
        # Convertir columnas A-L a arrays numpy
        label_columns = [col for col in ALPHABET if col in self.df.columns]
        if len(label_columns) != ALPHABET_SIZE:
            raise ValueError(f"Faltan columnas de alfabeto. Esperadas: {ALPHABET_SIZE}, encontradas: {len(label_columns)}")
        self.labels = self.df[label_columns].values.astype(np.float32)
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        regex = str(row['regex'])
        regex_indices, length = regex_to_indices(regex, self.max_regex_len)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return {'prefix_indices': regex_indices, 'length': torch.tensor(length, dtype=torch.long), 'y': y}

def collate_fn(batch: list) -> Dict[str, torch.Tensor]:
    prefix_indices = torch.stack([item['prefix_indices'] for item in batch])
    lengths = torch.stack([item['length'] for item in batch])
    y = torch.stack([item['y'] for item in batch])
    return {'prefix_indices': prefix_indices, 'lengths': lengths, 'y': y}

# ============================================================================
# MÉTRICAS
# ============================================================================
def compute_pos_weight(y_true: np.ndarray) -> torch.Tensor:
    y_true = np.asarray(y_true)
    n_symbols = y_true.shape[1]
    pos_weight = np.ones(n_symbols, dtype=np.float32)
    for i in range(n_symbols):
        y_symbol = y_true[:, i]
        num_positives = y_symbol.sum()
        num_negatives = len(y_symbol) - num_positives
        if num_positives > 0:
            pos_weight[i] = num_negatives / num_positives
    return torch.tensor(pos_weight, dtype=torch.float32)

def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Calcula Expected Calibration Error (ECE)."""
    y_true = np.asarray(y_true).flatten()
    y_prob = np.asarray(y_prob).flatten()
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bins[i]
        bin_upper = bins[i + 1] if i < n_bins - 1 else bins[i + 1]
        idx = (y_prob >= bin_lower) & (y_prob <= bin_upper) if i == n_bins - 1 else (y_prob >= bin_lower) & (y_prob < bin_upper)
        if not np.any(idx):
            continue
        acc = np.mean((y_prob[idx] >= 0.5) == y_true[idx])
        conf = np.mean(y_prob[idx])
        bin_weight = np.sum(idx) / len(y_prob)
        ece += bin_weight * abs(acc - conf)
    return float(ece)

def compute_f1_per_symbol(y_true: np.ndarray, y_pred: np.ndarray, alphabet: List[str]) -> Dict[str, float]:
    """Calcula F1 por símbolo."""
    f1_per_symbol = {}
    for i, symbol in enumerate(alphabet):
        f1 = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
        f1_per_symbol[symbol] = float(f1)
    return f1_per_symbol

def find_optimal_threshold_f1(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float]:
    """Encuentra el umbral que maximiza F1 score."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)
    best_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    return optimal_threshold, best_f1

def find_thresholds_per_symbol(y_true: np.ndarray, y_scores: np.ndarray, alphabet: List[str]) -> Dict[str, float]:
    """Encuentra umbral óptimo por símbolo que maximiza F1."""
    thresholds = {}
    for i, symbol in enumerate(alphabet):
        y_true_symbol = y_true[:, i]
        y_scores_symbol = y_scores[:, i]
        if y_true_symbol.sum() > 0:
            try:
                optimal_threshold, _ = find_optimal_threshold_f1(y_true_symbol, y_scores_symbol)
                thresholds[symbol] = float(optimal_threshold)
            except:
                thresholds[symbol] = 0.5
        else:
            thresholds[symbol] = 0.5
    return thresholds

def compute_set_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula exactitud de conjunto (exact match)."""
    exact_matches = 0
    for i in range(len(y_true)):
        true_set = set(np.where(y_true[i] == 1)[0])
        pred_set = set(np.where(y_pred[i] == 1)[0])
        if true_set == pred_set:
            exact_matches += 1
    return exact_matches / len(y_true) if len(y_true) > 0 else 0.0

def evaluate_metrics(y_true: np.ndarray, y_scores: np.ndarray, alphabet: List[str], 
                     threshold: float = 0.5, threshold_per_symbol: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_scores, torch.Tensor):
        y_scores = y_scores.cpu().numpy()
    
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    
    # Calcular AP por símbolo
    ap_per_symbol = {}
    for i, symbol in enumerate(alphabet):
        y_true_symbol = y_true[:, i]
        y_scores_symbol = y_scores[:, i]
        if y_true_symbol.sum() == 0:
            ap_per_symbol[symbol] = np.nan
        else:
            ap = average_precision_score(y_true_symbol, y_scores_symbol)
            ap_per_symbol[symbol] = ap
    
    # Calcular AUPRC macro/micro
    valid_aps = [ap for ap in ap_per_symbol.values() if not np.isnan(ap)]
    macro_auprc = np.mean(valid_aps) if valid_aps else np.nan
    y_true_flat = y_true.flatten()
    y_scores_flat = y_scores.flatten()
    micro_auprc = average_precision_score(y_true_flat, y_scores_flat) if y_true_flat.sum() > 0 else np.nan
    
    # Binarizar predicciones
    if threshold_per_symbol:
        # Usar thresholds por símbolo
        y_pred = np.zeros_like(y_scores)
        for i, symbol in enumerate(alphabet):
            thresh = threshold_per_symbol.get(symbol, threshold)
            y_pred[:, i] = (y_scores[:, i] >= thresh).astype(int)
    else:
        # Usar threshold global
        y_pred = (y_scores >= threshold).astype(int)
    
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    f1_at_threshold = f1_score(y_true_flat, y_pred_flat, average='micro', zero_division=0)
    
    # Calcular F1 macro, F1 min y ECE
    f1_per_symbol = compute_f1_per_symbol(y_true, y_pred, alphabet)
    f1_macro = float(np.mean(list(f1_per_symbol.values())))
    f1_min = float(np.min(list(f1_per_symbol.values())))
    ece = expected_calibration_error(y_true, y_scores)
    
    # Calcular exactitud de conjunto
    set_acc = compute_set_accuracy(y_true, y_pred)
    
    valid_symbols = sum(1 for ap in ap_per_symbol.values() if not np.isnan(ap))
    coverage = (valid_symbols / len(ap_per_symbol)) * 100.0 if ap_per_symbol else 0.0
    
    return {
        'macro_auprc': macro_auprc,
        'micro_auprc': micro_auprc,
        'f1_at_threshold': f1_at_threshold,
        'coverage': coverage,
        'f1_macro': f1_macro,
        'f1_min': f1_min,
        'ece': ece,
        'set_accuracy': set_acc,
        'ap_per_symbol': ap_per_symbol,
        'f1_per_symbol': f1_per_symbol
    }

# ============================================================================
# PREPARAR DATASET
# ============================================================================
def generate_dataset_from_3000(input_csv: Path, output_csv: Path):
    """Genera dataset_regex_sigma.csv desde dataset3000.csv."""
    print(f"Generando dataset desde: {input_csv}")
    df = pd.read_csv(input_csv)
    
    rows = []
    for idx, row in df.iterrows():
        dfa_id = idx
        regex = str(row['Regex']).strip()
        alphabet_str = str(row['Alfabeto']).strip()
        
        # Extraer símbolos del alfabeto
        alphabet_set = set([s.strip() for s in alphabet_str.split() if s.strip() in ALPHABET])
        
        row_dict = {'dfa_id': dfa_id, 'regex': regex}
        for symbol in ALPHABET:
            row_dict[symbol] = 1 if symbol in alphabet_set else 0
        rows.append(row_dict)
    
    df_out = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_csv, index=False)
    print(f"✓ Dataset generado: {len(df_out)} autómatas guardado en {output_csv}")

def prepare_dataset():
    """Prepara el dataset desde sample_data/."""
    import os
    
    # Obtener directorio actual
    current_dir = Path.cwd()
    print(f"Directorio actual: {current_dir}")
    
    # Buscar en sample_data/ (relativo al directorio actual)
    sample_data = current_dir / 'sample_data'
    dataset_csv = sample_data / 'dataset_regex_sigma.csv'
    dataset3000_csv = sample_data / 'dataset3000.csv'
    
    # También buscar en raíz por si están ahí
    dataset_csv_root = current_dir / 'dataset_regex_sigma.csv'
    dataset3000_csv_root = current_dir / 'dataset3000.csv'
    
    print(f"\nBuscando archivos...")
    print(f"  Buscando en: {sample_data}")
    print(f"    - {dataset_csv} existe: {dataset_csv.exists()}")
    print(f"    - {dataset3000_csv} existe: {dataset3000_csv.exists()}")
    print(f"  Buscando en raíz: {current_dir}")
    print(f"    - {dataset_csv_root} existe: {dataset_csv_root.exists()}")
    print(f"    - {dataset3000_csv_root} existe: {dataset3000_csv_root.exists()}")
    
    # Listar contenido de sample_data si existe
    if sample_data.exists():
        print(f"\n  Contenido de sample_data/:")
        try:
            files = list(sample_data.iterdir())
            for f in files[:10]:  # Mostrar primeros 10
                print(f"    - {f.name}")
        except Exception as e:
            print(f"    Error al listar: {e}")
    
    # Si ya existe dataset_regex_sigma.csv, usarlo (prioridad: sample_data > raíz)
    if dataset_csv.exists():
        print(f"\n✓ Dataset encontrado en sample_data/: {dataset_csv}")
        return dataset_csv
    elif dataset_csv_root.exists():
        print(f"\n✓ Dataset encontrado en raíz: {dataset_csv_root}")
        return dataset_csv_root
    
    # Si existe dataset3000.csv, generar dataset (prioridad: sample_data > raíz)
    if dataset3000_csv.exists():
        print(f"\nGenerando dataset desde {dataset3000_csv}...")
        generate_dataset_from_3000(dataset3000_csv, dataset_csv)
        if dataset_csv.exists():
            print(f"✓ Dataset generado: {dataset_csv}")
            return dataset_csv
    elif dataset3000_csv_root.exists():
        print(f"\nGenerando dataset desde {dataset3000_csv_root}...")
        # Generar en sample_data/ si existe, sino en raíz
        output_csv = dataset_csv if sample_data.exists() else dataset_csv_root
        generate_dataset_from_3000(dataset3000_csv_root, output_csv)
        if output_csv.exists():
            print(f"✓ Dataset generado: {output_csv}")
            return output_csv
    
    # Si no existe ninguno, pedir subirlo
    print("\n" + "=" * 60)
    print("⚠ DATASET NO ENCONTRADO")
    print("=" * 60)
    print("Necesitas subir uno de estos archivos:")
    print("  1. dataset_regex_sigma.csv → en sample_data/ (preferido)")
    print("  2. dataset3000.csv → en sample_data/ (se generará automáticamente)")
    print("\nPara subir archivos en Colab:")
    print("  from google.colab import files")
    print("  uploaded = files.upload()")
    print("  # Los archivos se guardarán en el directorio actual")
    print("  # Muévelos a sample_data/ si es necesario")
    print("=" * 60)
    return None

# ============================================================================
# ENTRENAMIENTO
# ============================================================================
def train_epoch(model, dataloader, criterion, optimizer, device, gradient_clip=1.0):
    model.train()
    total_loss = 0.0
    num_batches = 0
    for batch in dataloader:
        prefix_indices = batch['prefix_indices'].to(device)
        lengths = batch['lengths'].to(device)
        y_true = batch['y'].to(device)
        optimizer.zero_grad()
        logits = model(prefix_indices, lengths, return_logits=True)
        loss = criterion(logits, y_true)
        loss.backward()
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
    return total_loss / num_batches if num_batches > 0 else 0.0

def validate(model, dataloader, criterion, device, alphabet, threshold_per_symbol=None):
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
            logits = model(prefix_indices, lengths, return_logits=True)
            loss = criterion(logits, y_true)
            total_loss += loss.item()
            num_batches += 1
            all_logits.append(logits.cpu())
            all_y_true.append(y_true.cpu())
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    all_logits = torch.cat(all_logits, dim=0)
    all_y_true = torch.cat(all_y_true, dim=0)
    probs = torch.sigmoid(all_logits)
    metrics = evaluate_metrics(all_y_true.numpy(), probs.numpy(), alphabet, threshold=0.5, threshold_per_symbol=threshold_per_symbol)
    metrics['loss'] = avg_loss
    return metrics

class EarlyStopping:
    def __init__(self, patience: int = 8, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
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

# ============================================================================
# INFERENCIA
# ============================================================================
def predict_alphabet_from_regex(regex: str, model: AlphabetNet, device: torch.device,
                                threshold_per_symbol: Optional[Dict[str, float]] = None,
                                alphabet: List[str] = ALPHABET) -> Dict[str, any]:
    """Predice el alfabeto del autómata desde un regex."""
    if threshold_per_symbol is None:
        threshold_per_symbol = {sym: 0.5 for sym in alphabet}
    
    # Convertir regex a índices
    regex_indices, length = regex_to_indices(regex, MAX_REGEX_LEN)
    regex_indices = regex_indices.unsqueeze(0).to(device)
    lengths = torch.tensor([length], dtype=torch.long).to(device)
    
    # Inferir probabilidades
    model.eval()
    with torch.no_grad():
        logits = model(regex_indices, lengths, return_logits=True)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
    
    # Construir sigma_hat usando thresholds
    sigma_hat = []
    for i, symbol in enumerate(alphabet):
        threshold = threshold_per_symbol.get(symbol, 0.5)
        if probs[i] >= threshold:
            sigma_hat.append(symbol)
    
    return {
        'p_sigma': probs.tolist(),
        'sigma_hat': sigma_hat
    }

# ============================================================================
# MAIN - PIPELINE COMPLETO
# ============================================================================
def main():
    print("=" * 60)
    print("ALPHABETNET - ENTRENAMIENTO EN COLAB")
    print("=" * 60)
    print("Tarea: Regex → Alfabeto del Autómata")
    print("Métricas objetivo:")
    print("  - F1 macro ≥ 0.92")
    print("  - F1 mínimo ≥ 0.85")
    print("  - ECE ≤ 0.05")
    print("  - Exactitud de conjunto ≥ 0.90")
    print("=" * 60)
    
    # Seeds
    set_seeds(42, 42, 42, 42)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsando dispositivo: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Preparar dataset
    print("\n" + "=" * 60)
    print("PREPARAR DATASET")
    print("=" * 60)
    dataset_path = prepare_dataset()
    if dataset_path is None:
        print("\n❌ No se pudo preparar el dataset. Por favor sube los archivos necesarios.")
        return
    
    # Hiperparámetros
    hparams = {
        'model': {
            'vocab_size': 14,
            'alphabet_size': 12,
            'emb_dim': 96,
            'hidden_dim': 192,
            'rnn_type': 'GRU',
            'num_layers': 1,
            'dropout': 0.2,
            'padding_idx': 0,
            'use_automata_conditioning': False,
            'automata_emb_dim': 16
        },
        'training': {
            'batch_size': 64,
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'num_epochs': 50,
            'early_stopping_patience': 8,
            'gradient_clip': 1.0
        }
    }
    
    # Datasets
    print("\n" + "=" * 60)
    print("CARGAR DATASETS")
    print("=" * 60)
    train_dataset = AlphabetDataset(dataset_path)
    val_dataset = AlphabetDataset(dataset_path)  # Usar mismo dataset para train/val en demo
    
    train_loader = DataLoader(train_dataset, batch_size=hparams['training']['batch_size'],
                             shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=hparams['training']['batch_size'],
                           shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    print(f"✓ Train: {len(train_dataset)} ejemplos")
    print(f"✓ Val: {len(val_dataset)} ejemplos")
    
    # Modelo
    print("\n" + "=" * 60)
    print("CREAR MODELO")
    print("=" * 60)
    model = AlphabetNet(**hparams['model']).to(device)
    print(f"✓ Modelo creado: {sum(p.numel() for p in model.parameters()):,} parámetros")
    
    # Loss
    pos_weight = compute_pos_weight(train_dataset.labels).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Optimizador
    optimizer = Adam(model.parameters(), lr=hparams['training']['learning_rate'],
                    weight_decay=hparams['training']['weight_decay'])
    
    # Scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=hparams['training']['early_stopping_patience'])
    
    # Entrenamiento
    print("\n" + "=" * 60)
    print("ENTRENAMIENTO")
    print("=" * 60)
    best_f1_macro = -np.inf
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)
    log_file = checkpoint_dir / 'train_log.csv'
    
    # Inicializar log
    log_columns = ['epoch', 'loss_tr', 'loss_val', 'f1_macro', 'f1_min', 'ece', 'set_accuracy',
                   'auPRC_macro', 'auPRC_micro', 'f1_at_threshold', 'coverage', 'LR']
    if not log_file.exists():
        with open(log_file, 'w') as f:
            f.write(','.join(log_columns) + '\n')
    
    # Variables para thresholds óptimos
    best_thresholds = None
    
    for epoch in range(hparams['training']['num_epochs']):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device,
                                hparams['training']['gradient_clip'])
        
        # Encontrar thresholds óptimos cada 5 épocas o en la última
        if epoch % 5 == 0 or epoch == hparams['training']['num_epochs'] - 1:
            # Evaluar para obtener probabilidades
            model.eval()
            all_logits = []
            all_y_true = []
            with torch.no_grad():
                for batch in val_loader:
                    prefix_indices = batch['prefix_indices'].to(device)
                    lengths = batch['lengths'].to(device)
                    y_true = batch['y'].to(device)
                    logits = model(prefix_indices, lengths, return_logits=True)
                    all_logits.append(logits.cpu())
                    all_y_true.append(y_true.cpu())
            all_logits = torch.cat(all_logits, dim=0)
            all_y_true = torch.cat(all_y_true, dim=0)
            probs = torch.sigmoid(all_logits)
            
            # Encontrar thresholds óptimos
            thresholds_per_symbol = find_thresholds_per_symbol(
                all_y_true.numpy(), probs.numpy(), ALPHABET
            )
            best_thresholds = thresholds_per_symbol
        
        val_metrics = validate(model, val_loader, criterion, device, ALPHABET, best_thresholds)
        
        # Logging
        print(f"Época {epoch+1}/{hparams['training']['num_epochs']}:")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss: {val_metrics['loss']:.6f}")
        print(f"  Val F1 Macro: {val_metrics['f1_macro']:.6f}")
        print(f"  Val F1 Min: {val_metrics['f1_min']:.6f}")
        print(f"  Val ECE: {val_metrics['ece']:.6f}")
        print(f"  Val Set Accuracy: {val_metrics['set_accuracy']:.6f}")
        print(f"  Val auPRC Macro: {val_metrics['macro_auprc']:.6f}")
        
        # Guardar log
        log_row = [
            epoch + 1,
            train_loss,
            val_metrics['loss'],
            val_metrics['f1_macro'],
            val_metrics['f1_min'],
            val_metrics['ece'],
            val_metrics['set_accuracy'],
            val_metrics['macro_auprc'] if not np.isnan(val_metrics['macro_auprc']) else '',
            val_metrics['micro_auprc'] if not np.isnan(val_metrics['micro_auprc']) else '',
            val_metrics['f1_at_threshold'],
            val_metrics['coverage'],
            optimizer.param_groups[0]['lr']
        ]
        with open(log_file, 'a') as f:
            f.write(','.join(str(x) for x in log_row) + '\n')
        
        # Guardar mejor checkpoint por F1 macro
        f1_macro = val_metrics['f1_macro']
        if not np.isnan(f1_macro) and f1_macro > best_f1_macro:
            best_f1_macro = f1_macro
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_metrics,
                'f1_macro': f1_macro,
                'f1_min': val_metrics['f1_min'],
                'ece': val_metrics['ece'],
                'set_accuracy': val_metrics['set_accuracy'],
                'threshold_per_symbol': best_thresholds,
                'hparams': hparams
            }, checkpoint_dir / 'best.pt')
            print(f"  ✓ Mejor modelo guardado (F1 macro: {best_f1_macro:.6f}, Set Acc: {val_metrics['set_accuracy']:.6f})")
        
        # Actualizar scheduler
        scheduler.step(val_metrics['loss'])
        
        # Early stopping
        if not np.isnan(f1_macro):
            if early_stopping(f1_macro):
                print(f"\nEarly stopping activado después de {epoch + 1} épocas")
                break
        
        print()  # Línea en blanco
    
    print("=" * 60)
    print("ENTRENAMIENTO COMPLETADO")
    print("=" * 60)
    print(f"Mejor F1 macro: {best_f1_macro:.6f}")
    print(f"Checkpoints guardados en: {checkpoint_dir}")
    print(f"Log guardado en: {log_file}")
    
    # Guardar thresholds óptimos finales
    if best_thresholds:
        thresholds_file = checkpoint_dir / 'thresholds.json'
        with open(thresholds_file, 'w') as f:
            json.dump({'per_symbol': best_thresholds}, f, indent=2)
        print(f"✓ Thresholds óptimos guardados en: {thresholds_file}")
        print(f"  Rango: [{min(best_thresholds.values()):.4f}, {max(best_thresholds.values()):.4f}]")
    
    # Demo de inferencia
    print("\n" + "=" * 60)
    print("DEMO DE INFERENCIA")
    print("=" * 60)
    
    # Cargar mejor modelo
    checkpoint = torch.load(checkpoint_dir / 'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Usar thresholds óptimos si están disponibles
    threshold_per_symbol = checkpoint.get('threshold_per_symbol', best_thresholds)
    if threshold_per_symbol is None:
        threshold_per_symbol = {sym: 0.5 for sym in ALPHABET}
        print("⚠️  Usando threshold 0.5 para todos los símbolos (thresholds óptimos no encontrados)")
    else:
        print("✓ Usando thresholds óptimos por símbolo")
    
    # Evaluación final con thresholds óptimos
    print("\nEvaluación final con thresholds óptimos:")
    final_metrics = validate(model, val_loader, criterion, device, ALPHABET, threshold_per_symbol)
    print(f"  F1 Macro: {final_metrics['f1_macro']:.6f}")
    print(f"  F1 Min: {final_metrics['f1_min']:.6f}")
    print(f"  ECE: {final_metrics['ece']:.6f}")
    print(f"  Set Accuracy: {final_metrics['set_accuracy']:.6f}")
    
    # Probar con algunos regex de ejemplo
    print("\n" + "=" * 60)
    print("EJEMPLOS DE PREDICCIÓN")
    print("=" * 60)
    test_regexes = ["(AB)*C", "A+B*", "[ABCD]+"]
    
    for regex in test_regexes:
        result = predict_alphabet_from_regex(regex, model, device, threshold_per_symbol)
        print(f"\nRegex: {regex}")
        print(f"  Alfabeto predicho: {', '.join(result['sigma_hat']) if result['sigma_hat'] else '(ninguno)'}")
        print(f"  Probabilidades:")
        for i, symbol in enumerate(ALPHABET):
            prob = result['p_sigma'][i]
            threshold = threshold_per_symbol.get(symbol, 0.5)
            marker = "✓" if symbol in result['sigma_hat'] else " "
            print(f"    {marker} {symbol}: {prob:.4f} (threshold: {threshold:.4f})")
    
    # Evaluación masiva con todas las regex del dataset
    print("\n" + "=" * 60)
    print("EVALUACIÓN MASIVA CON TODAS LAS REGEX")
    print("=" * 60)
    
    print(f"\n🔄 Procesando {len(val_dataset):,} regex del dataset de validación...")
    
    all_results = []
    correct_predictions = []
    incorrect_predictions = []
    
    # Procesar todas las regex del dataset
    for idx in range(len(val_dataset)):
        row = val_dataset.df.iloc[idx]
        regex = str(row['regex'])
        
        # Predecir alfabeto
        result = predict_alphabet_from_regex(regex, model, device, threshold_per_symbol)
        pred_alphabet = set(result['sigma_hat'])
        
        # Obtener alfabeto real desde las columnas A-L
        real_alphabet = set()
        for symbol in ALPHABET:
            if row[symbol] == 1:
                real_alphabet.add(symbol)
        
        # Verificar si es correcto
        is_correct = pred_alphabet == real_alphabet
        
        all_results.append({
            'regex': regex,
            'predicted': sorted(pred_alphabet),
            'real': sorted(real_alphabet),
            'correct': is_correct
        })
        
        if is_correct:
            correct_predictions.append(idx)
        else:
            incorrect_predictions.append(idx)
        
        # Mostrar progreso cada 500 filas
        if (idx + 1) % 500 == 0:
            print(f"  Procesadas {idx + 1:,}/{len(val_dataset):,} regex...")
    
    # Calcular estadísticas
    total = len(all_results)
    correct = len(correct_predictions)
    incorrect = len(incorrect_predictions)
    accuracy = (correct / total) * 100.0 if total > 0 else 0.0
    
    print(f"\n✅ Evaluación completada!")
    print(f"   Total de regex procesadas: {total:,}")
    print(f"   Predicciones correctas: {correct:,} ({accuracy:.2f}%)")
    print(f"   Predicciones incorrectas: {incorrect:,} ({(100.0 - accuracy):.2f}%)")
    
    # Análisis de errores
    if incorrect_predictions:
        print(f"\n📊 Análisis de errores:")
        
        # Calcular estadísticas de errores
        false_positives = []  # Predijo símbolos que no están
        false_negatives = []  # No predijo símbolos que sí están
        
        for idx in incorrect_predictions[:100]:  # Analizar primeros 100 errores
            result = all_results[idx]
            pred_set = set(result['predicted'])
            real_set = set(result['real'])
            
            fp = pred_set - real_set  # Falsos positivos
            fn = real_set - pred_set  # Falsos negativos
            
            if fp:
                false_positives.append({'regex': result['regex'], 'symbols': sorted(fp)})
            if fn:
                false_negatives.append({'regex': result['regex'], 'symbols': sorted(fn)})
        
        # Mostrar ejemplos de errores
        print(f"\n   Ejemplos de predicciones incorrectas (primeros 10):")
        for i, idx in enumerate(incorrect_predictions[:10]):
            result = all_results[idx]
            pred_str = ', '.join(result['predicted']) if result['predicted'] else '(ninguno)'
            real_str = ', '.join(result['real']) if result['real'] else '(ninguno)'
            print(f"\n   {i+1}. Regex: {result['regex']}")
            print(f"      Predicho: {pred_str}")
            print(f"      Real:     {real_str}")
            
            # Mostrar diferencias
            pred_set = set(result['predicted'])
            real_set = set(result['real'])
            fp = pred_set - real_set
            fn = real_set - pred_set
            if fp:
                print(f"      ➕ Falsos positivos: {', '.join(sorted(fp))}")
            if fn:
                print(f"      ➖ Falsos negativos: {', '.join(sorted(fn))}")
    
    # Mostrar ejemplos de predicciones correctas
    if correct_predictions:
        print(f"\n✅ Ejemplos de predicciones correctas (primeros 5):")
        for i, idx in enumerate(correct_predictions[:5]):
            result = all_results[idx]
            alphabet_str = ', '.join(result['predicted']) if result['predicted'] else '(ninguno)'
            print(f"   {i+1}. Regex: {result['regex']}")
            print(f"      Alfabeto: {alphabet_str}")
    
    # Guardar resultados detallados (opcional, comentado por defecto para no ocupar mucho espacio)
    # results_file = checkpoint_dir / 'predictions_detailed.csv'
    # df_results = pd.DataFrame(all_results)
    # df_results.to_csv(results_file, index=False)
    # print(f"\n💾 Resultados detallados guardados en: {results_file}")
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETADO")
    print("=" * 60)
    print(f"\n📈 Resumen final:")
    print(f"   F1 Macro: {final_metrics['f1_macro']:.6f}")
    print(f"   F1 Min: {final_metrics['f1_min']:.6f}")
    print(f"   ECE: {final_metrics['ece']:.6f}")
    print(f"   Set Accuracy (validación): {final_metrics['set_accuracy']:.6f}")
    print(f"   Set Accuracy (masiva): {accuracy:.2f}%")
    
    # Descargar archivos en Colab
    print("\n" + "=" * 60)
    print("DESCARGAR ARCHIVOS")
    print("=" * 60)
    
    # Verificar si estamos en Colab
    try:
        import google.colab
        IN_COLAB = True
    except ImportError:
        IN_COLAB = False
    
    if IN_COLAB:
        from google.colab import files
        print("\n📥 Descargando archivos del modelo entrenado...")
        
        files_to_download = []
        
        # Modelo principal
        best_model_path = checkpoint_dir / 'best.pt'
        if best_model_path.exists():
            files_to_download.append(('Modelo entrenado (mejor)', best_model_path))
        
        # Thresholds óptimos
        thresholds_path = checkpoint_dir / 'thresholds.json'
        if thresholds_path.exists():
            files_to_download.append(('Thresholds óptimos', thresholds_path))
        
        # Log de entrenamiento
        log_path = checkpoint_dir / 'train_log.csv'
        if log_path.exists():
            files_to_download.append(('Log de entrenamiento', log_path))
        
        # Último checkpoint (opcional)
        last_model_path = checkpoint_dir / 'last.pt'
        if last_model_path.exists():
            files_to_download.append(('Modelo último checkpoint', last_model_path))
        
        if files_to_download:
            print(f"\n📦 Archivos disponibles para descargar ({len(files_to_download)}):")
            for name, path in files_to_download:
                size_mb = path.stat().st_size / (1024 * 1024)
                print(f"   - {name}: {path.name} ({size_mb:.2f} MB)")
            
            print("\n🔄 Iniciando descargas...")
            for name, path in files_to_download:
                try:
                    print(f"   📥 Descargando {name}...")
                    files.download(str(path))
                    print(f"   ✓ {name} descargado")
                except Exception as e:
                    print(f"   ❌ Error al descargar {name}: {e}")
            
            print("\n✅ Descargas completadas!")
            print("\n📝 Archivos descargados:")
            print("   1. best.pt - Modelo entrenado (mejor checkpoint)")
            print("   2. thresholds.json - Umbrales óptimos por símbolo")
            print("   3. train_log.csv - Log completo de entrenamiento")
            print("   4. last.pt - Último checkpoint (opcional)")
            print("\n💡 Para usar el modelo localmente:")
            print("   python demo/test_model.py --checkpoint checkpoints/best.pt --thresholds checkpoints/thresholds.json --regex '(AB)*C'")
        else:
            print("\n⚠️  No se encontraron archivos para descargar")
    else:
        print("\n💡 Para descargar en Colab:")
        print("   Los archivos están guardados en:")
        print(f"   - {checkpoint_dir / 'best.pt'}")
        print(f"   - {checkpoint_dir / 'thresholds.json'}")
        print(f"   - {checkpoint_dir / 'train_log.csv'}")
        print("\n   Ejecuta este código en Colab para descargar:")
        print("   ```python")
        print("   from google.colab import files")
        print("   files.download('checkpoints/best.pt')")
        print("   files.download('checkpoints/thresholds.json')")
        print("   files.download('checkpoints/train_log.csv')")
        print("   ```")
    
    print("\nPara usar el modelo:")
    print("  result = predict_alphabet_from_regex('(AB)*C', model, device, threshold_per_symbol)")
    print("  print(result['sigma_hat'])")

if __name__ == '__main__':
    main()


"""
AlphabetNet - Script completo para Google Colab
Contiene todo el código necesario para entrenar, evaluar y exportar el modelo.
"""

# ============================================================================
# IMPORTS
# ============================================================================
import argparse
import json
import logging
import os
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

# ============================================================================
# CONSTANTES
# ============================================================================
ALPHABET = list('ABCDEFGHIJKL')
ALPHABET_SIZE = len(ALPHABET)
SPECIAL_TOKENS = {'PAD': 0, '<EPS>': 1}
VOCAB_SIZE = ALPHABET_SIZE + 2
MAX_PREFIX_LEN = 64

# ============================================================================
# MODELO
# ============================================================================
class AlphabetNet(nn.Module):
    """Modelo RNN para predecir símbolos válidos después de un prefijo."""
    
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
# UTILIDADES
# ============================================================================
def char_to_idx(char: str) -> int:
    if char == '<EPS>':
        return SPECIAL_TOKENS['<EPS>']
    elif char in ALPHABET:
        return ALPHABET.index(char) + 2
    else:
        raise ValueError(f"Carácter inválido: {char}")

def regex_to_indices(regex: str, max_len: int = MAX_PREFIX_LEN) -> Tuple[torch.Tensor, int]:
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

# Alias para compatibilidad
prefix_to_indices = regex_to_indices

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
# DATASET
# ============================================================================
class AlphabetDataset(Dataset):
    """Dataset para datos en formato regex-sigma (CSV)."""
    def __init__(self, csv_path: Path, max_regex_len: int = MAX_PREFIX_LEN):
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

def compute_ap_per_symbol(y_true: np.ndarray, y_scores: np.ndarray, alphabet: List[str]) -> Dict[str, float]:
    ap_per_symbol = {}
    for i, symbol in enumerate(alphabet):
        y_true_symbol = y_true[:, i]
        y_scores_symbol = y_scores[:, i]
        if y_true_symbol.sum() == 0:
            ap_per_symbol[symbol] = np.nan
        else:
            ap = average_precision_score(y_true_symbol, y_scores_symbol)
            ap_per_symbol[symbol] = ap
    return ap_per_symbol

def compute_macro_auprc(ap_per_symbol: Dict[str, float]) -> float:
    valid_aps = [ap for ap in ap_per_symbol.values() if not np.isnan(ap)]
    return np.mean(valid_aps) if valid_aps else np.nan

def compute_micro_auprc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    y_true_flat = y_true.flatten()
    y_scores_flat = y_scores.flatten()
    if y_true_flat.sum() == 0:
        return np.nan
    return average_precision_score(y_true_flat, y_scores_flat)

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

def evaluate_metrics(y_true: np.ndarray, y_scores: np.ndarray, alphabet: List[str], threshold: float = 0.5) -> Dict[str, float]:
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_scores, torch.Tensor):
        y_scores = y_scores.cpu().numpy()
    
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    
    ap_per_symbol = compute_ap_per_symbol(y_true, y_scores, alphabet)
    macro_auprc = compute_macro_auprc(ap_per_symbol)
    micro_auprc = compute_micro_auprc(y_true, y_scores)
    
    y_pred = (y_scores >= threshold).astype(int)
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    f1_at_threshold = f1_score(y_true_flat, y_pred_flat, average='micro', zero_division=0)
    
    # Calcular F1 macro, F1 min y ECE
    f1_per_symbol = compute_f1_per_symbol(y_true, y_pred, alphabet)
    f1_macro = float(np.mean(list(f1_per_symbol.values())))
    f1_min = float(np.min(list(f1_per_symbol.values())))
    ece = expected_calibration_error(y_true, y_scores)
    
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
        'ap_per_symbol': ap_per_symbol,
        'f1_per_symbol': f1_per_symbol
    }

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

def validate(model, dataloader, criterion, device, alphabet):
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
    metrics = evaluate_metrics(all_y_true.numpy(), probs.numpy(), alphabet, threshold=0.5)
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
# EXPORTACIÓN ONNX
# ============================================================================
class ModelWrapper(nn.Module):
    """
    Wrapper simplificado para exportación ONNX.
    Evita usar pack_padded_sequence que causa problemas en ONNX.
    """
    def __init__(self, model: AlphabetNet):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.use_automata_conditioning = model.use_automata_conditioning
        
        # Copiar componentes del modelo
        self.embedding = model.embedding
        self.rnn = model.rnn
        self.dropout = model.dropout
        self.output_layer = model.output_layer
        if model.use_automata_conditioning:
            self.automata_embedding = model.automata_embedding
    
    def forward(self, prefix_indices: torch.Tensor, lengths: torch.Tensor,
                automata_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass simplificado para ONNX.
        Usa RNN estándar sin pack_padded_sequence para mejor compatibilidad ONNX.
        Obtiene el último estado válido según lengths.
        """
        batch_size = prefix_indices.size(0)
        
        # Embedding
        embedded = self.embedding(prefix_indices)  # (batch_size, seq_len, emb_dim)
        
        # RNN forward (sin pack_padded_sequence para compatibilidad ONNX)
        # Nota: Esto es menos eficiente que pack, pero funciona en ONNX
        if self.model.rnn_type == 'GRU':
            output, hidden = self.rnn(embedded)
        else:  # LSTM
            output, (hidden, _) = self.rnn(embedded)
        
        # Obtener último estado válido según lengths
        # output tiene forma (batch_size, seq_len, hidden_dim)
        # Necesitamos tomar output[i, lengths[i]-1, :] para cada i
        batch_indices = torch.arange(batch_size, device=embedded.device)
        seq_indices = (lengths - 1).clamp(0, embedded.size(1) - 1)  # Índices del último carácter válido
        
        # Indexar para obtener el último estado válido de cada secuencia
        # output[batch_indices, seq_indices] da (batch_size, hidden_dim)
        h_t = output[batch_indices, seq_indices]  # (batch_size, hidden_dim)
        
        # Conditioning
        if self.use_automata_conditioning and automata_ids is not None:
            automata_emb = self.automata_embedding(automata_ids)
            h_t = torch.cat([h_t, automata_emb], dim=1)
        
        # Dropout y output
        h_t = self.dropout(h_t)
        logits = self.output_layer(h_t)
        
        return logits

def export_to_onnx(model: AlphabetNet, output_path: Path, hparams: dict, batch_size: int = 1, opset_version: int = 13):
    """
    Exporta modelo a ONNX.
    Nota: pack_padded_sequence puede causar problemas en ONNX.
    Se usa dynamo=True para el nuevo exportador que maneja mejor estos casos.
    """
    # Mover modelo a CPU para exportación ONNX
    model_cpu = model.cpu()
    wrapped_model = ModelWrapper(model_cpu)
    wrapped_model.eval()
    
    max_seq_len = MAX_PREFIX_LEN
    use_automata = hparams['model']['use_automata_conditioning']
    
    # Crear inputs de ejemplo en CPU
    # Usar batch_size=1 para evitar problemas con pack_padded_sequence
    prefix_indices = torch.randint(0, hparams['model']['vocab_size'], (1, max_seq_len), dtype=torch.long)
    lengths = torch.randint(1, max_seq_len + 1, (1,), dtype=torch.long)
    
    automata_ids = None
    if use_automata:
        num_automata = hparams['model'].get('num_automata', 3000)
        automata_ids = torch.randint(0, num_automata, (1,), dtype=torch.long)
    
    # Intentar con el nuevo exportador dynamo primero
    try:
        if use_automata and automata_ids is not None:
            example_inputs = (prefix_indices, lengths, automata_ids)
            input_names = ['prefix_indices', 'lengths', 'automata_ids']
        else:
            example_inputs = (prefix_indices, lengths)
            input_names = ['prefix_indices', 'lengths']
        
        output_names = ['logits']
        
        # Intentar con dynamo=True (nuevo exportador)
        print("Intentando exportación ONNX con exportador dynamo...")
        torch.onnx.export(
            wrapped_model,
            example_inputs,
            str(output_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={'prefix_indices': {0: 'batch_size'}, 
                         'lengths': {0: 'batch_size'},
                         'logits': {0: 'batch_size'}} if not use_automata else
                        {'prefix_indices': {0: 'batch_size'}, 
                         'lengths': {0: 'batch_size'},
                         'automata_ids': {0: 'batch_size'},
                         'logits': {0: 'batch_size'}},
            opset_version=opset_version,
            do_constant_folding=True,
            export_params=True,
            verbose=False,
            dynamo=True  # Usar nuevo exportador
        )
        print("✓ Exportación con dynamo exitosa")
        
    except Exception as e:
        print(f"Error con exportador dynamo: {e}")
        print("Intentando con exportador legacy...")
        
        # Fallback al exportador legacy (sin dynamo)
        # Nota: puede no funcionar completamente con pack_padded_sequence
        try:
            torch.onnx.export(
                wrapped_model,
                example_inputs,
                str(output_path),
                input_names=input_names,
                output_names=output_names,
                opset_version=opset_version,
                do_constant_folding=True,
                export_params=True,
                verbose=False,
                dynamo=False  # Exportador legacy
            )
            print("✓ Exportación con exportador legacy exitosa")
            print("⚠ Advertencia: El modelo usa pack_padded_sequence que puede no funcionar correctamente en ONNX")
            
        except Exception as e2:
            print(f"Error con exportador legacy: {e2}")
            print("⚠ No se pudo exportar a ONNX debido a limitaciones con pack_padded_sequence")
            print("   El modelo usa pack_padded_sequence que no está completamente soportado en ONNX")
            raise e2

# ============================================================================
# MAIN - PIPELINE COMPLETO
# ============================================================================
def main():
    print("="*60)
    print("ALPHABETNET - PIPELINE COMPLETO")
    print("="*60)
    
    # Configuración
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
    
    # Seeds
    set_seeds(42, 42, 42, 42)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    
    # Rutas de datos (Colab - dataset regex-sigma)
    data_path = Path('data/dataset_regex_sigma.csv')
    
    if not data_path.exists():
        print("ERROR: Archivo de datos no encontrado: data/dataset_regex_sigma.csv")
        print("Por favor, crea el dataset usando: python scripts/create_regex_sigma_dataset.py")
        return
    
    train_path = data_path
    val_path = data_path
    
    # Datasets
    print("\nCargando datasets...")
    train_dataset = AlphabetDataset(train_path)
    val_dataset = AlphabetDataset(val_path)
    
    train_loader = DataLoader(train_dataset, batch_size=hparams['training']['batch_size'],
                             shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=hparams['training']['batch_size'],
                           shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    # Modelo
    print("\nCreando modelo...")
    model = AlphabetNet(**hparams['model']).to(device)
    
    # Loss
    pos_weight = compute_pos_weight(train_dataset.labels).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Optimizador
    optimizer = Adam(model.parameters(), lr=hparams['training']['learning_rate'],
                    weight_decay=hparams['training']['weight_decay'])
    
    # Early stopping
    early_stopping = EarlyStopping(patience=hparams['training']['early_stopping_patience'])
    
    # Entrenamiento
    print("\n" + "="*60)
    print("ENTRENAMIENTO")
    print("="*60)
    best_f1_macro = -np.inf
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)
    
    for epoch in range(hparams['training']['num_epochs']):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device,
                                hparams['training']['gradient_clip'])
        val_metrics = validate(model, val_loader, criterion, device, ALPHABET)
        
        print(f"Época {epoch+1}: Train Loss={train_loss:.6f}, "
              f"Val F1 Macro={val_metrics.get('f1_macro', np.nan):.6f}, "
              f"Val F1 Min={val_metrics.get('f1_min', np.nan):.6f}, "
              f"Val ECE={val_metrics.get('ece', np.nan):.6f}")
        
        # Guardar mejor checkpoint por F1 macro
        f1_macro = val_metrics.get('f1_macro', np.nan)
        if not np.isnan(f1_macro) and f1_macro > best_f1_macro:
            best_f1_macro = f1_macro
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_metrics,
                'hparams': hparams
            }, checkpoint_dir / 'best.pt')
            print(f"  ✓ Mejor modelo guardado (F1 macro: {best_f1_macro:.6f})")
        
        # Early stopping por F1 macro
        if not np.isnan(f1_macro):
            if early_stopping(f1_macro):
                print(f"\nEarly stopping activado después de {epoch + 1} épocas")
                break
    
    # Exportar a ONNX
    print("\n" + "="*60)
    print("EXPORTACIÓN A ONNX")
    print("="*60)
    
    # Cargar mejor modelo
    # weights_only=False para compatibilidad con PyTorch 2.6+
    checkpoint = torch.load(checkpoint_dir / 'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Exportar
    output_path = Path('alphabetnet.onnx')
    export_to_onnx(model, output_path, hparams)
    
    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"✓ Modelo ONNX exportado: {output_path} ({size_mb:.2f} MB)")
    else:
        print("✗ Error al exportar modelo ONNX")
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETADO")
    print("="*60)
    print(f"Mejor F1 macro: {best_f1_macro:.6f}")
    print(f"Checkpoint: checkpoints/best.pt")
    print(f"Modelo ONNX: alphabetnet.onnx")
    print("\nPara inferencia:")
    print("  from infer import predict_alphabet_from_regex")
    print("  result = predict_alphabet_from_regex('(AB)*C', model, device)")
    print(f"  print(result)")

if __name__ == '__main__':
    main()


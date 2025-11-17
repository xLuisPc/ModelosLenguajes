"""
Modelo AlphabetNet: RNN para predecir símbolos válidos después de un prefijo.

Arquitectura:
- Entrada: prefix (índices de caracteres, padded) y opcionalmente automata_id
- Embedding + RNN (GRU o LSTM) → tomar h_t del último carácter no-PAD
- Linear → |Σ| (tamaño del alfabeto)
- Sigmoid en inferencia (logits en training)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class AlphabetNet(nn.Module):
    """
    Modelo RNN para predecir qué símbolos son válidos después de un prefijo.
    
    Args:
        vocab_size: Tamaño del vocabulario (|Σ| + PAD + <EPS> = 14)
        alphabet_size: Tamaño del alfabeto |Σ| (12 para A-L)
        emb_dim: Dimensión de embeddings (64-128)
        hidden_dim: Dimensión oculta de la RNN (128-256)
        rnn_type: Tipo de RNN ('GRU' o 'LSTM')
        num_layers: Número de capas RNN (1-2)
        dropout: Dropout (0.1-0.3)
        padding_idx: Índice del token PAD (0)
        use_automata_conditioning: Si True, incluye embedding de automata_id
        num_automata: Número de autómatas únicos (para conditioning)
        automata_emb_dim: Dimensión del embedding de automata (16)
    """
    
    def __init__(
        self,
        vocab_size=14,  # |Σ| + PAD + <EPS> = 12 + 1 + 1
        alphabet_size=12,  # |Σ| = A-L
        emb_dim=96,
        hidden_dim=192,
        rnn_type='GRU',
        num_layers=1,
        dropout=0.2,
        padding_idx=0,
        use_automata_conditioning=False,
        num_automata=None,
        automata_emb_dim=16
    ):
        super(AlphabetNet, self).__init__()
        
        self.vocab_size = vocab_size
        self.alphabet_size = alphabet_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.rnn_type = rnn_type.upper()
        self.num_layers = num_layers
        self.padding_idx = padding_idx
        self.use_automata_conditioning = use_automata_conditioning
        
        # Embedding para caracteres
        self.embedding = nn.Embedding(
            vocab_size,
            emb_dim,
            padding_idx=padding_idx
        )
        
        # Embedding opcional para automata_id
        if use_automata_conditioning:
            if num_automata is None:
                raise ValueError("num_automata debe ser especificado si use_automata_conditioning=True")
            self.automata_embedding = nn.Embedding(num_automata, automata_emb_dim)
            self.automata_emb_dim = automata_emb_dim
        else:
            self.automata_embedding = None
            self.automata_emb_dim = 0
        
        # RNN (unidireccional)
        if self.rnn_type == 'GRU':
            self.rnn = nn.GRU(
                emb_dim,
                hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
                bidirectional=False
            )
        elif self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                emb_dim,
                hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
                bidirectional=False
            )
        else:
            raise ValueError(f"rnn_type debe ser 'GRU' o 'LSTM', recibido: {rnn_type}")
        
        # Capa de dropout antes de la salida
        self.dropout = nn.Dropout(dropout)
        
        # Capa lineal final
        # Input: hidden_dim (+ automata_emb_dim si hay conditioning)
        # Output: alphabet_size (|Σ|)
        linear_input_dim = hidden_dim + self.automata_emb_dim
        self.output_layer = nn.Linear(linear_input_dim, alphabet_size)
        
    def forward(self, prefix_indices, lengths, automata_ids=None, return_logits=True):
        """
        Forward pass del modelo.
        
        Args:
            prefix_indices: Tensor de forma (batch_size, seq_len) con índices de caracteres
            lengths: Tensor 1D de longitudes reales (sin padding) para cada secuencia
            automata_ids: Tensor 1D opcional con IDs de autómatas (batch_size,)
            return_logits: Si True, retorna logits; si False, retorna sigmoid
        
        Returns:
            Tensor de forma (batch_size, alphabet_size) con logits o probabilidades
        """
        batch_size = prefix_indices.size(0)
        
        # Embedding de caracteres
        # (batch_size, seq_len, emb_dim)
        embedded = self.embedding(prefix_indices)
        
        # Pack padded sequence para que la RNN ignore PAD
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        
        # RNN forward
        if self.rnn_type == 'GRU':
            packed_output, hidden = self.rnn(packed)
        else:  # LSTM
            packed_output, (hidden, _) = self.rnn(packed)
        
        # hidden tiene forma (num_layers, batch_size, hidden_dim)
        # Tomamos la última capa
        if self.num_layers > 1:
            # (batch_size, hidden_dim)
            h_t = hidden[-1]
        else:
            # (batch_size, hidden_dim)
            h_t = hidden.squeeze(0)
        
        # Concatenar embedding de automata si está disponible
        if self.use_automata_conditioning and automata_ids is not None:
            # (batch_size, automata_emb_dim)
            automata_emb = self.automata_embedding(automata_ids)
            # (batch_size, hidden_dim + automata_emb_dim)
            h_t = torch.cat([h_t, automata_emb], dim=1)
        
        # Dropout
        h_t = self.dropout(h_t)
        
        # Capa lineal final
        # (batch_size, alphabet_size)
        logits = self.output_layer(h_t)
        
        if return_logits:
            return logits
        else:
            # Sigmoid para inferencia (probabilidades)
            return torch.sigmoid(logits)
    
    def predict(self, prefix_indices, lengths, automata_ids=None, threshold=0.5):
        """
        Método de conveniencia para inferencia.
        
        Args:
            prefix_indices: Tensor de forma (batch_size, seq_len) con índices de caracteres
            lengths: Tensor 1D de longitudes reales
            automata_ids: Tensor 1D opcional con IDs de autómatas
            threshold: Umbral para binarizar las predicciones (default: 0.5)
        
        Returns:
            Tensor binario de forma (batch_size, alphabet_size) con predicciones
        """
        self.eval()
        with torch.no_grad():
            # Obtener probabilidades (sigmoid)
            probs = self.forward(
                prefix_indices,
                lengths,
                automata_ids,
                return_logits=False
            )
            # Binarizar según threshold
            predictions = (probs >= threshold).long()
        return predictions


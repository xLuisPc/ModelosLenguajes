"""
Script para exportar modelo AlphabetNet a formato ONNX.

Características:
- Exporta modelo desde checkpoint a ONNX
- Maneja casos con/sin automata_id embedding
- Documenta entrada/salida del modelo
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional, Dict

import torch
import torch.nn as nn

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from model import AlphabetNet
from train import ALPHABET, MAX_PREFIX_LEN

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


class ModelWrapper(nn.Module):
    """
    Wrapper del modelo para exportación ONNX.
    
    Simplifica la interfaz del modelo para ONNX export, manejando
    automáticamente el caso con/sin automata_id.
    """
    
    def __init__(self, model: AlphabetNet):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.use_automata_conditioning = model.use_automata_conditioning
    
    def forward(self, prefix_indices: torch.Tensor, lengths: torch.Tensor,
                automata_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass simplificado para ONNX.
        
        Args:
            prefix_indices: Tensor de forma (batch_size, max_seq_len) con índices de caracteres
            lengths: Tensor 1D de forma (batch_size,) con longitudes reales
            automata_ids: Tensor 1D opcional de forma (batch_size,) con IDs de autómatas
        
        Returns:
            Tensor de forma (batch_size, alphabet_size) con logits
        """
        return self.model(prefix_indices, lengths, automata_ids=automata_ids, return_logits=True)


def export_to_onnx(model: AlphabetNet, output_path: Path, hparams: dict,
                   batch_size: int = 1, opset_version: int = 13):
    """
    Exporta modelo a formato ONNX.
    
    Args:
        model: Modelo AlphabetNet entrenado
        output_path: Path donde guardar el modelo ONNX
        hparams: Hiperparámetros del modelo
        batch_size: Tamaño de batch para ejemplo (puede ser dinámico)
        opset_version: Versión del opset ONNX (default: 13)
    """
    logger.info("="*60)
    logger.info("EXPORTACIÓN A ONNX")
    logger.info("="*60)
    
    # Crear wrapper del modelo
    wrapped_model = ModelWrapper(model)
    wrapped_model.eval()
    
    # Preparar inputs de ejemplo
    max_seq_len = MAX_PREFIX_LEN
    alphabet_size = hparams['model']['alphabet_size']
    use_automata = hparams['model']['use_automata_conditioning']
    
    # Input: prefix_indices
    # Shape: (batch_size, max_seq_len)
    # Tipo: int64 (índices de caracteres)
    prefix_indices = torch.randint(0, hparams['model']['vocab_size'], 
                                   (batch_size, max_seq_len), dtype=torch.long)
    
    # Input: lengths
    # Shape: (batch_size,)
    # Tipo: int64 (longitudes reales de prefijos)
    lengths = torch.randint(1, max_seq_len + 1, (batch_size,), dtype=torch.long)
    
    # Input opcional: automata_ids
    # Shape: (batch_size,)
    # Tipo: int64 (IDs de autómatas)
    automata_ids = None
    if use_automata:
        num_automata = hparams['model'].get('num_automata', 3000)
        automata_ids = torch.randint(0, num_automata, (batch_size,), dtype=torch.long)
    
    logger.info("Preparando inputs de ejemplo:")
    logger.info(f"  prefix_indices: {prefix_indices.shape} (dtype: {prefix_indices.dtype})")
    logger.info(f"  lengths: {lengths.shape} (dtype: {lengths.dtype})")
    if automata_ids is not None:
        logger.info(f"  automata_ids: {automata_ids.shape} (dtype: {automata_ids.dtype})")
    else:
        logger.info(f"  automata_ids: No se usa")
    
    # Preparar inputs para ONNX export
    if use_automata and automata_ids is not None:
        example_inputs = (prefix_indices, lengths, automata_ids)
        input_names = ['prefix_indices', 'lengths', 'automata_ids']
        dynamic_axes = {
            'prefix_indices': {0: 'batch_size'},
            'lengths': {0: 'batch_size'},
            'automata_ids': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    else:
        example_inputs = (prefix_indices, lengths)
        input_names = ['prefix_indices', 'lengths']
        dynamic_axes = {
            'prefix_indices': {0: 'batch_size'},
            'lengths': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    
    # Output
    output_names = ['logits']
    
    # Exportar a ONNX
    logger.info(f"Exportando a ONNX (opset_version={opset_version})...")
    logger.info(f"  Output: {output_path}")
    
    try:
        torch.onnx.export(
            wrapped_model,
            example_inputs,
            str(output_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
            export_params=True,
            verbose=False
        )
        logger.info("✓ Modelo exportado exitosamente a ONNX")
        
        # Verificar que el archivo se creó
        if output_path.exists():
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"  Tamaño del archivo: {file_size_mb:.2f} MB")
        
    except Exception as e:
        logger.error(f"Error al exportar a ONNX: {e}")
        raise


def save_model_with_thresholds(checkpoint_path: Path, output_path: Path,
                                thresholds_path: Optional[Path] = None):
    """
    Guarda un checkpoint mejorado con thresholds incluidos.
    
    Args:
        checkpoint_path: Path al checkpoint original
        output_path: Path donde guardar el checkpoint mejorado
        thresholds_path: Path opcional al archivo JSON con thresholds
    """
    logger.info(f"Cargando checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Cargar thresholds si están disponibles
    thresholds = None
    if thresholds_path and thresholds_path.exists():
        with open(thresholds_path, 'r') as f:
            thresholds_data = json.load(f)
            thresholds = thresholds_data.get('per_symbol', {})
            logger.info(f"✓ Thresholds cargados desde: {thresholds_path}")
    elif 'thresholds' in checkpoint:
        thresholds = checkpoint['thresholds']
        logger.info("✓ Thresholds encontrados en checkpoint original")
    
    # Crear nuevo checkpoint con thresholds
    enhanced_checkpoint = checkpoint.copy()
    if thresholds is not None:
        enhanced_checkpoint['thresholds'] = thresholds
        enhanced_checkpoint['alphabet'] = ALPHABET
    
    # Guardar checkpoint mejorado
    torch.save(enhanced_checkpoint, output_path)
    logger.info(f"✓ Checkpoint mejorado guardado en: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Exportar modelo AlphabetNet a ONNX')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path al checkpoint del modelo (best.pt o last.pt)')
    parser.add_argument('--hparams', type=str, default='hparams.json',
                       help='Path al archivo de hiperparámetros')
    parser.add_argument('--output', type=str, default='alphabetnet.onnx',
                       help='Path de salida para el modelo ONNX (default: alphabetnet.onnx)')
    parser.add_argument('--thresholds', type=str, default=None,
                       help='Path al archivo JSON con thresholds por símbolo')
    parser.add_argument('--enhanced_checkpoint', type=str, default=None,
                       help='Path opcional para guardar checkpoint mejorado con thresholds')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Tamaño de batch para ejemplo (default: 1)')
    parser.add_argument('--opset_version', type=int, default=13,
                       help='Versión del opset ONNX (default: 13)')
    
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
    
    # Guardar checkpoint mejorado con thresholds si se especifica
    if args.enhanced_checkpoint:
        thresholds_path = Path(args.thresholds) if args.thresholds else None
        output_checkpoint = Path(args.enhanced_checkpoint)
        save_model_with_thresholds(checkpoint_path, output_checkpoint, thresholds_path)
    
    # Exportar a ONNX
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    export_to_onnx(
        model,
        output_path,
        hparams,
        batch_size=args.batch_size,
        opset_version=args.opset_version
    )
    
    logger.info("="*60)
    logger.info("EXPORTACIÓN COMPLETADA")
    logger.info("="*60)
    logger.info(f"Modelo ONNX guardado en: {output_path}")
    if args.enhanced_checkpoint:
        logger.info(f"Checkpoint mejorado guardado en: {args.enhanced_checkpoint}")
    logger.info("="*60)


if __name__ == '__main__':
    main()


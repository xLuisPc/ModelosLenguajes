"""
Utilidades para reproducibilidad y configuración.

Incluye:
- Configuración de seeds para reproducibilidad
- Funciones auxiliares para inicialización determinística
"""

import os
import random
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)


def set_seeds(random_seed: int, numpy_seed: int, torch_seed: int, 
              cuda_seed: int, deterministic: bool = True):
    """
    Configura seeds para reproducibilidad completa.
    
    Args:
        random_seed: Seed para Python random
        numpy_seed: Seed para NumPy
        torch_seed: Seed para PyTorch
        cuda_seed: Seed para CUDA (si está disponible)
        deterministic: Si True, configura PyTorch para determinismo (puede reducir rendimiento)
    """
    # Python random
    random.seed(random_seed)
    
    # NumPy
    np.random.seed(numpy_seed)
    
    # PyTorch
    torch.manual_seed(torch_seed)
    
    # CUDA seeds
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cuda_seed)
        torch.cuda.manual_seed_all(cuda_seed)
    
    # Configurar PyTorch para determinismo
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Variable de entorno para reproducibilidad adicional
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    
    logger.info("✓ Seeds configurados para reproducibilidad")
    logger.info(f"  Python random: {random_seed}")
    logger.info(f"  NumPy: {numpy_seed}")
    logger.info(f"  PyTorch: {torch_seed}")
    if torch.cuda.is_available():
        logger.info(f"  CUDA: {cuda_seed}")
        if deterministic:
            logger.info(f"  CUDNN deterministic: True")


def load_seeds_from_hparams(hparams: dict, deterministic: bool = True):
    """
    Carga y configura seeds desde hparams.json.
    
    Args:
        hparams: Dict con hiperparámetros (debe incluir 'seeds')
        deterministic: Si True, configura PyTorch para determinismo
    """
    seeds = hparams.get('seeds', {})
    
    set_seeds(
        random_seed=seeds.get('random_seed', 42),
        numpy_seed=seeds.get('numpy_seed', 42),
        torch_seed=seeds.get('torch_seed', 42),
        cuda_seed=seeds.get('cuda_seed', 42),
        deterministic=deterministic
    )


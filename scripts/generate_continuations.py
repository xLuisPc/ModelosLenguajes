"""
Script para generar dataset de continuations (prefijos → próximos símbolos válidos).

Genera ejemplos para entrenar una RNN multi-etiqueta que, dado un prefijo de una cadena válida,
prediga qué símbolos podrían venir después.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import json

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Alfabeto global: A-L (12 símbolos)
ALPHABET = list('ABCDEFGHIJKL')
ALPHABET_SIZE = len(ALPHABET)
MAX_PREFIX_LEN = 64
MIN_SUPPORT = 2  # Mínimo soporte para mantener un prefijo


def generate_prefixes_and_continuations(df_positive):
    """
    Genera prefijos y sus continuaciones desde cadenas positivas.
    
    Args:
        df_positive: DataFrame con solo cadenas positivas (label=1)
        
    Returns:
        dict: {dfa_id: {prefix: {symbol: count}}}
    """
    logger.info("Generando prefijos y continuaciones...")
    
    # Estructura: dfa_id -> prefix -> {symbol: count}
    continuations = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    
    for _, row in df_positive.iterrows():
        dfa_id = row['dfa_id']
        string = row['string']
        
        # Convertir <EPS> a cadena vacía
        if string == '<EPS>':
            s = ''
        else:
            s = string
        
        # Si la cadena está vacía, no hay prefijos (solo continuaciones desde <EPS>)
        if len(s) == 0:
            continue
        
        # Generar prefijos: <EPS>, c1, c1c2, ..., c1...cT-1
        # Prefijo <EPS> (vacío)
        if len(s) > 0:
            first_symbol = s[0]
            continuations[dfa_id]['<EPS>'][first_symbol] += 1
        
        # Prefijos de longitud 1 a T-1
        for i in range(1, len(s)):
            prefix = s[:i]
            next_symbol = s[i]
            
            # Truncar prefijo si es muy largo
            if len(prefix) > MAX_PREFIX_LEN:
                prefix = prefix[-MAX_PREFIX_LEN:]
            
            continuations[dfa_id][prefix][next_symbol] += 1
    
    logger.info(f"✓ Generados prefijos para {len(continuations)} autómatas")
    return continuations


def build_wide_format(continuations):
    """
    Construye el dataset en formato "ancho" (multi-hot).
    
    Returns:
        pd.DataFrame con columnas: dfa_id, prefix, y (lista de 12 ints), support_pos (lista de 12 ints)
    """
    logger.info("Construyendo formato ancho (multi-hot)...")
    
    rows = []
    
    for dfa_id, prefix_dict in continuations.items():
        for prefix, symbol_counts in prefix_dict.items():
            # Calcular soporte total
            total_support = sum(symbol_counts.values())
            
            # Filtrar prefijos con soporte mínimo
            if total_support < MIN_SUPPORT:
                continue
            
            # Crear vector multi-hot y support
            y = [0] * ALPHABET_SIZE
            support_pos = [0] * ALPHABET_SIZE
            
            for symbol, count in symbol_counts.items():
                if symbol in ALPHABET:
                    idx = ALPHABET.index(symbol)
                    y[idx] = 1
                    support_pos[idx] = count
                else:
                    logger.warning(f"Símbolo fuera del alfabeto: {symbol} en dfa_id={dfa_id}, prefix={prefix}")
            
            rows.append({
                'dfa_id': dfa_id,
                'prefix': prefix,
                'y': y,
                'support_pos': support_pos
            })
    
    df_wide = pd.DataFrame(rows)
    logger.info(f"✓ Formato ancho: {len(df_wide):,} filas generadas")
    return df_wide


def build_long_format(continuations, neg_ratio=1.0, random_seed=42):
    """
    Construye el dataset en formato "largo" (binario) con negative sampling.
    
    Args:
        continuations: dict con continuaciones
        neg_ratio: ratio de negativos por positivo (1.0 = 1:1)
        random_seed: semilla para reproducibilidad
        
    Returns:
        pd.DataFrame con columnas: dfa_id, prefix, symbol, label
    """
    logger.info(f"Construyendo formato largo (binario) con ratio neg:pos = {neg_ratio}...")
    
    np.random.seed(random_seed)
    rows = []
    
    for dfa_id, prefix_dict in continuations.items():
        for prefix, symbol_counts in prefix_dict.items():
            # Calcular soporte total
            total_support = sum(symbol_counts.values())
            
            # Filtrar prefijos con soporte mínimo
            if total_support < MIN_SUPPORT:
                continue
            
            # Símbolos positivos
            positive_symbols = set(symbol_counts.keys())
            # Filtrar símbolos válidos
            positive_symbols = {s for s in positive_symbols if s in ALPHABET}
            
            if not positive_symbols:
                continue
            
            # Agregar ejemplos positivos y contar total
            num_positives = 0
            for symbol in positive_symbols:
                count = symbol_counts[symbol]
                # Agregar múltiples ejemplos según el count
                for _ in range(count):
                    rows.append({
                        'dfa_id': dfa_id,
                        'prefix': prefix,
                        'symbol': symbol,
                        'label': 1
                    })
                    num_positives += 1
            
            # Negative sampling: generar neg_ratio negativos por cada positivo
            negative_symbols = set(ALPHABET) - positive_symbols
            if negative_symbols and num_positives > 0:
                num_negatives = int(num_positives * neg_ratio)
                
                # Seleccionar símbolos negativos aleatoriamente
                sampled_negatives = np.random.choice(
                    list(negative_symbols),
                    size=num_negatives,
                    replace=True
                )
                
                for symbol in sampled_negatives:
                    rows.append({
                        'dfa_id': dfa_id,
                        'prefix': prefix,
                        'symbol': symbol,
                        'label': 0
                    })
    
    df_long = pd.DataFrame(rows)
    logger.info(f"✓ Formato largo: {len(df_long):,} filas generadas")
    
    # Verificar ratio
    pos_count = (df_long['label'] == 1).sum()
    neg_count = (df_long['label'] == 0).sum()
    actual_ratio = neg_count / pos_count if pos_count > 0 else 0
    logger.info(f"  - Positivos: {pos_count:,}")
    logger.info(f"  - Negativos: {neg_count:,}")
    logger.info(f"  - Ratio real: {actual_ratio:.2f}")
    
    return df_long


def generate_statistics(df_positive, continuations, df_wide=None, df_long=None):
    """
    Genera estadísticas del dataset de continuations.
    
    Returns:
        dict con estadísticas
    """
    logger.info("Generando estadísticas...")
    
    stats = {}
    
    # Estadísticas básicas
    stats['total_positive_strings'] = len(df_positive)
    stats['unique_dfas'] = df_positive['dfa_id'].nunique()
    
    # Prefijos por autómata
    prefixes_per_dfa = {}
    eps_prefixes = 0
    total_prefixes = 0
    
    for dfa_id, prefix_dict in continuations.items():
        # Filtrar por soporte mínimo
        valid_prefixes = {p: s for p, s in prefix_dict.items() 
                         if sum(s.values()) >= MIN_SUPPORT}
        num_prefixes = len(valid_prefixes)
        prefixes_per_dfa[dfa_id] = num_prefixes
        total_prefixes += num_prefixes
        
        if '<EPS>' in valid_prefixes:
            eps_prefixes += 1
    
    stats['prefixes_per_dfa'] = prefixes_per_dfa
    stats['total_prefixes'] = total_prefixes
    stats['avg_prefixes_per_dfa'] = total_prefixes / len(prefixes_per_dfa) if prefixes_per_dfa else 0
    stats['dfas_with_eps'] = eps_prefixes
    stats['pct_eps'] = (eps_prefixes / len(prefixes_per_dfa) * 100) if prefixes_per_dfa else 0
    
    # Distribución de #positivos por prefijo
    positives_per_prefix = []
    for dfa_id, prefix_dict in continuations.items():
        for prefix, symbol_counts in prefix_dict.items():
            total_support = sum(symbol_counts.values())
            if total_support >= MIN_SUPPORT:
                positives_per_prefix.append(total_support)
    
    if positives_per_prefix:
        stats['positives_per_prefix'] = {
            'min': min(positives_per_prefix),
            'max': max(positives_per_prefix),
            'mean': np.mean(positives_per_prefix),
            'median': np.median(positives_per_prefix),
            'p95': np.percentile(positives_per_prefix, 95),
            'p99': np.percentile(positives_per_prefix, 99)
        }
    
    # Top prefijos por frecuencia
    prefix_freq = defaultdict(int)
    for dfa_id, prefix_dict in continuations.items():
        for prefix, symbol_counts in prefix_dict.items():
            total_support = sum(symbol_counts.values())
            if total_support >= MIN_SUPPORT:
                prefix_freq[prefix] += total_support
    
    top_prefixes = sorted(prefix_freq.items(), key=lambda x: x[1], reverse=True)[:20]
    stats['top_prefixes'] = top_prefixes
    
    # Distribución de longitudes de prefijos
    prefix_lengths = [len(p) for p in prefix_freq.keys()]
    if prefix_lengths:
        stats['prefix_length_dist'] = {
            'min': min(prefix_lengths),
            'max': max(prefix_lengths),
            'mean': np.mean(prefix_lengths),
            'median': np.median(prefix_lengths),
            'p95': np.percentile(prefix_lengths, 95),
            'p99': np.percentile(prefix_lengths, 99)
        }
    
    # Estadísticas del formato ancho
    if df_wide is not None:
        stats['wide_format'] = {
            'total_rows': len(df_wide),
            'unique_dfas': df_wide['dfa_id'].nunique(),
            'unique_prefixes': df_wide['prefix'].nunique()
        }
    
    # Estadísticas del formato largo
    if df_long is not None:
        pos_count = (df_long['label'] == 1).sum()
        neg_count = (df_long['label'] == 0).sum()
        stats['long_format'] = {
            'total_rows': len(df_long),
            'positive_rows': pos_count,
            'negative_rows': neg_count,
            'ratio_neg_pos': neg_count / pos_count if pos_count > 0 else 0,
            'unique_dfas': df_long['dfa_id'].nunique(),
            'unique_prefixes': df_long['prefix'].nunique()
        }
    
    logger.info("✓ Estadísticas generadas")
    return stats


def generate_report(stats, output_file):
    """
    Genera reporte Markdown con estadísticas.
    """
    logger.info("Generando reporte...")
    
    report = []
    report.append("# Dataset de Continuations - Reporte")
    report.append("")
    report.append(f"Generado el: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    report.append("## 1. Resumen General")
    report.append("")
    report.append(f"- **Total de cadenas positivas procesadas:** {stats['total_positive_strings']:,}")
    report.append(f"- **Total de autómatas únicos:** {stats['unique_dfas']:,}")
    report.append(f"- **Total de prefijos generados:** {stats['total_prefixes']:,}")
    report.append(f"- **Promedio de prefijos por autómata:** {stats['avg_prefixes_per_dfa']:.2f}")
    report.append("")
    
    report.append("## 2. Prefijos por Autómata")
    report.append("")
    report.append(f"- **Autómatas con prefijo <EPS>:** {stats['dfas_with_eps']:,} ({stats['pct_eps']:.2f}%)")
    report.append("")
    
    # Distribución de prefijos por autómata
    prefixes_per_dfa_list = list(stats['prefixes_per_dfa'].values())
    if prefixes_per_dfa_list:
        report.append("| Estadística | Valor |")
        report.append("|------------|-------|")
        report.append(f"| Mínimo | {min(prefixes_per_dfa_list)} |")
        report.append(f"| Máximo | {max(prefixes_per_dfa_list)} |")
        report.append(f"| Media | {np.mean(prefixes_per_dfa_list):.2f} |")
        report.append(f"| Mediana | {np.median(prefixes_per_dfa_list):.2f} |")
        report.append("")
    
    report.append("## 3. Distribución de Positivos por Prefijo")
    report.append("")
    if 'positives_per_prefix' in stats:
        pp = stats['positives_per_prefix']
        report.append("| Estadística | Valor |")
        report.append("|------------|-------|")
        report.append(f"| Mínimo | {pp['min']} |")
        report.append(f"| Máximo | {pp['max']} |")
        report.append(f"| Media | {pp['mean']:.2f} |")
        report.append(f"| Mediana | {pp['median']:.2f} |")
        report.append(f"| Percentil 95 | {pp['p95']:.2f} |")
        report.append(f"| Percentil 99 | {pp['p99']:.2f} |")
        report.append("")
    
    report.append("## 4. Top 20 Prefijos por Frecuencia")
    report.append("")
    report.append("| Prefijo | Frecuencia Total |")
    report.append("|---------|------------------|")
    for prefix, freq in stats['top_prefixes']:
        display_prefix = prefix if prefix else '<EPS>'
        report.append(f"| `{display_prefix}` | {freq:,} |")
    report.append("")
    
    report.append("## 5. Distribución de Longitudes de Prefijos")
    report.append("")
    if 'prefix_length_dist' in stats:
        pld = stats['prefix_length_dist']
        report.append("| Estadística | Valor |")
        report.append("|------------|-------|")
        report.append(f"| Mínimo | {pld['min']} |")
        report.append(f"| Máximo | {pld['max']} |")
        report.append(f"| Media | {pld['mean']:.2f} |")
        report.append(f"| Mediana | {pld['median']:.2f} |")
        report.append(f"| Percentil 95 | {pld['p95']:.2f} |")
        report.append(f"| Percentil 99 | {pld['p99']:.2f} |")
        report.append("")
        
        if pld['p95'] <= MAX_PREFIX_LEN:
            report.append(f"✅ **El percentil 95 es {pld['p95']:.2f}, que está por debajo de max_len={MAX_PREFIX_LEN}.**")
        else:
            report.append(f"⚠️ **El percentil 95 es {pld['p95']:.2f}, que está por encima de max_len={MAX_PREFIX_LEN}.**")
        report.append("")
    
    # Estadísticas de formato ancho
    if 'wide_format' in stats:
        report.append("## 6. Formato Ancho (Multi-hot)")
        report.append("")
        wf = stats['wide_format']
        report.append(f"- **Total de filas:** {wf['total_rows']:,}")
        report.append(f"- **Autómatas únicos:** {wf['unique_dfas']:,}")
        report.append(f"- **Prefijos únicos:** {wf['unique_prefixes']:,}")
        report.append("")
    
    # Estadísticas de formato largo
    if 'long_format' in stats:
        report.append("## 7. Formato Largo (Binario)")
        report.append("")
        lf = stats['long_format']
        report.append(f"- **Total de filas:** {lf['total_rows']:,}")
        report.append(f"- **Filas positivas:** {lf['positive_rows']:,}")
        report.append(f"- **Filas negativas:** {lf['negative_rows']:,}")
        report.append(f"- **Ratio neg:pos:** {lf['ratio_neg_pos']:.2f}")
        report.append(f"- **Autómatas únicos:** {lf['unique_dfas']:,}")
        report.append(f"- **Prefijos únicos:** {lf['unique_prefixes']:,}")
        report.append("")
        
        # Verificar ratio
        expected_ratio = 1.0  # Asumimos 1:1 por defecto
        ratio_diff = abs(lf['ratio_neg_pos'] - expected_ratio) / expected_ratio * 100
        if ratio_diff <= 10:
            report.append(f"✅ **El ratio está dentro del ±10% del esperado ({expected_ratio:.2f}).**")
        else:
            report.append(f"⚠️ **El ratio difiere más del 10% del esperado ({expected_ratio:.2f}).**")
        report.append("")
    
    report.append("## 8. Criterios de Aceptación")
    report.append("")
    
    # Verificar criterios
    min_prefixes_per_dfa = 20
    prefixes_per_dfa_list = list(stats['prefixes_per_dfa'].values())
    dfas_with_min_prefixes = sum(1 for x in prefixes_per_dfa_list if x >= min_prefixes_per_dfa)
    pct_with_min = (dfas_with_min_prefixes / len(prefixes_per_dfa_list) * 100) if prefixes_per_dfa_list else 0
    
    report.append(f"1. **Cada autómata tiene ≥ {min_prefixes_per_dfa} prefijos:**")
    report.append(f"   - Autómatas con ≥ {min_prefixes_per_dfa} prefijos: {dfas_with_min_prefixes:,} ({pct_with_min:.2f}%)")
    if pct_with_min == 100:
        report.append(f"   ✅ **Cumplido**")
    else:
        report.append(f"   ⚠️ **No cumplido completamente**")
    report.append("")
    
    report.append("2. **No hay símbolos fuera del vocabulario:**")
    report.append("   ✅ **Verificado durante la construcción**")
    report.append("")
    
    if 'long_format' in stats:
        report.append("3. **Ratio pos:neg se respeta (±10%):**")
        lf = stats['long_format']
        expected_ratio = 1.0
        ratio_diff = abs(lf['ratio_neg_pos'] - expected_ratio) / expected_ratio * 100
        if ratio_diff <= 10:
            report.append(f"   ✅ **Cumplido (ratio: {lf['ratio_neg_pos']:.2f})**")
        else:
            report.append(f"   ⚠️ **No cumplido (ratio: {lf['ratio_neg_pos']:.2f}, diferencia: {ratio_diff:.2f}%)**")
        report.append("")
    
    if 'prefix_length_dist' in stats:
        report.append("4. **p95 de longitud de prefijo ≤ max_len:**")
        pld = stats['prefix_length_dist']
        if pld['p95'] <= MAX_PREFIX_LEN:
            report.append(f"   ✅ **Cumplido (p95: {pld['p95']:.2f} ≤ {MAX_PREFIX_LEN})**")
        else:
            report.append(f"   ⚠️ **No cumplido (p95: {pld['p95']:.2f} > {MAX_PREFIX_LEN})**")
        report.append("")
    
    # Escribir reporte
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    logger.info(f"✓ Reporte guardado en {output_file}")


def main():
    """Función principal."""
    logger.info("="*60)
    logger.info("GENERACIÓN DE DATASET DE CONTINUATIONS")
    logger.info("="*60)
    
    project_root = Path(__file__).parent.parent
    
    # Leer dataset procesado
    input_file = project_root / 'data' / 'dataset3000_procesado.csv'
    logger.info(f"Leyendo dataset: {input_file}")
    df = pd.read_csv(input_file)
    logger.info(f"Total de filas: {len(df):,}")
    
    # Filtrar solo cadenas positivas
    df_positive = df[df['label'] == 1].copy()
    logger.info(f"Cadenas positivas: {len(df_positive):,}")
    logger.info(f"Autómatas únicos: {df_positive['dfa_id'].nunique():,}")
    
    # Generar prefijos y continuaciones
    continuations = generate_prefixes_and_continuations(df_positive)
    
    # Crear directorio de salida
    output_dir = project_root / 'data' / 'alphabet'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generar formato ancho
    df_wide = build_wide_format(continuations)
    wide_file = output_dir / 'continuations.parquet'
    df_wide.to_parquet(wide_file, index=False)
    logger.info(f"✓ Formato ancho guardado en {wide_file}")
    
    # Generar formato largo
    df_long = build_long_format(continuations, neg_ratio=1.0)
    long_file = output_dir / 'continuations_long.parquet'
    df_long.to_parquet(long_file, index=False)
    logger.info(f"✓ Formato largo guardado en {long_file}")
    
    # Generar estadísticas y reporte
    stats = generate_statistics(df_positive, continuations, df_wide, df_long)
    report_file = project_root / 'reports' / 'alphabetnet_A1_continuations.md'
    generate_report(stats, report_file)
    
    logger.info("")
    logger.info("="*60)
    logger.info("PROCESO COMPLETADO")
    logger.info("="*60)
    logger.info(f"Formato ancho: {wide_file}")
    logger.info(f"Formato largo: {long_file}")
    logger.info(f"Reporte: {report_file}")
    logger.info("="*60)


if __name__ == '__main__':
    main()


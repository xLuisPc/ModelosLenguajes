import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re

# Configurar estilo de gráficas
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

def analyze_length_distribution(df):
    """
    Analiza la distribución de la columna len.
    Calcula percentiles y genera gráficas.
    """
    print("Analizando distribución de len...")
    
    # Estadísticas descriptivas
    stats = {
        'p50': df['len'].quantile(0.50),
        'p95': df['len'].quantile(0.95),
        'p99': df['len'].quantile(0.99),
        'max': df['len'].max(),
        'mean': df['len'].mean(),
        'median': df['len'].median(),
        'std': df['len'].std()
    }
    
    # Crear gráfica de distribución
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histograma
    axes[0].hist(df['len'], bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(stats['p50'], color='r', linestyle='--', label=f"p50: {stats['p50']:.2f}")
    axes[0].axvline(stats['p95'], color='orange', linestyle='--', label=f"p95: {stats['p95']:.2f}")
    axes[0].axvline(stats['p99'], color='green', linestyle='--', label=f"p99: {stats['p99']:.2f}")
    axes[0].axvline(stats['max'], color='purple', linestyle='--', label=f"max: {stats['max']}")
    axes[0].axvline(64, color='red', linestyle='-', linewidth=2, label="max_len=64")
    axes[0].set_xlabel('Longitud (len)')
    axes[0].set_ylabel('Frecuencia')
    axes[0].set_title('Distribución de Longitudes de Strings')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Boxplot
    axes[1].boxplot(df['len'], vert=True)
    axes[1].axhline(64, color='red', linestyle='-', linewidth=2, label="max_len=64")
    axes[1].set_ylabel('Longitud (len)')
    axes[1].set_title('Boxplot de Longitudes')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('reports/figures/len_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return stats

def analyze_label_balance(df):
    """
    Analiza el balance de labels (0/1) global y por autómata.
    """
    print("Analizando balance de labels...")
    
    # Balance global
    global_balance = df['label'].value_counts().sort_index()
    global_balance_pct = df['label'].value_counts(normalize=True).sort_index() * 100
    
    # Balance por autómata
    balance_by_dfa = df.groupby('dfa_id')['label'].agg(['count', 'sum', 'mean']).reset_index()
    balance_by_dfa.columns = ['dfa_id', 'total_strings', 'accepted_count', 'acceptance_rate']
    balance_by_dfa['rejected_count'] = balance_by_dfa['total_strings'] - balance_by_dfa['accepted_count']
    balance_by_dfa['acceptance_rate'] = balance_by_dfa['acceptance_rate'] * 100
    
    # Autómatas desbalanceados (más del 90% de un tipo)
    imbalanced = balance_by_dfa[
        (balance_by_dfa['acceptance_rate'] >= 90) | 
        (balance_by_dfa['acceptance_rate'] <= 10)
    ]
    
    # Crear gráficas
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Balance global - barras
    axes[0, 0].bar(global_balance.index, global_balance.values, color=['red', 'green'], alpha=0.7)
    axes[0, 0].set_xlabel('Label')
    axes[0, 0].set_ylabel('Cantidad')
    axes[0, 0].set_title('Balance Global de Labels')
    axes[0, 0].set_xticks([0, 1])
    axes[0, 0].grid(True, alpha=0.3)
    for i, (idx, val) in enumerate(global_balance.items()):
        axes[0, 0].text(idx, val, f'{val:,}\n({global_balance_pct[idx]:.2f}%)', 
                       ha='center', va='bottom')
    
    # Distribución de tasa de aceptación por autómata
    axes[0, 1].hist(balance_by_dfa['acceptance_rate'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Tasa de Aceptación (%)')
    axes[0, 1].set_ylabel('Número de Autómatas')
    axes[0, 1].set_title('Distribución de Tasa de Aceptación por Autómata')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Boxplot de tasa de aceptación
    axes[1, 0].boxplot(balance_by_dfa['acceptance_rate'], vert=True)
    axes[1, 0].set_ylabel('Tasa de Aceptación (%)')
    axes[1, 0].set_title('Boxplot de Tasa de Aceptación')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Distribución de cantidad de strings por autómata
    axes[1, 1].hist(balance_by_dfa['total_strings'], bins=50, edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('Cantidad de Strings')
    axes[1, 1].set_ylabel('Número de Autómatas')
    axes[1, 1].set_title('Distribución de Strings por Autómata')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('reports/figures/label_balance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'global_balance': global_balance.to_dict(),
        'global_balance_pct': global_balance_pct.to_dict(),
        'balance_by_dfa': balance_by_dfa,
        'imbalanced_count': len(imbalanced),
        'imbalanced_dfas': imbalanced
    }

def analyze_symbol_frequency(df):
    """
    Analiza la frecuencia de símbolos global y por autómata.
    """
    print("Analizando frecuencia de símbolos...")
    
    # Frecuencia global de símbolos
    all_strings = df[df['string'] != '<EPS>']['string'].str.cat()
    symbol_counts_global = pd.Series(list(all_strings)).value_counts().sort_index()
    symbol_freq_global = (symbol_counts_global / symbol_counts_global.sum() * 100).round(2)
    
    # Frecuencia por autómata (para cada dfa_id)
    symbol_freq_by_dfa = []
    for dfa_id in df['dfa_id'].unique():
        dfa_strings = df[df['dfa_id'] == dfa_id]
        dfa_strings_only = dfa_strings[dfa_strings['string'] != '<EPS>']['string'].str.cat()
        if len(dfa_strings_only) > 0:
            symbol_counts = pd.Series(list(dfa_strings_only)).value_counts()
            symbol_freq = (symbol_counts / symbol_counts.sum() * 100).round(2)
            for symbol, freq in symbol_freq.items():
                symbol_freq_by_dfa.append({
                    'dfa_id': dfa_id,
                    'symbol': symbol,
                    'frequency': freq
                })
    
    symbol_freq_by_dfa_df = pd.DataFrame(symbol_freq_by_dfa)
    
    # Crear gráficas
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Frecuencia global
    axes[0].bar(symbol_freq_global.index, symbol_freq_global.values, alpha=0.7)
    axes[0].set_xlabel('Símbolo')
    axes[0].set_ylabel('Frecuencia (%)')
    axes[0].set_title('Frecuencia Global de Símbolos')
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].set_xticks(range(len(symbol_freq_global.index)))
    axes[0].set_xticklabels(symbol_freq_global.index)
    
    # Frecuencia promedio por autómata
    avg_freq_by_symbol = symbol_freq_by_dfa_df.groupby('symbol')['frequency'].mean().sort_index()
    axes[1].bar(avg_freq_by_symbol.index, avg_freq_by_symbol.values, alpha=0.7, color='orange')
    axes[1].set_xlabel('Símbolo')
    axes[1].set_ylabel('Frecuencia Promedio (%)')
    axes[1].set_title('Frecuencia Promedio de Símbolos por Autómata')
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_xticks(range(len(avg_freq_by_symbol.index)))
    axes[1].set_xticklabels(avg_freq_by_symbol.index)
    
    plt.tight_layout()
    plt.savefig('reports/figures/symbol_frequency.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'global_frequency': symbol_freq_global.to_dict(),
        'global_counts': symbol_counts_global.to_dict(),
        'by_dfa': symbol_freq_by_dfa_df,
        'avg_by_symbol': avg_freq_by_symbol.to_dict()
    }

def find_unique_class_automatas(df):
    """
    Encuentra autómatas con clase única (solo 1s o solo 0s).
    """
    print("Buscando autómatas con clase única...")
    
    # Agrupar por dfa_id y verificar si todos los labels son iguales
    label_stats = df.groupby('dfa_id')['label'].agg(['min', 'max', 'count', 'sum']).reset_index()
    label_stats.columns = ['dfa_id', 'min_label', 'max_label', 'total_strings', 'accepted_count']
    
    # Autómatas que solo aceptan (todos los labels son 1)
    only_accept = label_stats[label_stats['min_label'] == 1]
    
    # Autómatas que solo rechazan (todos los labels son 0)
    only_reject = label_stats[label_stats['max_label'] == 0]
    
    # Crear gráfica
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Distribución de tipos de autómatas
    automata_types = {
        'Solo acepta': len(only_accept),
        'Solo rechaza': len(only_reject),
        'Mixto': len(label_stats) - len(only_accept) - len(only_reject)
    }
    
    axes[0].bar(automata_types.keys(), automata_types.values(), color=['green', 'red', 'blue'], alpha=0.7)
    axes[0].set_ylabel('Cantidad de Autómatas')
    axes[0].set_title('Distribución de Tipos de Autómatas')
    axes[0].grid(True, alpha=0.3, axis='y')
    for i, (key, val) in enumerate(automata_types.items()):
        axes[0].text(i, val, f'{val}', ha='center', va='bottom')
    
    # Distribución de cantidad de strings en autómatas con clase única
    if len(only_accept) > 0 or len(only_reject) > 0:
        unique_class_counts = pd.concat([
            only_accept['total_strings'],
            only_reject['total_strings']
        ])
        axes[1].hist(unique_class_counts, bins=30, edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Cantidad de Strings')
        axes[1].set_ylabel('Número de Autómatas')
        axes[1].set_title('Distribución de Strings en Autómatas con Clase Única')
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'No hay autómatas\ncon clase única', 
                    ha='center', va='center', fontsize=14)
        axes[1].set_title('Distribución de Strings en Autómatas con Clase Única')
    
    plt.tight_layout()
    plt.savefig('reports/figures/unique_class_automatas.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'only_accept': only_accept,
        'only_reject': only_reject,
        'only_accept_count': len(only_accept),
        'only_reject_count': len(only_reject),
        'total_unique_class': len(only_accept) + len(only_reject)
    }

def generate_report(stats_len, stats_labels, stats_symbols, stats_unique_class, output_file, df):
    """
    Genera el reporte Markdown con todos los análisis.
    """
    print(f"Generando reporte en {output_file}...")
    
    # Obtener nombre del archivo de entrada desde la ruta del output
    input_filename = Path(output_file).stem.replace('_eda', '')
    
    report = []
    report.append("# EDA - Análisis Exploratorio de Datos")
    report.append("")
    report.append("## Resumen Ejecutivo")
    report.append("")
    report.append(f"Este reporte presenta un análisis exploratorio del dataset `{input_filename}.csv`.")
    report.append("")
    
    # 1. Distribución de len
    report.append("## 1. Distribución de Longitudes (len)")
    report.append("")
    report.append("### Estadísticas Descriptivas")
    report.append("")
    report.append("| Estadística | Valor |")
    report.append("|------------|-------|")
    report.append(f"| Media | {stats_len['mean']:.2f} |")
    report.append(f"| Mediana (p50) | {stats_len['p50']:.2f} |")
    report.append(f"| Percentil 95 (p95) | {stats_len['p95']:.2f} |")
    report.append(f"| Percentil 99 (p99) | {stats_len['p99']:.2f} |")
    report.append(f"| Máximo | {stats_len['max']} |")
    report.append(f"| Desviación Estándar | {stats_len['std']:.2f} |")
    report.append("")
    report.append("### Justificación de max_len=64")
    report.append("")
    
    if stats_len['p99'] <= 64:
        report.append(f"✅ **El percentil 99 es {stats_len['p99']:.2f}, que está por debajo de 64.**")
        report.append("")
        report.append("Esto significa que el 99% de las strings tienen una longitud menor o igual a 64.")
        report.append("Por lo tanto, `max_len=64` es una elección adecuada que captura la gran mayoría")
        report.append("de los casos sin perder información significativa.")
    else:
        report.append(f"⚠️ **El percentil 99 es {stats_len['p99']:.2f}, que está por encima de 64.**")
        report.append("")
        report.append(f"Esto significa que aproximadamente el 1% de las strings exceden la longitud de 64.")
        report.append(f"El máximo encontrado es {stats_len['max']}.")
        report.append("")
        report.append("**Recomendación:** Considerar aumentar `max_len` o implementar estrategias de truncamiento")
        report.append("para strings muy largas si se pierde información crítica.")
    
    report.append("")
    report.append("![Distribución de Longitudes](figures/len_distribution.png)")
    report.append("")
    
    # 2. Balance de labels
    report.append("## 2. Balance de Labels (0/1)")
    report.append("")
    report.append("### Balance Global")
    report.append("")
    report.append("| Label | Cantidad | Porcentaje |")
    report.append("|-------|----------|------------|")
    report.append(f"| 0 (Rechazado) | {stats_labels['global_balance'].get(0, 0):,} | {stats_labels['global_balance_pct'].get(0, 0):.2f}% |")
    report.append(f"| 1 (Aceptado) | {stats_labels['global_balance'].get(1, 0):,} | {stats_labels['global_balance_pct'].get(1, 0):.2f}% |")
    report.append("")
    
    imbalance_pct = abs(stats_labels['global_balance_pct'].get(1, 0) - stats_labels['global_balance_pct'].get(0, 0))
    if imbalance_pct > 20:
        report.append(f"⚠️ **El dataset está desbalanceado globalmente ({imbalance_pct:.2f}% de diferencia).**")
    else:
        report.append(f"✅ **El dataset está relativamente balanceado globalmente.**")
    report.append("")
    
    report.append("### Balance por Autómata")
    report.append("")
    report.append(f"**Autómatas desbalanceados (≥90% de un tipo):** {stats_labels['imbalanced_count']}")
    report.append("")
    report.append("Un autómata se considera desbalanceado si tiene más del 90% de strings aceptadas o rechazadas.")
    report.append("")
    
    if stats_labels['imbalanced_count'] > 0:
        report.append("**Muestra de autómatas desbalanceados (primeros 10):**")
        report.append("")
        sample_imbalanced = stats_labels['imbalanced_dfas'].head(10)
        report.append("| dfa_id | Total Strings | Aceptados | Rechazados | Tasa Aceptación (%) |")
        report.append("|--------|---------------|-----------|------------|---------------------|")
        for _, row in sample_imbalanced.iterrows():
            report.append(f"| {row['dfa_id']} | {row['total_strings']} | {row['accepted_count']} | {row['rejected_count']} | {row['acceptance_rate']:.2f} |")
        report.append("")
    
    report.append("![Balance de Labels](figures/label_balance.png)")
    report.append("")
    
    # 3. Frecuencia de símbolos
    report.append("## 3. Frecuencia de Símbolos")
    report.append("")
    report.append("### Frecuencia Global")
    report.append("")
    report.append("| Símbolo | Frecuencia (%) | Cantidad |")
    report.append("|---------|----------------|----------|")
    for symbol in sorted(stats_symbols['global_frequency'].keys()):
        freq = stats_symbols['global_frequency'][symbol]
        count = stats_symbols['global_counts'][symbol]
        report.append(f"| {symbol} | {freq:.2f}% | {count:,} |")
    report.append("")
    
    report.append("### Frecuencia Promedio por Autómata")
    report.append("")
    report.append("| Símbolo | Frecuencia Promedio (%) |")
    report.append("|---------|------------------------|")
    for symbol in sorted(stats_symbols['avg_by_symbol'].keys()):
        freq = stats_symbols['avg_by_symbol'][symbol]
        report.append(f"| {symbol} | {freq:.2f}% |")
    report.append("")
    
    report.append("**Observaciones:**")
    report.append("")
    most_common = max(stats_symbols['global_frequency'].items(), key=lambda x: x[1])
    least_common = min(stats_symbols['global_frequency'].items(), key=lambda x: x[1])
    report.append(f"- Símbolo más frecuente: **{most_common[0]}** ({most_common[1]:.2f}%)")
    report.append(f"- Símbolo menos frecuente: **{least_common[0]}** ({least_common[1]:.2f}%)")
    report.append("")
    
    report.append("![Frecuencia de Símbolos](figures/symbol_frequency.png)")
    report.append("")
    
    # 4. Autómatas con clase única
    report.append("## 4. Autómatas con Clase Única")
    report.append("")
    report.append("### Resumen")
    report.append("")
    report.append(f"- **Autómatas que solo aceptan (todos los labels son 1):** {stats_unique_class['only_accept_count']}")
    report.append(f"- **Autómatas que solo rechazan (todos los labels son 0):** {stats_unique_class['only_reject_count']}")
    report.append(f"- **Total de autómatas con clase única:** {stats_unique_class['total_unique_class']}")
    report.append("")
    
    if stats_unique_class['total_unique_class'] > 0:
        report.append("**Impacto:**")
        report.append("")
        report.append(f"- Estos autómatas representan el {stats_unique_class['total_unique_class']/len(df['dfa_id'].unique())*100:.2f}% del total de autómatas.")
        report.append("")
        report.append("- Los autómatas que solo aceptan o solo rechazan pueden ser problemáticos para")
        report.append("  modelos de aprendizaje automático, ya que no proporcionan variabilidad en las")
        report.append("  etiquetas dentro del mismo autómata.")
        report.append("")
        report.append("**Recomendaciones:**")
        report.append("")
        report.append("1. Considerar filtrar estos autómatas si no aportan información útil.")
        report.append("2. Si se mantienen, asegurarse de que el modelo pueda manejar estos casos extremos.")
        report.append("3. Evaluar si estos autómatas representan casos reales o errores en la generación de datos.")
        report.append("")
        
        if stats_unique_class['only_accept_count'] > 0:
            report.append("**Ejemplos de autómatas que solo aceptan (primeros 5):**")
            report.append("")
            sample_accept = stats_unique_class['only_accept'].head(5)
            report.append("| dfa_id | Total Strings |")
            report.append("|--------|---------------|")
            for _, row in sample_accept.iterrows():
                report.append(f"| {row['dfa_id']} | {row['total_strings']} |")
            report.append("")
        
        if stats_unique_class['only_reject_count'] > 0:
            report.append("**Ejemplos de autómatas que solo rechazan (primeros 5):**")
            report.append("")
            sample_reject = stats_unique_class['only_reject'].head(5)
            report.append("| dfa_id | Total Strings |")
            report.append("|--------|---------------|")
            for _, row in sample_reject.iterrows():
                report.append(f"| {row['dfa_id']} | {row['total_strings']} |")
            report.append("")
    else:
        report.append("✅ **No se encontraron autómatas con clase única.**")
        report.append("")
        report.append("Todos los autómatas tienen al menos una string aceptada y una rechazada.")
        report.append("")
    
    report.append("![Autómatas con Clase Única](figures/unique_class_automatas.png)")
    report.append("")
    
    # 5. Conclusiones y decisiones
    report.append("## 5. Conclusiones y Decisiones")
    report.append("")
    report.append("### Vocabulario")
    report.append("")
    report.append(f"El vocabulario está compuesto por los símbolos: {', '.join(sorted(stats_symbols['global_frequency'].keys()))}")
    report.append("")
    report.append("Estos símbolos representan las letras mayúsculas de A a L, que son válidas para los autómatas.")
    report.append("El símbolo especial `<EPS>` representa la cadena vacía.")
    report.append("")
    
    report.append("### max_len")
    report.append("")
    if stats_len['p99'] <= 64:
        report.append(f"✅ **max_len=64 es apropiado** porque el percentil 99 ({stats_len['p99']:.2f}) está por debajo de este valor.")
    else:
        report.append(f"⚠️ **max_len=64 puede ser insuficiente** porque el percentil 99 ({stats_len['p99']:.2f}) excede este valor.")
        report.append("")
        report.append("**Recomendación:** Considerar aumentar `max_len` o implementar estrategias de truncamiento.")
    report.append("")
    
    report.append("### Manejo de Casos Extremos")
    report.append("")
    report.append("1. **Autómatas con clase única:**")
    report.append(f"   - Se encontraron {stats_unique_class['total_unique_class']} autómatas con clase única.")
    if stats_unique_class['total_unique_class'] > 0:
        report.append("   - Considerar filtrar o manejar estos casos de manera especial.")
    else:
        report.append("   - No se requiere manejo especial.")
    report.append("")
    
    report.append("2. **Autómatas desbalanceados:**")
    report.append(f"   - Se encontraron {stats_labels['imbalanced_count']} autómatas desbalanceados.")
    if stats_labels['imbalanced_count'] > 0:
        report.append("   - Considerar técnicas de balanceo o ponderación de muestras.")
    else:
        report.append("   - El balance es adecuado en la mayoría de los autómatas.")
    report.append("")
    
    report.append("3. **Strings muy largas:**")
    if stats_len['max'] > 64:
        report.append(f"   - Se encontraron strings con longitud máxima de {stats_len['max']}.")
        report.append("   - Implementar truncamiento o padding para strings que excedan `max_len`.")
    else:
        report.append("   - Todas las strings tienen longitud <= 64.")
        report.append("   - No se requiere truncamiento.")
    report.append("")
    
    # Escribir reporte
    report_path = Path(output_file)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"✓ Reporte generado en {output_file}")

def main():
    """
    Función principal que ejecuta todo el análisis EDA.
    """
    # Obtener directorio raíz del proyecto
    project_root = Path(__file__).parent.parent
    
    # Rutas de archivos
    input_file = project_root / 'data' / 'dataset3000_flat_vc.csv'
    
    # Generar nombre del archivo de salida basándose en el archivo de entrada
    input_path = Path(input_file)
    output_file = project_root / 'reports' / f"{input_path.stem}_eda.md"
    figures_dir = project_root / 'reports' / 'figures'
    
    # Crear directorio de figuras
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Cambiar al directorio de trabajo para guardar figuras
    import os
    os.chdir(project_root)
    
    print(f"Leyendo {input_file}...")
    df = pd.read_csv(input_file)
    
    print(f"Total de filas: {len(df):,}")
    print(f"Total de autómatas: {df['dfa_id'].nunique():,}")
    print("")
    
    # Ejecutar análisis
    stats_len = analyze_length_distribution(df)
    stats_labels = analyze_label_balance(df)
    stats_symbols = analyze_symbol_frequency(df)
    stats_unique_class = find_unique_class_automatas(df)
    
    # Generar reporte
    generate_report(stats_len, stats_labels, stats_symbols, stats_unique_class, str(output_file), df)
    
    print("\n" + "="*50)
    print("ANÁLISIS EDA COMPLETADO")
    print("="*50)
    print(f"Reporte generado: {output_file}")
    print(f"Figuras guardadas en: {figures_dir}")

if __name__ == '__main__':
    main()


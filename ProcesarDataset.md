# ProcesarDataset - Resumen del Proceso

## Objetivo

Crear un script unificado (`process_dataset.py`) que procese datasets de autómatas finitos deterministas (DFA) desde su formato original hasta un formato plano validado y serializado, incluyendo análisis exploratorio y versionado.

## ¿Qué se buscaba?

1. **Conversión**: Transformar el formato original (con JSON embebido) a formato plano (una fila por string).
2. **Validación**: Asegurar que los datos cumplan con reglas de negocio (símbolos válidos, sin duplicados, tipos correctos).
3. **Análisis**: Realizar análisis exploratorio para entender la distribución de los datos.
4. **Serialización**: Generar un archivo final ordenado determinísticamente con hash para reproducibilidad.
5. **Versionado**: Mantener metadatos del procesamiento para trazabilidad.

## Estructura del Proceso

### Paso 1: Conversión a Formato Plano

**¿Qué hace?**
- Lee el CSV original que contiene autómatas con su información en formato JSON.
- Extrae cada string del JSON de la columna "Clase" y crea una fila por cada una.
- Convierte el alfabeto de formato con espacios a formato con comas.

**¿Por qué?**
- El formato original tiene datos anidados (JSON) que dificultan el análisis.
- El formato plano permite trabajar fácilmente con cada string individual.
- Facilita el procesamiento posterior y el análisis estadístico.

**Implementación:**
```python
# Para cada autómata, parsea el JSON y crea filas
clase_dict = json.loads(clase_json)
for string, accepted in clase_dict.items():
    # Crea fila con: dfa_id, string, label, regex, alphabet_decl, len
```

### Paso 2: Validación y Limpieza

**¿Qué hace?**
1. **Valida strings**: Verifica que solo contengan símbolos A-L (mayúsculas) o `<EPS>`.
2. **Elimina duplicados**: Detecta y elimina duplicados exactos (mismo `dfa_id` y `string`).
3. **Valida tipos**: Asegura que los tipos de datos sean correctos (int64, object, etc.).
4. **Valida nulos**: Verifica que no haya valores nulos en columnas calculadas.

**¿Por qué?**
- **Validación de strings**: El vocabulario válido es A-L, cualquier otro símbolo es un error.
- **Eliminación de duplicados**: Los duplicados pueden causar problemas en modelos de ML.
- **Validación de tipos**: Asegura consistencia de datos y evita errores en procesamiento posterior.
- **Validación de nulos**: Los nulos en datos calculados indican errores en el proceso.

**Implementación:**
- Regex para validar strings: `^[A-L]+$` o `<EPS>`
- `drop_duplicates()` de pandas manteniendo la primera ocurrencia
- Conversión explícita de tipos y verificación de nulos

### Paso 3: Análisis Exploratorio (EDA)

**¿Qué hace?**
1. **Distribución de longitudes**: Calcula percentiles (p50, p95, p99, max) para justificar `max_len=64`.
2. **Balance de labels**: Analiza la proporción de strings aceptadas (1) vs rechazadas (0).
3. **Frecuencia de símbolos**: Calcula qué símbolos son más comunes globalmente y por autómata.
4. **Autómatas con clase única**: Identifica autómatas que solo aceptan o solo rechazan.

**¿Por qué?**
- **Distribución de longitudes**: Necesario para determinar un `max_len` adecuado que capture la mayoría de los casos.
- **Balance de labels**: Importante para detectar sesgos en los datos que puedan afectar modelos.
- **Frecuencia de símbolos**: Ayuda a entender la distribución del vocabulario.
- **Autómatas con clase única**: Estos casos extremos pueden ser problemáticos para modelos.

**Implementación:**
- Uso de `matplotlib` y `seaborn` para visualizaciones
- Cálculo de estadísticas descriptivas con pandas
- Generación de reporte Markdown con tablas y gráficas

### Paso 4: Serialización

**¿Qué hace?**
1. Ordena los datos determinísticamente (por `dfa_id` y `string`).
2. Guarda el CSV ordenado.
3. Calcula hash SHA256 del archivo generado.

**¿Por qué?**
- **Orden determinístico**: Garantiza que el mismo input produzca el mismo output (reproducibilidad).
- **Hash SHA256**: Permite verificar la integridad del archivo y detectar cambios.
- **Serialización**: Proceso de guardar datos en un formato estable para uso posterior.

**Implementación:**
- Ordenamiento con `sort_values(['dfa_id', 'string'])`
- Cálculo de hash en bloques para eficiencia
- Guardado con nombres basados en el archivo de entrada

### Paso 5: Versionado

**¿Qué hace?**
- Genera archivo JSON con metadatos: fecha, hash, filas, columnas, tipos de datos, etc.

**¿Por qué?**
- **Trazabilidad**: Permite saber cuándo y cómo se procesó el dataset.
- **Reproducibilidad**: El hash permite verificar que el archivo no ha cambiado.
- **Documentación**: Los metadatos documentan la estructura del dataset procesado.

**Implementación:**
- Guardado en `meta/dataset_version.json`
- Incluye información completa del procesamiento

## Decisiones de Diseño

### ¿Por qué un solo script unificado?

**Ventajas:**
- **Simplicidad**: Un solo comando ejecuta todo el proceso.
- **Consistencia**: Garantiza que todos los pasos se ejecuten en orden.
- **Mantenibilidad**: Más fácil de mantener y actualizar.
- **Reproducibilidad**: Un solo script asegura que el proceso sea reproducible.

**Desventajas:**
- Script más largo, pero bien organizado en funciones.
- Menos modular, pero más eficiente al evitar guardar/cargar archivos intermedios.

### ¿Por qué CSV en lugar de Parquet?

- **Compatibilidad**: CSV es más universal y fácil de leer.
- **Simplicidad**: No requiere dependencias adicionales (aunque pandas soporta Parquet).
- **Debugging**: Más fácil inspeccionar y verificar manualmente.
- **Flexibilidad**: Más fácil de procesar con diferentes herramientas.

### ¿Por qué orden determinístico?

- **Reproducibilidad**: El mismo input produce el mismo hash.
- **Consistencia**: Los resultados son predecibles y verificables.
- **Debugging**: Más fácil comparar resultados entre ejecuciones.

### ¿Por qué SHA256 para el hash?

- **Seguridad**: SHA256 es resistente a colisiones y ampliamente usado.
- **Confianza**: Algoritmo estándar y probado.
- **Longitud**: 64 caracteres hex proporcionan suficiente entropía.

### ¿Por qué logging en lugar de print?

- **Profesionalismo**: Logging es más apropiado para scripts de producción.
- **Niveles**: Permite controlar la verbosidad (INFO, WARNING, ERROR).
- **Formato**: Formato estructurado con timestamps.
- **Debugging**: Más fácil de filtrar y analizar logs.

## Estructura del Script

```
process_dataset.py
├── Funciones de conversión
│   └── convert_to_flat()
├── Funciones de validación
│   ├── validate_string()
│   ├── validate_and_fix_types()
│   └── validate_and_clean()
├── Funciones de EDA
│   ├── analyze_eda()
│   ├── analyze_length_distribution()
│   ├── analyze_label_balance()
│   ├── analyze_symbol_frequency()
│   ├── find_unique_class_automatas()
│   └── generate_eda_report()
├── Funciones de serialización
│   ├── calculate_file_hash()
│   ├── serialize_csv()
│   └── generate_version_metadata()
└── Función principal
    └── main()
```

## Archivos Generados

### Archivos de Salida

1. **`data/{nombre}_procesado.csv`**
   - Archivo final serializado y validado
   - Formato: CSV con columnas: dfa_id, string, label, regex, alphabet_decl, len
   - Ordenado determinísticamente

2. **`reports/{nombre}_eda.md`**
   - Reporte de análisis exploratorio
   - Incluye tablas y referencias a gráficas
   - Explicaciones y justificaciones

3. **`reports/figures/*.png`**
   - Gráficas del análisis EDA
   - 4 gráficas: distribución de len, balance de labels, frecuencia de símbolos, autómatas con clase única

4. **`meta/dataset_version.json`**
   - Metadatos de versión
   - Incluye: fecha, hash, filas, columnas, tipos, tamaño

5. **`data/{nombre}_duplicates_log.csv`** (si hay duplicados)
   - Log de duplicados encontrados y eliminados

6. **`data/{nombre}_quarantine.csv`** (si hay filas problemáticas)
   - Filas con strings inválidos que fueron marcadas para cuarentena

## Uso del Script

```bash
# Ejecutar el script con un archivo CSV
python scripts/process_dataset.py dataset3000.csv

# Ver ayuda
python scripts/process_dataset.py --help
```

### Requisitos

- El archivo de entrada debe estar en el directorio raíz del proyecto
- El archivo debe tener el formato esperado (columnas: Regex, Alfabeto, Clase, etc.)
- Python 3.7+ con las librerías: pandas, matplotlib, seaborn, numpy

## Resultados Esperados

### Para dataset3000.csv:

- **Archivo final**: `data/dataset3000_procesado.csv`
  - 192,119 filas
  - 6 columnas
  - ~7.8 MB
  - Hash SHA256: `93808e01f7f5153d87ae139f67f613c93a03270c4598630db539e411ef7ad118`

- **Reporte EDA**: `reports/dataset3000_eda.md`
  - Estadísticas de distribución de longitudes
  - Análisis de balance de labels
  - Frecuencia de símbolos
  - Autómatas con clase única

- **Metadatos**: `meta/dataset_version.json`
  - Información completa del procesamiento
  - Hash para verificación
  - Tipos de datos y estructura

## Ventajas del Enfoque

1. **Reproducibilidad**: Orden determinístico y hash garantizan resultados consistentes.
2. **Trazabilidad**: Metadatos documentan todo el proceso.
3. **Validación**: Múltiples capas de validación aseguran calidad de datos.
4. **Análisis**: EDA integrado proporciona insights inmediatos.
5. **Flexibilidad**: Script unificado pero con funciones modulares.
6. **Robustez**: Manejo de errores y logging para debugging.

## Consideraciones Futuras

1. **Parquet**: Considerar usar Parquet para mejor compresión y rendimiento.
2. **Paralelización**: Algunos pasos podrían paralelizarse para datasets más grandes.
3. **Validación adicional**: Agregar más validaciones según necesidades del proyecto.
4. **Configuración**: Hacer algunos parámetros configurables (ej: max_len, compresión).
5. **Testing**: Agregar tests unitarios para cada función.

## Conclusiones

El script `process_dataset.py` unifica todo el proceso de conversión, validación, análisis y serialización de datasets de autómatas en un solo pipeline robusto y reproducible. Utiliza mejores prácticas (logging, manejo de errores, orden determinístico) para asegurar calidad y trazabilidad de los datos procesados.


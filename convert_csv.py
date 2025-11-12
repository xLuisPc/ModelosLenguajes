import pandas as pd
import json
import csv

def convert_csv(input_file, output_file):
    """
    Convierte el CSV del formato original al formato plano (flat).
    
    Formato original:
    - Regex, Alfabeto, Estados de aceptación, Estados, Transiciones, Clase, Error
    
    Formato nuevo:
    - dfa_id, string, label, regex, alphabet_decl, len
    """
    # Leer el CSV original
    print(f"Leyendo {input_file}...")
    df = pd.read_csv(input_file)
    
    # Lista para almacenar las filas del nuevo formato
    new_rows = []
    
    # Procesar cada fila (cada autómata)
    for idx, row in df.iterrows():
        dfa_id = idx  # Empieza en 0
        regex = row['Regex']
        alphabet = row['Alfabeto']
        clase_json = row['Clase']
        
        # Convertir el alfabeto de espacios a comas
        alphabet_decl = ', '.join(alphabet.split())
        
        # Parsear el JSON de la columna Clase
        try:
            clase_dict = json.loads(clase_json)
        except json.JSONDecodeError as e:
            print(f"Error al parsear JSON en fila {idx}: {e}")
            continue
        
        # Para cada string en el JSON, crear una fila
        for string, accepted in clase_dict.items():
            # Si la cadena está vacía, usar <EPS>
            display_string = '<EPS>' if string == '' else string
            
            # Label: 1 si acepta, 0 si rechaza
            label = 1 if accepted else 0
            
            # Longitud: 0 si es <EPS>, sino la longitud de la cadena
            string_len = 0 if string == '' else len(string)
            
            # Agregar la fila
            new_rows.append({
                'dfa_id': dfa_id,
                'string': display_string,
                'label': label,
                'regex': regex,
                'alphabet_decl': alphabet_decl,
                'len': string_len
            })
    
    # Crear DataFrame con las nuevas filas
    new_df = pd.DataFrame(new_rows)
    
    # Ordenar por dfa_id y luego por string
    new_df = new_df.sort_values(['dfa_id', 'string'])
    
    # Guardar el nuevo CSV
    print(f"Guardando {output_file}...")
    new_df.to_csv(output_file, index=False)
    
    print(f"Conversión completada. Total de filas: {len(new_df)}")
    print(f"Total de autómatas (dfa_id): {new_df['dfa_id'].nunique()}")
    
    return new_df

if __name__ == '__main__':
    input_file = 'dataset3000.csv'
    output_file = 'dataset3000_flat.csv'
    
    convert_csv(input_file, output_file)


import pandas as pd
from datetime import datetime

# 1. Cargar el archivo original
df = pd.read_csv("beneficiarios_comedor_2023_unheval.csv", encoding='latin1')

# 2. Convertir la columna de fecha de nacimiento a datetime
df['FECHA_NACIMIENTO'] = pd.to_datetime(df['FECHA_NACIMIENTO'], format='%Y%m%d', errors='coerce')

# 3. Calcular edad con base en 31 de diciembre de 2023
fecha_actual = datetime(2023, 12, 31)

def calcular_edad(fecha_nac, fecha_actual):
    return fecha_actual.year - fecha_nac.year - ((fecha_actual.month, fecha_actual.day) < (fecha_nac.month, fecha_nac.day))

df['EDAD_CALCULADA'] = df['FECHA_NACIMIENTO'].apply(lambda x: calcular_edad(x, fecha_actual) if pd.notnull(x) else None)
df['EDAD_CORRECTA'] = df['EDAD'] == df['EDAD_CALCULADA']
df = df[df['EDAD_CORRECTA'] == True]  # Filtrar solo edades correctas

# 4. Eliminar columnas innecesarias o redundantes
df = df.drop(columns=[
    'UUID',             # Datos anonimizados, no aportan al análisis
    'FECHA_CORTE',      # Fecha de corte del dataset
    'FECHA_NACIMIENTO', # Ya usamos EDAD
    'EDAD_CALCULADA',   # Ya fue usada para validar
    'EDAD_CORRECTA'     # Columna auxiliar
])

# 5. Revisión de valores nulos o atípicos en raciones
# Eliminar o corregir registros con valores negativos o nulos
for col in ['N_RAC_DESAYUNO', 'N_RAC_ALMUERZO', 'N_RAC_CENA']:
    df = df[df[col].notnull()]  # eliminar nulos
    df = df[df[col] >= 0]       # eliminar negativos (si los hay)

# 6. Convertir variables categóricas a variables numéricas
# Puedes usar get_dummies() para regresión lineal

# 7. Guardar dataset limpio
df.to_csv("beneficiarios_comedor_2023_unheval_limpio_final.csv", index=False)

# 8. Vista previa del resultado
print("✅ Dataset limpio guardado con éxito. Total de registros finales:", len(df))
print("✅ Columnas finales:", df.columns.tolist())

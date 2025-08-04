import pandas as pd
import numpy as np

def limpiar(df):
    # 1. Convertir columnas con símbolos a float
    for col in ['precio', 'ingreso']:
        df[col] = df[col].replace('[\$,]', '', regex=True).astype(float)

    # 2. Convertir fechas a datetime si no lo son
    if not np.issubdtype(df['fecha_venta'].dtype, np.datetime64):
        df['fecha_venta'] = pd.to_datetime(df['fecha_venta'], errors='coerce')

    # 3. Imputar valores numéricos faltantes
    df['precio'] = df['precio'].fillna(df['precio'].mean())
    df['cantidad'] = df['cantidad'].fillna(df['cantidad'].median())

    # 4. Imputar fechas faltantes con fechas válidas aleatorias
    fechas_validas = df['fecha_venta'].dropna().values
    if len(fechas_validas) > 0:
        indices_nan = df[df['fecha_venta'].isna()].index
        for idx in indices_nan:
            fecha_random = np.random.choice(fechas_validas)
            df.at[idx, 'fecha_venta'] = pd.Timestamp(fecha_random)
    else:
        print("⚠️ No hay fechas válidas para asignar.")

    # 5. Recalcular ingreso si falta
    df['ingreso'] = df['precio'] * df['cantidad']

    return df


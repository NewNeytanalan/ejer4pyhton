//Migrar el dataset CSV a ARFF utilizando pandas y scipy
import pandas as pd
from scipy.io import arff

# Cargar el dataset CSV
df = pd.read_csv('peliculas_marvel.csv')

# Guardar el dataset como ARFF
def save_to_arff(df, file_name):
    # Convertir el DataFrame a un formato compatible para ARFF
    arff_data = {
        'description': '',
        'relation': 'marvel_movies',
        'attributes': [],
        'data': df.values
    }
    
    for col in df.columns:
        # Si la columna es numérica, usar 'NUMERIC'
        if df[col].dtype in ['int64', 'float64']:
            arff_data['attributes'].append((col, 'NUMERIC'))
        # Si es categórica, extraer las categorías
        else:
            unique_vals = df[col].unique()
            arff_data['attributes'].append((col, list(unique_vals)))
    
    # Guardar en formato ARFF
    with open(file_name, 'w') as f:
        arff.dump(arff_data, f)

# Guardar el archivo en formato ARFF
save_to_arff(df, 'peliculas_marvel.arff')
//OneHotEncoder y LabelEncoder con scikit-learn
from sklearn.preprocessing import LabelEncoder

# Crear el LabelEncoder para la columna 'Fase'
label_encoder = LabelEncoder()

# Aplicar la codificación
df['Fase_encoded'] = label_encoder.fit_transform(df['Fase'])

# Ver los resultados
print(df[['Fase', 'Fase_encoded']].head())
----------------------------------------------------------------------
from sklearn.preprocessing import OneHotEncoder

# Crear el OneHotEncoder para la columna 'Personajes Principales'
onehot_encoder = OneHotEncoder(sparse=False)

# Aplicar el OneHotEncoder
onehot_encoded = onehot_encoder.fit_transform(df[['Personajes Principales']])

# Crear un nuevo DataFrame con las columnas codificadas
onehot_df = pd.DataFrame(onehot_encoded, columns=onehot_encoder.get_feature_names_out(['Personajes Principales']))

# Combinar el DataFrame codificado con el original
df = pd.concat([df, onehot_df], axis=1)

# Ver los resultados
print(df.head())

---------------------------------------------------------------------------------
from sklearn.preprocessing import KBinsDiscretizer

# Crear un discretizador
discretizer = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='uniform')

# Aplicar la discretización a la columna 'Recaudación Mundial (USD)'
df['Recaudacion_discretizada'] = discretizer.fit_transform(df[['Recaudación Mundial (USD)']])

# Ver los resultados
print(df[['Recaudación Mundial (USD)', 'Recaudacion_discretizada']].head())
------------------------------------------------------------
from sklearn.preprocessing import MinMaxScaler

# Crear un MinMaxScaler
scaler = MinMaxScaler()

# Aplicar la normalización a la columna 'Presupuesto (USD)'
df['Presupuesto_normalizado'] = scaler.fit_transform(df[['Presupuesto (USD)']])

# Ver los resultados
print(df[['Presupuesto (USD)', 'Presupuesto_normalizado']].head())
-------------------------------------------------------------------------------
from sklearn.preprocessing import StandardScaler

# Crear un StandardScaler
scaler = StandardScaler()

# Aplicar la estandarización a la columna 'Duración (min)'
df['Duracion_estandarizada'] = scaler.fit_transform(df[['Duración (min)']])

# Ver los resultados
print(df[['Duración (min)', 'Duracion_estandarizada']].head())


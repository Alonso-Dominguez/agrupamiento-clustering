import numpy as np  
import pandas as pd
from sklearn.cluster import KMeans 
import joblib
import matplotlib.pyplot as plt

# 1. Importar el dataset con los datos de entrenamiento
df_datos_clientes = pd.read_csv("clientes_entrenamiento.csv")  # ✅ Verifica que el nombre del archivo esté bien escrito

print(df_datos_clientes.info())
print(df_datos_clientes.head())

# 2. Convertir el dataframe a un array de Numpy
X = df_datos_clientes.values
print(X)

# 3. Entrenar el modelo
modelo = KMeans(n_clusters=3, random_state=1234, n_init=10)  # ✅ Correcciones: 'n_clusters' y 'n_init' con nombres correctos
modelo.fit(X)

# 4. Análisis del modelo
df_datos_clientes['cluster'] = modelo.labels_  # ✅ Corrección: 'labels_' con guion bajo
analisis = df_datos_clientes.groupby('cluster').mean()
print(analisis)

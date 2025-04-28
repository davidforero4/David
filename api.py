#!/usr/bin/env python
# coding: utf-8

# In[234]:


import warnings
warnings.filterwarnings('ignore')


# In[235]:


# Importación librerías
import pandas as pd
import numpy as np


# In[236]:


# Carga de datos de archivo .csv
dataTraining = pd.read_csv('https://raw.githubusercontent.com/davidzarruk/MIAD_ML_NLP_2025/main/datasets/dataTrain_Spotify.csv')
dataTesting = pd.read_csv('https://raw.githubusercontent.com/davidzarruk/MIAD_ML_NLP_2025/main/datasets/dataTest_Spotify.csv', index_col=0)


# In[237]:


# Visualización datos de entrenamiento
dataTraining.head()


# In[238]:


# Visualización datos de test
dataTesting.head()


# In[239]:


# Predicción del conjunto de test - acá se genera un número aleatorio como ejemplo
np.random.seed(42)
y_pred = pd.DataFrame(np.random.rand(dataTesting.shape[0]) * 100, index=dataTesting.index, columns=['Popularity'])


# In[240]:


# Guardar predicciones en formato exigido en la competencia de kaggle
y_pred.to_csv('test_submission_file.csv', index_label='ID')
y_pred.head()


# In[241]:


dataTraining.shape


# In[242]:


dataTesting.shape


# In[243]:


import requests
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib


# In[244]:


print(dataTraining.shape)
print(dataTesting.shape)


# In[245]:


pd.set_option('display.max_columns', 21)
pd.set_option('display.max_rows', 30)
dataTraining.head()


# In[246]:


dataTraining.info()


# In[247]:


# Cargue librerias adicionales
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Preprocesamiento de datos

# Separar características (X) y variable objetivo (y)
X = dataTraining.drop(['popularity', 'Unnamed: 0', 'track_id', 'track_name', 'artists', 'album_name', 'track_genre'], axis=1)  # Se eliminan columnas no relevantes
y = dataTraining['popularity']

# Dividir en conjuntos de entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Codificación de variables categóricas
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
categorical_features = ['key', 'mode', 'explicit']
encoded_train = encoder.fit_transform(X_train[categorical_features])
encoded_val = encoder.transform(X_val[categorical_features])

# Crea un DataFrame con las variables codificadas
encoded_train_df = pd.DataFrame(encoded_train, columns=encoder.get_feature_names_out(categorical_features), index=X_train.index)
encoded_val_df = pd.DataFrame(encoded_val, columns=encoder.get_feature_names_out(categorical_features), index=X_val.index)

# Concatena las variables codificadas con las numéricas
X_train = pd.concat([X_train.drop(categorical_features, axis=1), encoded_train_df], axis=1)
X_val = pd.concat([X_val.drop(categorical_features, axis=1), encoded_val_df], axis=1)

# Exploración final
print(X_train.head())
print(X_train.info())
print(X_train.describe())
print(X_train.isnull().sum())


# In[248]:


# Selección y entrenamiento de modelos

# Define los modelos a probar
models = {
    'Linear Regression': LinearRegression(),
}

# Crea una lista para almacenar las métricas
results = []

# Entrena y evalúa cada modelo
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    results.append([name, mse, r2])  # Guarda las métricas en la lista

# Crea un DataFrame con los resultados
results_df = pd.DataFrame(results, columns=['Model', 'MSE', 'R2'])

# Muestra la tabla de resultados
print(results_df)


# In[249]:


joblib.dump(model, 'modelo_popularidad_canciones.pkl', compress=3)


# In[ ]:


import joblib
from flask import Flask
from flask_restx import Api, Resource, fields, reqparse

# Guardar el mejor modelo entrenado
joblib.dump(model, 'modelo_popularidad_canciones.pkl')
print("Modelo guardado exitosamente.")

# Definir la aplicación Flask
app = Flask(__name__)

# Definir la API
api = Api(
    app,
    version='1.0',
    title='Predicción Popularidad Canciones API',
    description='API para predecir la popularidad de una canción en Spotify'
)

ns = api.namespace('predict', description='Predicción de Popularidad')

# Definición de los argumentos de entrada
parser = reqparse.RequestParser()
parser.add_argument('duration_ms', type=int, required=True, help='duration_ms', location='args')
parser.add_argument('danceability', type=float, required=True, help='Danceability de la canción', location='args')
parser.add_argument('energy', type=float, required=True, help='Energy de la canción', location='args')
parser.add_argument('loudness', type=float, required=True, help='loudness de la canción', location='args')
parser.add_argument('key_0', type=int, required=True, help='Key de la canción', location='args')
parser.add_argument('key_1', type=int, required=True, help='key_1 de la canción', location='args')
parser.add_argument('key_2', type=int, required=True, help='key_2 de la canción', location='args')
parser.add_argument('key_3', type=int, required=True, help='key_3 de la canción', location='args')
parser.add_argument('key_4', type=int, required=True, help='key_4 de la canción', location='args')
parser.add_argument('key_5', type=int, required=True, help='key_5 de la canción', location='args')
parser.add_argument('key_6', type=int, required=True, help='key_6 de la canción', location='args')
parser.add_argument('key_7', type=int, required=True, help='key_7 de la canción', location='args')
parser.add_argument('key_8', type=int, required=True, help='key_8 de la canción', location='args')
parser.add_argument('key_9', type=int, required=True, help='key_9 de la canción', location='args')
parser.add_argument('key_10', type=int, required=True, help='key_10 de la canción', location='args')
parser.add_argument('key_11', type=int, required=True, help='key_11 de la canción', location='args')
parser.add_argument('mode_0', type=int, required=True, help='key_1 de la canción', location='args')
parser.add_argument('mode_1', type=int, required=True, help='key_1 de la canción', location='args')
parser.add_argument('explicit_False', type=float, required=True, help=' explicit_False de la canción', location='args')
parser.add_argument('explicit_True', type=float, required=True, help=' explicit_True de la canción', location='args')
parser.add_argument('speechiness', type=float, required=True, help='Speechiness de la canción', location='args')
parser.add_argument('acousticness', type=float, required=True, help='Acousticness de la canción', location='args')
parser.add_argument('instrumentalness', type=float, required=True, help='Instrumentalness de la canción', location='args')
parser.add_argument('liveness', type=float, required=True, help='Liveness de la canción', location='args')
parser.add_argument('valence', type=float, required=True, help='Valence de la canción', location='args')
parser.add_argument('tempo', type=float, required=True, help='Tempo de la canción', location='args')
parser.add_argument('time_signature', type=int, required=True, help='Time signature de la canción', location='args')

# Definición del formato de respuesta
resource_fields = api.model('Resource', {
    'popularity': fields.Float
})

# Definición de la clase para disponibilizar la predicción
@ns.route('/')
class PopularidadAPI(Resource):
    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()

        # Se convierte la entrada en una matriz de características
        input_features = [[

            args['duration_ms'], args['danceability'], args['energy'],
            args['loudness'], args['key_0'], args['key_1'],
            args['key_2'], args['key_3'], args['key_4'],
            args['key_5'], args['key_6'], args['key_7'],
            args['key_8'], args['key_9'], args['key_10'],
            args['key_11'], args['mode_0'], args['mode_1'],
            args['explicit_False'], args['explicit_True'], args['speechiness'],
            args['acousticness'], args['instrumentalness'], args['liveness'],
            args['valence'], args['tempo'], args['time_signature']
        ]]

        # Cargar el modelo
        modelo = joblib.load('modelo_popularidad_canciones.pkl')

        # Hacer predicción
        prediction = modelo.predict(input_features)[0]

        return {'popularity': prediction}, 200

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





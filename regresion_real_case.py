import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Cargar el archivo CSV winequality-red.csv
df = pd.read_csv('winequality-red.csv', delimiter=';')

# Separar las características (X) y la variable objetivo (y)
X = df.drop(columns=['quality'])
y = df['quality']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Inicializar y entrenar el modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluar el modelo
train_score = model.score(X_train, y_train)
print("Train score:", train_score)
test_score = model.score(X_test, y_test)
print("Test score:", test_score)

# Predicciones
y_pred = model.predict(X_test)

# Calcular métricas de evaluación
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2_score = metrics.r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R-squared Score:", r2_score)

# Gráfico de dispersión de valores reales vs. predichos
plt.scatter(y_test, y_pred)
plt.xlabel("Valores reales")
plt.ylabel("Valores predichos")
plt.title("Valores reales vs. predichos")
plt.show()

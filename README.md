# Regresiones_2
Se realizo una comparación de modelos de regresión lineal aplicados a diferentes conjuntos de datos. Se utilizaron dos conjuntos de datos distintos: uno relacionado con gastos de marketing y ventas, y otro relacionado con la calidad del vino tinto. Se aplico regresión lineal simple en ambos conjuntos de datos y se evaluaron los modelos resultantes.

# Metodologıa
Se utilizo el lenguaje de programacion Python y las bibliotecas ‘pandas‘, ‘numpy‘, ‘matplotlib.pyplot‘ y ‘scikit-learn‘ para la implementacion de los modelos de regresion lineal. Se dividio cada conjunto de datos en conjuntos de entrenamiento y prueba, se entrenaron modelos de regresion lineal simple y se evaluaron utilizando metricas como el coeficiente de determinacion, el error absoluto medio y el error cuadratico medio.

# Resultados
Los resultados obtenidos para cada conjunto de datos fueron los siguientes:

![tv](https://github.com/LuisRosado/Regresiones_2/assets/140114139/5609e25f-790d-4f6a-8353-67100300ae51)

Figura 1: Grafica de dispersion de tvmarketing.csv.csv.

![Wine](https://github.com/LuisRosado/Regresiones_2/assets/140114139/47f41bff-873a-4ea4-af5e-f1344096557a)

Figura 2: Grafica de dispersion de winequality.csv.

Conjunto de datos: Gastos de marketing y ventas
- Coeficiente de determinacion (R²) en el conjunto de entrenamiento: 0.651
- Coeficiente de determinacion (R²) en el conjunto de prueba: 0.623
- Error absoluto medio (MAE): 1434.29
- Error cuadr´atico medio (MSE):3168794.73
- Raız del error cuadratico medio (RMSE): 1780.34
  
Conjunto de datos: Calidad del vino tinto
- Coeficiente de determinacion (R²) en el conjunto de entrenamiento: 0.358
- Coeficiente de determinacion (R²) en el conjunto de prueba: 0.328
- Error absoluto medio (MAE): 0.545
- Error cuadratico medio (MSE): 0.539
- Raız del error cuadratico medio (RMSE): 0.734

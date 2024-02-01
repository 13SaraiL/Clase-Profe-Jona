
import numpy as np
import pandas as pd
# Definir la función de activación
def activation_function(x):
    return 1 if x >= 0 else 0

# Definir los pesos y el umbral
weights = np.array([0.5, 0.5])  # Pesos para las entradas x1 y x2
bias = -0.7  # Umbral

# Función para cargar datos de una hoja de Excel
def cargar_datos_excel(archivo_excel):
    df = pd.read_excel(archivo_excel)
    X = df[['Input1', 'Input2']]
    y = df['Output']
    return X, y

# Definir la función del perceptrón AND
def perceptron_and(x1, x2):
    inputs = np.array([x1, x2])  # Vector de entradas
    weighted_sum = np.dot(inputs, weights) + bias  # Suma ponderada
    output = activation_function(weighted_sum)  # Aplicar la función de activación
    return output

# Cargar datos de una hoja de Excel para la compuerta AND
X_and, y_and = cargar_datos_excel('datos_and.xlsx')

# Probar el perceptrón AND con los datos cargados de Excel
for i in range(len(X_and)):
    x1, x2 = X_and.iloc[i]
    print(f"Entradas: {x1}, {x2} - Salida: {perceptron_and(x1, x2)}")

# Cargar datos de una hoja de Excel para la compuerta OR
X_or, y_or = cargar_datos_excel('datos_or.xlsx')

# Definir la función del perceptrón OR
def perceptron_or(x1, x2):
    inputs = np.array([x1, x2])  # Vector de entradas
    weighted_sum = np.dot(inputs, weights) + bias  # Suma ponderada
    output = activation_function(weighted_sum)  # Aplicar la función de activación
    return output

# Probar el perceptrón OR con los datos cargados de Excel
for i in range(len(X_or)):
    x1, x2 = X_or.iloc[i]
    print(f"Entradas: {x1}, {x2} - Salida: {perceptron_or(x1, x2)}")

# -*- coding: utf-8 -*-
"""Implementando_ejercicio_1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1FgZm_Go7qoo2_-Wb6xzkl-vy6gbLd8IE
"""

#aplicando el ejercicio de aplicacion practica
#de el mismo codigo compartido en clases se aimportan las librerias necesarias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Implementar el cálculo de los coeficientes paso a paso
def calcular_regresion_lineal(x, y, mostrar_pasos=True):
    """Calcula los coeficientes de regresión lineal mostrando cada paso"""

    n = len(x)

    # Paso 1: Calcular las medias
    x_media = np.mean(x)
    y_media = np.mean(y)

    if mostrar_pasos:
        print("=== PASO 1: Calcular las medias ===")
        print(f"x̄ = Σx/n = {np.sum(x):.2f}/{n} = {x_media:.2f}")
        print(f"ȳ = Σy/n = {np.sum(y):.2f}/{n} = {y_media:.2f}")

    # Paso 2: Calcular las diferencias
    x_diff = x - x_media
    y_diff = y - y_media

    if mostrar_pasos:
        print("\n=== PASO 2: Calcular diferencias respecto a la media ===")
        print("Primeros 5 valores:")
        for i in range(min(5, n)):
            print(f"x[{i}] - x̄ = {x[i]:.2f} - {x_media:.2f} = {x_diff[i]:.2f}")

    # Paso 3: Calcular las sumas necesarias
    suma_xy = np.sum(x_diff * y_diff)
    suma_xx = np.sum(x_diff * x_diff)

    if mostrar_pasos:
        print(f"\n=== PASO 3: Calcular sumas ===")
        print(f"Σ(x - x̄)(y - ȳ) = {suma_xy:.2f}")
        print(f"Σ(x - x̄)² = {suma_xx:.2f}")

    # Paso 4: Calcular la pendiente (m)
    m = suma_xy / suma_xx

    if mostrar_pasos:
        print(f"\n=== PASO 4: Calcular la pendiente ===")
        print(f"m = Σ(x - x̄)(y - ȳ) / Σ(x - x̄)²")
        print(f"m = {suma_xy:.2f} / {suma_xx:.2f} = {m:.2f}")

    # Paso 5: Calcular el intercepto (b)
    b = y_media - m * x_media

    if mostrar_pasos:
        print(f"\n=== PASO 5: Calcular el intercepto ===")
        print(f"b = ȳ - m * x̄")
        print(f"b = {y_media:.2f} - {m:.2f} * {x_media:.2f} = {b:.2f}")

    return m, b

def practica_regresion(x_nuevo, y_nuevo, titulo="Tu Regresión"):
    """
    Función para que los estudiantes practiquen con sus propios datos
    """
    # Calcular coeficientes
    m, b = calcular_regresion_lineal(x_nuevo, y_nuevo, mostrar_pasos=False)

    # Crear visualización
    plt.figure(figsize=(10, 6))
    plt.scatter(x_nuevo, y_nuevo, s=100, alpha=0.7, label='Tus datos')

    # Línea de regresión
    x_linea = np.linspace(np.min(x_nuevo), np.max(x_nuevo), 100)
    y_linea = m * x_linea + b
    plt.plot(x_linea, y_linea, 'r-', linewidth=3,
             label=f'Regresión: y = {m:.2f}x + {b:.2f}')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(titulo)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Mostrar estadísticas
    y_pred = m * x_nuevo + b
    mse = np.mean((y_nuevo - y_pred)**2)
    print(f"Ecuación: y = {m:.2f}x + {b:.2f}")
    print(f"MSE: {mse:.2f}")

    return m, b

# Ejemplo 1: Horas de estudio vs Calificación
# Para comprender la funcionaldiad del mismo, se cambiara los valores
print("=== EJEMPLO 1: Horas de Estudio vs Calificación ===")
horas_estudio = np.array([8, 7, 6, 5, 4, 3, 2, 1])
calificaciones = np.array([100, 95, 79, 80, 64, 50, 13, 11])

m1, b1 = practica_regresion(horas_estudio, calificaciones,
                           "Horas de Estudio vs Calificación")

print("\nInterpretación:")
print(f"- Por cada hora adicional de estudio, la calificación aumenta {m1:.1f} puntos")
print(f"- Sin estudiar (0 horas), la calificación esperada es {b1:.1f}")
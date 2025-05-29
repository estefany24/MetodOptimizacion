import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyRegressor

df = pd.read_csv("beneficiarios_comedor_2023_unheval_limpio_final.csv")


X = df[['EDAD', 'ANIO_MAT1', 'N_RAC_DESAYUNO', 'N_RAC_CENA']]
y = df['N_RAC_ALMUERZO']

print((y <= 0).sum()) 
mask = y > 0
X, y = X[mask], y[mask]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)

print("📊 Comparación de modelos de regresión:\n")
print(models)

print((y <= 0).sum())  
mask = y > 0
X, y = X[mask], y[mask]


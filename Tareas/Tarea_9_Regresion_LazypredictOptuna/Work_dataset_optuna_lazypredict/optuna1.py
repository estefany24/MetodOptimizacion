import optuna
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
df = pd.read_csv("beneficiarios_comedor_2023_unheval_limpio_final.csv")[lambda x: x['N_RAC_ALMUERZO'] > 0]

X = df[['EDAD', 'ANIO_MAT1', 'N_RAC_DESAYUNO', 'N_RAC_CENA']]
y = df['N_RAC_ALMUERZO']

def objective(trial):
    model = LinearRegression()
    
    # Validación cruzada
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(model, X, y, cv=kf, scoring='neg_root_mean_squared_error')
    return -1.0 * score.mean()  # Minimizar RMSE

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

print("Mejor RMSE:", study.best_value)
print("Mejores hiperparámetros:", study.best_params)

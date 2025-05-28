import optuna, numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold

df = pd.read_csv("beneficiarios_comedor_2023_unheval_limpio_final.csv")[lambda x: x['N_RAC_ALMUERZO'] > 0]
X, y = df[['EDAD', 'ANIO_MAT1', 'N_RAC_DESAYUNO', 'N_RAC_CENA']], df['N_RAC_ALMUERZO']

def objective(trial): return -cross_val_score(LinearRegression(), X, y, cv=KFold(5, shuffle=True, random_state=42), scoring='neg_root_mean_squared_error').mean()

study = optuna.create_study(direction="minimize").optimize(objective, n_trials=30)
print(f"Mejor RMSE: {study.best_value:.4f}\nMejores hiperparámetros: {study.best_params}")
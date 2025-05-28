import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from lazypredict.Supervised import LazyRegressor
from sklearn.linear_model import Ridge
import optuna

# Leer y preparar los datos
df = pd.read_csv("beneficiarios_comedor_2023_unheval_limpio_final.csv")
df = df[df['N_RAC_ALMUERZO'] > 0]
X = df[['EDAD', 'ANIO_MAT1', 'N_RAC_DESAYUNO', 'N_RAC_CENA']]
y = df['N_RAC_ALMUERZO']

# Separar entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Parte 1: LazyRegressor para elegir el mejor modelo
lazy = LazyRegressor(verbose=0, ignore_warnings=True)

models, predictions = lazy.fit(X_train, X_test, y_train, y_test)

print("Modelos evaluados con LazyRegressor:\n")
print(models)

# Parte 2: Tuning de hiperparámetros con Optuna para Ridge
def objective(trial):
    alpha = trial.suggest_float('alpha', 1e-4, 100.0, log=True)
    model = Ridge(alpha=alpha)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(model, X, y, cv=kf, scoring='neg_root_mean_squared_error')
    return -score.mean()  # Minimizar RMSE

# Ejecutar búsqueda de hiperparámetros
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

# Resultados
print("\n Optuna - Mejor resultado para Ridge:")
print(" Mejor alpha:", study.best_params)
print(" Mejor RMSE:", study.best_value)

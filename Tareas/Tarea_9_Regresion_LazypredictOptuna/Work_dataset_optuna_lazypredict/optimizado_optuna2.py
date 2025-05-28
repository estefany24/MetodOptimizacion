import pandas as pd; from sklearn.model_selection import train_test_split, KFold, cross_val_score
from lazypredict.Supervised import LazyRegressor; from sklearn.linear_model import Ridge; import optuna

df = pd.read_csv("beneficiarios_comedor_2023_unheval_limpio_final.csv").query('N_RAC_ALMUERZO > 0')
X, y = df[['EDAD', 'ANIO_MAT1', 'N_RAC_DESAYUNO', 'N_RAC_CENA']], df['N_RAC_ALMUERZO']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Modelos evaluados:\n", LazyRegressor(verbose=0, ignore_warnings=True).fit(X_train, X_test, y_train, y_test)[0])

def objective(trial): return -cross_val_score(Ridge(alpha=trial.suggest_float('alpha', 1e-4, 100.0, log=True)), X, y, cv=KFold(5, shuffle=True, random_state=42), scoring='neg_root_mean_squared_error').mean()

study = optuna.create_study(direction="minimize").optimize(objective, n_trials=30)
print("\nOptuna - Ridge:\n Mejor alpha:", study.best_params, "\n Mejor RMSE:", study.best_value)
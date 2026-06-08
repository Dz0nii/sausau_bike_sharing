import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib
import os

#========================
# 1: UCITAVANJE PODATAKA
#========================

df = pd.read_csv('data/processed/hour_processed.csv')
X  = df.drop(columns=['cnt'])
y  = df['cnt']

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42)

os.makedirs('plots', exist_ok=True)

#==================
# 2: RANDOM FOREST
#==================

rf_grid = GridSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    param_grid={
        'n_estimators':      [100, 200],
        'max_depth':         [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf':  [1, 2]
    },
    cv=5, scoring='r2', verbose=1, n_jobs=-1
)
rf_grid.fit(X_train, y_train)
# print(f"RF  — Najbolji parametri: {rf_grid.best_params_}")
# print(f"RF  — Najbolji CV R²:     {rf_grid.best_score_:.4f}")

#============
# 3: XGBOOST
#============

xgb_random = RandomizedSearchCV(
    XGBRegressor(random_state=42, verbosity=0),
    param_distributions={
        'n_estimators':     [100, 200, 300],
        'max_depth':        [3, 5, 7, 9],
        'learning_rate':    [0.01, 0.05, 0.1, 0.2],
        'subsample':        [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
    },
    n_iter=20, cv=5, scoring='r2', verbose=1, random_state=42, n_jobs=-1
)
xgb_random.fit(X_train, y_train)
# print(f"XGB — Najbolji parametri: {xgb_random.best_params_}")
# print(f"XGB — Najbolji CV R²:     {xgb_random.best_score_:.4f}")

#==============================================
# 4: POREDJENJE ORIGINALNIH I PODESENIH MODELA
#==============================================

modeli = {
    "RF  — originalni": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "RF  — podešeni":   rf_grid.best_estimator_,
    "XGB — originalni": XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
    "XGB — podešeni":   xgb_random.best_estimator_
}

rezultati = []
# print("\nPoređenje na validacionom skupu:")
for naziv, model in modeli.items():
    if "originalni" in naziv:
        model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    mae  = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2   = r2_score(y_val, y_pred)
    rezultati.append({'Model': naziv, 'MAE': round(mae,2), 
                      'RMSE': round(rmse,2), 'R²': round(r2,4)})
    # print(f"{naziv:20s} | MAE: {mae:6.2f} | RMSE: {rmse:6.2f} | R²: {r2:.4f}")

#======================
# 5: GRAFIK POREDJENJA
#======================

rezultati_df = pd.DataFrame(rezultati)

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for i, (metrika, boja) in enumerate(zip(['MAE', 'RMSE', 'R²'], ['coral', 'orange', 'steelblue'])):
    axes[i].bar(rezultati_df['Model'], rezultati_df[metrika], color=boja, edgecolor='white')
    axes[i].set_title(f'Originalni vs. Podešeni — {metrika}')
    axes[i].tick_params(axis='x', rotation=20)

plt.tight_layout()
plt.savefig('plots/08_hyperparameter_tuning.png', dpi=150)
plt.show()

#=====================================
# 6: IZBOR I CUVANJE NAJBOLJEG MODELA
#=====================================

rf_r2  = r2_score(y_val, rf_grid.best_estimator_.predict(X_val))
xgb_r2 = r2_score(y_val, xgb_random.best_estimator_.predict(X_val))

if rf_r2 >= xgb_r2:
    najbolji_tuned       = rf_grid.best_estimator_
    najbolji_tuned_naziv = "Random Forest (podešeni)"
else:
    najbolji_tuned       = xgb_random.best_estimator_
    najbolji_tuned_naziv = "XGBoost (podešeni)"

# finalna evaluacija na test skupu
y_pred_test = najbolji_tuned.predict(X_test)
print(f"\nNajbolji podešeni model: {najbolji_tuned_naziv}")
print(f"Test MAE:  {mean_absolute_error(y_test, y_pred_test):.2f}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):.2f}")
print(f"Test R²:   {r2_score(y_test, y_pred_test):.4f}")

joblib.dump(najbolji_tuned,       'models/najbolji_model_tuned.pkl')
joblib.dump(najbolji_tuned_naziv, 'models/najbolji_model_naziv.pkl')
rezultati_df.to_csv('models/rezultati_tuning.csv', index=False)
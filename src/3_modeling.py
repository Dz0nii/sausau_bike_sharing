import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import joblib
import os

#========================
# 1: UCITAVANJE PODATAKA

df = pd.read_csv('data/processed/hour_processed.csv')
X  = df.drop(columns=['cnt'])
y  = df['cnt']

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42)

# print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

#=======================
# 2: DEFINISANJE MODELA

modeli = {
    "Linearna regresija (baseline)": Pipeline([
        ('scaler', StandardScaler()),
        ('model',  LinearRegression())
    ]),
    "Random Forest":    RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "XGBoost":          XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
}

#============================
# 3: TRENIRANJE I EVALUACIJA

os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

rezultati = []

for naziv, model in modeli.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    mae  = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2   = r2_score(y_val, y_pred)

    rezultati.append({'Model': naziv, 'MAE': round(mae,2), 
                      'RMSE': round(rmse,2), 'R²': round(r2,4)})
    print(f"{naziv:35s} | MAE: {mae:6.2f} | RMSE: {rmse:6.2f} | R²: {r2:.4f}")

#=====================
# 4: CROSS-VALIDATION

# print("\nCross-validation (5-fold R²):")
for naziv, model in modeli.items():
    cv = cross_val_score(model, X_temp, y_temp, cv=5, scoring='r2', n_jobs=-1)
    # print(f"{naziv:35s} | Prosek: {cv.mean():.4f} | Std: {cv.std():.4f}")

#======================
# 5: GRAFIK POREDJENJA

rezultati_df = pd.DataFrame(rezultati).sort_values('R²', ascending=False).reset_index(drop=True)

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for i, (metrika, boja) in enumerate(zip(['MAE', 'RMSE', 'R²'], ['coral', 'orange', 'steelblue'])):
    axes[i].bar(rezultati_df['Model'], rezultati_df[metrika], color=boja, edgecolor='white')
    axes[i].set_title(f'Poređenje — {metrika}')
    axes[i].tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig('plots/07_poredjenje_modela.png', dpi=150)
plt.show()

#===================
# 6: CUVANJE MODELA

najbolji_naziv = rezultati_df.iloc[0]['Model']
najbolji_model = modeli[najbolji_naziv]

joblib.dump(najbolji_model,  'models/najbolji_model.pkl')
joblib.dump(list(X.columns), 'models/feature_names.pkl')
joblib.dump(X_test,          'models/X_test.pkl')
joblib.dump(y_test,          'models/y_test.pkl')
rezultati_df.to_csv('models/rezultati_modela.csv', index=False)

print(f"\nNajbolji model: {najbolji_naziv}")
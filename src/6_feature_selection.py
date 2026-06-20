import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import joblib
import os

#=================================
# 1: UCITAVANJE PODATAKA I MODELA

model  = joblib.load('models/najbolji_model_tuned.pkl')
naziv  = joblib.load('models/najbolji_model_naziv.pkl')
X_test = joblib.load('models/X_test.pkl')
y_test = joblib.load('models/y_test.pkl')

df = pd.read_csv('data/processed/hour_processed.csv')
X  = df.drop(columns=['cnt'])
y  = df['cnt']

X_temp, _, y_temp, _ = train_test_split(X, y, test_size=0.15, random_state=42)
X_train, _, y_train, _ = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42)

os.makedirs('plots', exist_ok=True)

#=======================
# 2: FEATURE IMPORTANCE

importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
importances.head(15).plot(kind='barh', color='steelblue', edgecolor='white')
plt.gca().invert_yaxis()
plt.title(f'Feature Importance (ugrađena) — {naziv}')
plt.xlabel('Važnost')
plt.tight_layout()
plt.savefig('plots/12_feature_importance.png', dpi=150)
plt.show()

#===========================
# 3: PERMUTATION IMPORTANCE

perm_imp = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
perm_series = pd.Series(perm_imp.importances_mean, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
perm_series.head(15).plot(kind='barh', color='coral', edgecolor='white')
plt.gca().invert_yaxis()
plt.title(f'Permutation Importance — {naziv}')
plt.xlabel('Prosečno smanjenje R²')
plt.tight_layout()
plt.savefig('plots/13_permutation_importance.png', dpi=150)
plt.show()

#==========================
# 4: POREDJENJE DVE METODE

df_comp = pd.DataFrame({
    'Ugradjena':   importances / importances.max(),
    'Permutation': perm_series / perm_series.max()
}).dropna().sort_values('Ugradjena', ascending=False).head(15)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, kolona, boja in zip(axes, ['Ugradjena', 'Permutation'], ['steelblue', 'coral']):
    df_comp[kolona].plot(kind='barh', ax=ax, color=boja, edgecolor='white')
    ax.invert_yaxis()
    ax.set_title(f'{kolona} Importance')
    ax.set_xlabel('Važnost (normalizovana)')

plt.suptitle('Poređenje metoda za feature importance', fontsize=13)
plt.tight_layout()
plt.savefig('plots/14_poredjenje_importances.png', dpi=150)
plt.show()

#=================================
# 5: TRENIRANJE SA TOP ATRIBUTIMA

top_features = importances.head(15).index.tolist()

params = {k: v for k, v in model.get_params().items()
          if k in ['n_estimators', 'max_depth', 'min_samples_split',
                   'min_samples_leaf', 'learning_rate', 'subsample',
                   'colsample_bytree', 'random_state']}

model_top = (RandomForestRegressor(**params, n_jobs=-1) if "Random Forest" in naziv
             else XGBRegressor(**params, verbosity=0))

model_top.fit(X_train[top_features], y_train)

y_pred_svi = model.predict(X_test)
y_pred_top = model_top.predict(X_test[top_features])

print(f"\nPoređenje — svi atributi ({X.shape[1]}) vs. top 15:")
print(f"{'Metrika':6s} | {'Svi':>10s} | {'Top 15':>10s}")
print("-" * 32)
for ime, f in [('MAE',  mean_absolute_error),
               ('RMSE', lambda a,b: np.sqrt(mean_squared_error(a,b))),
               ('R²',   r2_score)]:
    print(f"{ime:6s} | {f(y_test, y_pred_svi):10.4f} | {f(y_test, y_pred_top):10.4f}")

#======================
# 6: CUVANJE REZULTATA

joblib.dump(top_features, 'models/top_features.pkl')
pd.DataFrame({'Atribut': importances.index,
              'Ugradjena': importances.values,
              'Permutation': perm_series.values
              }).to_csv('models/feature_importance.csv', index=False)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

#=================================
# 1: UCITAVANJE PODATAKA I MODELA

model  = joblib.load('models/najbolji_model_tuned.pkl')
naziv  = joblib.load('models/najbolji_model_naziv.pkl')
X_test = joblib.load('models/X_test.pkl')
y_test = joblib.load('models/y_test.pkl')
df_raw = pd.read_csv('data/hour.csv')
os.makedirs('plots', exist_ok=True)

y_pred    = model.predict(X_test)
reziduali = y_test - y_pred

#===================
# 2: PRIKAZ METRIKE

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print(f"Model: {naziv}")
print(f"MAE:   {mae:.2f} ({mae/y_test.mean()*100:.1f}% proseka)")
print(f"RMSE:  {rmse:.2f}")
print(f"R²:    {r2:.4f} — model objašnjava {r2*100:.1f}% varijabilnosti")

#====================================
# 3: STVARNE I PREDVIDJENE VREDNOSTI

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.15, color='steelblue', s=10)
plt.plot([0, 1000], [0, 1000], 'r--', linewidth=2, label='Savršena predikcija')
plt.xlabel('Stvarne vrednosti')
plt.ylabel('Predviđene vrednosti')
plt.title(f'Stvarne vs. Predviđene — {naziv}')
plt.legend()
plt.tight_layout()
plt.savefig('plots/09_stvarne_vs_predvidjene.png', dpi=150)
plt.show()

#==============
# 4: REZIDUALI

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(reziduali, bins=50, color='steelblue', edgecolor='white')
axes[0].axvline(x=0, color='red', linestyle='--')
axes[0].set_title('Raspodela grešaka')
axes[0].set_xlabel('Greška (stvarna - predviđena)')
axes[0].set_ylabel('Frekvencija')

axes[1].scatter(y_pred, reziduali, alpha=0.15, color='coral', s=10)
axes[1].axhline(y=0, color='red', linestyle='--')
axes[1].set_title('Reziduali vs. Predviđene vrednosti')
axes[1].set_xlabel('Predviđene vrednosti')
axes[1].set_ylabel('Rezidual')

plt.tight_layout()
plt.savefig('plots/10_reziduali.png', dpi=150)
plt.show()

#============================
# 5: GRESKA PO SATU I SEZONI

df_test = df_raw.loc[X_test.index].copy()
df_test['abs_greska'] = np.abs(reziduali.values)

season_labels = {1: 'Proleće', 2: 'Leto', 3: 'Jesen', 4: 'Zima'}
df_test['season_naziv'] = df_test['season'].map(season_labels)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

df_test.groupby('hr')['abs_greska'].mean().plot(
    kind='bar', ax=axes[0], color='steelblue', edgecolor='white')
axes[0].set_title('Prosečna greška po satu')
axes[0].set_xlabel('Sat')
axes[0].set_ylabel('Prosečna greška')

df_test.groupby('season_naziv')['abs_greska'].mean().plot(
    kind='bar', ax=axes[1], color='coral', edgecolor='white')
axes[1].set_title('Prosečna greška po sezoni')
axes[1].set_xlabel('Sezona')
axes[1].set_ylabel('Prosečna greška')

plt.tight_layout()
plt.savefig('plots/11_greska_po_satu_sezoni.png', dpi=150)
plt.show()
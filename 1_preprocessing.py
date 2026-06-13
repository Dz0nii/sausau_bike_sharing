import pandas as pd
import os

#========================
# 1: UCITAVANJE PODATAKA
#========================
df = pd.read_csv('data/hour.csv')

#===================================
# 2: PROVERA PRAZNIH POLJA U TABELI
#===================================
# print("\nBroj nedostajućih vrednosti po koloni:")
# print(df.isnull().sum())
# print("\nUkupno nedostajućih vrednosti:", df.isnull().sum().sum())

#======================
# 3: PROVERA ANOMALIJA
#======================
anomalije = {
    'season':     (1, 4),
    'hr':         (0, 23),
    'mnth':       (1, 12),
    'weekday':    (0, 6),
    'holiday':    (0, 1),
    'workingday': (0, 1),
    'weathersit': (1, 4),
    'temp':       (0, 1),
    'atemp':      (0, 1),
    'hum':        (0, 1),
    'windspeed':  (0, 1),
    'cnt':        (1, 1000)
}

# for kolona, (min_val, max_val) in anomalije.items():
#     van_opsega = df[(df[kolona] < min_val) | (df[kolona] > max_val)]
#     print(f"  {kolona}: {len(van_opsega)} anomalija")

#========================
# 4: UKLANJANJE ATRIBUTA
#========================
# instant: predstavlja redni broj, nema prediktivnu vrednost
df.drop(columns=['instant'], inplace=True)

# dteday: datum je vec rastavljen u yr, mnth, hr kolone
df.drop(columns=['dteday'], inplace=True)

# casual i registered: direktno cine cnt (data leakage)
df.drop(columns=['casual', 'registered'], inplace=True)

# print("\nPreostale kolone:", df.columns.tolist())
# print("Novi oblik dataseta:", df.shape)

#=======================================
# 5: ENKODIRANJE KATEGORIJSKIH ATRIBUTA
#=======================================
# season -> season_2, season_3, season_4 (season_1 kolona nije neophodna)
# isti princip se vrsi i za ostale kategorijske kolone
kategorijske_kolone = ['season', 'hr', 'mnth', 'weekday', 'weathersit']
df_encoded = pd.get_dummies(df, columns=kategorijske_kolone, drop_first=True)

#================================
# 6: CUVANJE OBRADJENIH PODATAKA
#================================
os.makedirs('data/processed', exist_ok=True)
df_encoded.to_csv('data/processed/hour_processed.csv', index=False)
#print("Finalni oblik dataseta:", df_encoded.shape)
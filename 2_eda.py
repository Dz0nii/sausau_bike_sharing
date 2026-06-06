import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

#========================
# 1: UCITAVANJE PODATAKA
#========================
# ucitava se i originalan csv zato sto enkodovane kolone nisu citljive na graficima
df_raw = pd.read_csv('data/hour.csv')
df = pd.read_csv('data/processed/hour_processed.csv')

os.makedirs('plots', exist_ok=True)

#==============================
# 2: ANALIZA CILJNE PROMENJIVE
#==============================
# print("Statistike ciljne promenljive (cnt):")
# print(df_raw['cnt'].describe())

plt.figure(figsize=(12, 4))

# koliko cesto se koja vrednost cnt pojavljuje
plt.subplot(1, 2, 1)
df_raw['cnt'].hist(bins=50, color='steelblue', edgecolor='white')
plt.title('Raspodela broja iznajmljivanja (cnt)')
plt.xlabel('Broj iznajmljivanja')
plt.ylabel('Frekvencija')

# logaritamska raspodela
# pokazuje da li je raspodela iskrivljena (skewed)
# zbog prirode podataka, pokazuje da ekstremne vrednosti nisu greske vec realna potraznja tokom odredjenih sati
plt.subplot(1, 2, 2)
np.log1p(df_raw['cnt']).hist(bins=50, color='coral', edgecolor='white')
plt.title('Log raspodela broja iznajmljivanja')
plt.xlabel('log(cnt + 1)')
plt.ylabel('Frekvencija')

plt.tight_layout()
plt.savefig('plots/01_raspodela_cnt.png', dpi=150)
plt.show()
# print("Sačuvano: plots/01_raspodela_cnt.png")

#===========================
# 3: IZNAJMLJIVANJA PO SATU
#===========================
plt.figure(figsize=(12, 5))

po_satu = df_raw.groupby('hr')['cnt'].mean()

plt.plot(po_satu.index, po_satu.values, marker='o', color='steelblue', linewidth=2)
plt.fill_between(po_satu.index, po_satu.values, alpha=0.2, color='steelblue')
plt.title('Prosečan broj iznajmljivanja po satu')
plt.xlabel('Sat u danu')
plt.ylabel('Prosečan broj iznajmljivanja')
plt.xticks(range(0, 24))
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('plots/02_iznajmljivanja_po_satu.png', dpi=150)
plt.show()
# print("Sačuvano: plots/02_iznajmljivanja_po_satu.png")

#=========================
# 4: KATEGORIJSKE ANALIZE
#=========================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# sezona
season_labels = {1: 'Proleće', 2: 'Leto', 3: 'Jesen', 4: 'Zima'}
po_sezoni = df_raw.groupby('season')['cnt'].mean()
po_sezoni.index = po_sezoni.index.map(season_labels)
po_sezoni.plot(kind='bar', ax=axes[0, 0], color='teal', edgecolor='white')
axes[0, 0].set_title('Prosečan broj iznajmljivanja po sezoni')
axes[0, 0].set_xlabel('')
axes[0, 0].tick_params(axis='x', rotation=0)

# mesec
df_raw.groupby('mnth')['cnt'].mean().plot(kind='bar', ax=axes[0, 1], color='orange', edgecolor='white')
axes[0, 1].set_title('Prosečan broj iznajmljivanja po mesecu')
axes[0, 1].set_xlabel('Mesec')
axes[0, 1].tick_params(axis='x', rotation=0)

# radni dan
workingday_labels = {0: 'Vikend/Praznik', 1: 'Radni dan'}
po_radnom = df_raw.groupby('workingday')['cnt'].mean()
po_radnom.index = po_radnom.index.map(workingday_labels)
po_radnom.plot(kind='bar', ax=axes[1, 0], color='green', edgecolor='white')
axes[1, 0].set_title('Prosečan broj iznajmljivanja: radni dan vs. vikend')
axes[1, 0].set_xlabel('')
axes[1, 0].tick_params(axis='x', rotation=0)

# praznik
holiday_labels = {0: 'Nije praznik', 1: 'Praznik'}
po_prazniku = df_raw.groupby('holiday')['cnt'].mean()
po_prazniku.index = po_prazniku.index.map(holiday_labels)
po_prazniku.plot(kind='bar', ax=axes[1, 1], color='red', edgecolor='white')
axes[1, 1].set_title('Prosečan broj iznajmljivanja: praznik vs. običan dan')
axes[1, 1].set_xlabel('')
axes[1, 1].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig('plots/03_kategorijske_analize.png', dpi=150)
plt.show()
# print("Sačuvano: plots/03_kategorijske_analize.png")

#=====================
# 5: VREMENSKI USLOVI
#=====================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# temperatura
axes[0].scatter(df_raw['temp'], df_raw['cnt'], alpha=0.05, color='steelblue')
axes[0].set_title('Temperatura vs. Broj iznajmljivanja')
axes[0].set_xlabel('Temperatura')
axes[0].set_ylabel('Broj iznajmljivanja')

# vlaznost
axes[1].scatter(df_raw['hum'], df_raw['cnt'], alpha=0.05, color='coral')
axes[1].set_title('Vlažnost vs. Broj iznajmljivanja')
axes[1].set_xlabel('Vlažnost')
axes[1].set_ylabel('Broj iznajmljivanja')

# brzina vetra
axes[2].scatter(df_raw['windspeed'], df_raw['cnt'], alpha=0.05, color='green')
axes[2].set_title('Brzina vetra vs. Broj iznajmljivanja')
axes[2].set_xlabel('Brzina vetra')
axes[2].set_ylabel('Broj iznajmljivanja')

plt.tight_layout()
plt.savefig('plots/04_vremenski_uslovi.png', dpi=150)
plt.show()
# print("Sačuvano: plots/04_vremenski_uslovi.png")

#========================
# 6: KORELACIONA MATRICA
#========================
plt.figure(figsize=(10, 8))

numericki = ['temp', 'atemp', 'hum', 'windspeed', 'yr',  'holiday', 'workingday', 'cnt']

korelacija = df_raw[numericki].corr()

sns.heatmap(
    korelacija,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    center=0,
    square=True,
    linewidths=0.5
)

plt.title('Korelaciona matrica numeričkih atributa')
plt.tight_layout()
plt.savefig('plots/05_korelaciona_matrica.png', dpi=150)
plt.show()
# print("Sačuvano: plots/05_korelaciona_matrica.png")

# print("\nNajveće korelacije sa cnt:")
# print(korelacija['cnt'].sort_values(ascending=False))

#=============================================================
# 7: VREMENSKI USLOVI (weathersit atribut) (mozda nepotrebno)
#=============================================================
plt.figure(figsize=(10, 5))

weather_labels = {
    1: 'Vedro',
    2: 'Magla/Oblačno',
    3: 'Kiša/Sneg',
    4: 'Jak sneg/Kiša'
}

df_plot = df_raw.copy()
df_plot['weathersit'] = df_plot['weathersit'].map(weather_labels)

sns.boxplot(
    data=df_plot,
    x='weathersit',
    y='cnt',
    order=['Vedro', 'Magla/Oblačno', 'Kiša/Sneg', 'Jak sneg/Kiša'],
    palette='Blues'
)

plt.title('Raspodela iznajmljivanja po vremenskim uslovima')
plt.xlabel('Vremenski uslovi')
plt.ylabel('Broj iznajmljivanja')
plt.tight_layout()
plt.savefig('plots/06_vremenski_uslovi_boxplot.png', dpi=150)
plt.show()
# print("Sačuvano: plots/06_vremenski_uslovi_boxplot.png")
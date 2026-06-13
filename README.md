# Bike Sharing — Predikcija broja iznajmljenih bicikala
Predmet: SAUSAU
FTN Novi Sad

# Opis projekta

Model mašinskog učenja koji predviđa ukupan broj iznajmljenih bicikala
u sistemu Capital Bikeshare na osnovu vremenskih, sezonskih i kalendarskih karakteristika.
Dataset pokriva period 2011–2012. godine i sadrži 17.379 zapisa.

# Struktura projekta

projekat/

├── data/

│   ├── hour.csv                  ← originalni dataset

│   └── processed/

│       └── hour_processed.csv    ← obrađeni dataset

├── models/                       ← sačuvani modeli i rezultati

├── plots/                        ← grafici

├── 1_preprocessing.py            ← preprocesiranje podataka

├── 2_eda.py                      ← eksplorativna analiza

├── 3_modeling.py                 ← treniranje modela

├── 4_hyperparameter_tuning.py    ← podešavanje hiperparametara

├── 5_results_analysis.py         ← analiza rezultata

├── 6_feature_selection.py        ← odabir atributa

├── 7_deployment.py               ← terminal UI

└── app.py                        ← Streamlit web UI

# Pokretanje

Faze se pokreću redom iz terminala:

```bash
python 1_preprocessing.py
python 2_eda.py
python 3_modeling.py
python 4_hyperparameter_tuning.py
python 5_results_analysis.py
python 6_feature_selection.py
python 7_deployment.py
```

Za pokretanje web UI:

```bash
streamlit run app.py
```

# Rezultati

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| Linearna regresija (baseline) | 74.05 | 99.55 | 0.6731 |
| Random Forest | 35.14 | 55.17 | 0.8996 |
| Gradient Boosting | 59.09 | 80.48 | 0.7864 |
| XGBoost | 33.24 | 50.21 | 0.9168 |

# Finalni model — XGBoost (podešeni)

| Metrika | Vrednost |
|---------|----------|
| MAE | 30.79 |
| RMSE | 47.31 |
| R² | 0.9293 |
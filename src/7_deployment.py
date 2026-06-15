import pandas as pd
import joblib

model         = joblib.load('models/najbolji_model_tuned.pkl')
naziv         = joblib.load('models/najbolji_model_naziv.pkl')
feature_names = joblib.load('models/feature_names.pkl')

# dozvoljeni opsezi za svaki atribut
OPSEZI = {
    'season':     (int,   1,   4),
    'yr':         (int,   0,   1),
    'mnth':       (int,   1,  12),
    'hr':         (int,   0,  23),
    'holiday':    (int,   0,   1),
    'weekday':    (int,   0,   6),
    'workingday': (int,   0,   1),
    'weathersit': (int,   1,   4),
    'temp':       (float, 0.0, 1.0),
    'atemp':      (float, 0.0, 1.0),
    'hum':        (float, 0.0, 1.0),
    'windspeed':  (float, 0.0, 1.0),
}

def pripremi_ulaz(season, yr, mnth, hr, holiday, weekday,
                  workingday, weathersit, temp, atemp, hum, windspeed):
    ulaz = {
        'yr': yr, 'holiday': holiday, 'workingday': workingday,
        'temp': temp, 'atemp': atemp, 'hum': hum, 'windspeed': windspeed
    }

    for kolona, vrednost, opseg in [
        ('season',     season,     range(1, 5)),
        ('hr',         hr,         range(0, 24)),
        ('mnth',       mnth,       range(1, 13)),
        ('weekday',    weekday,    range(0, 7)),
        ('weathersit', weathersit, range(1, 5))
    ]:
        for i in list(opseg)[1:]:
            ulaz[f'{kolona}_{i}'] = 1 if vrednost == i else 0

    return pd.DataFrame([ulaz]).reindex(columns=feature_names, fill_value=0)

def predvidi(*args):
    return max(0, int(round(model.predict(pripremi_ulaz(*args))[0])))

if __name__ == "__main__":
    print(f"\n{'='*50}\n  PREDIKCIJA BROJA IZNAJMLJENIH BICIKALA\n  Model: {naziv}\n{'='*50}")
    try:
        ulazi = [
            ("Sezona        (1=Proleće, 2=Leto, 3=Jesen, 4=Zima)", 'season'),
            ("Godina        (0=2011, 1=2012)",                      'yr'),
            ("Mesec         (1-12)",                                 'mnth'),
            ("Sat           (0-23)",                                 'hr'),
            ("Praznik       (0=ne, 1=da)",                          'holiday'),
            ("Dan u nedelji (0=ned, 1=pon, ..., 6=sub)",            'weekday'),
            ("Radni dan     (0=ne, 1=da)",                          'workingday'),
            ("Vreme         (1=vedro, 2=magla, 3=kiša, 4=sneg)",    'weathersit'),
            ("Temperatura   (0.0-1.0 normalizovana)",                'temp'),
            ("Feels like    (0.0-1.0 normalizovana)",                'atemp'),
            ("Vlažnost      (0.0-1.0 normalizovana)",                'hum'),
            ("Brzina vetra  (0.0-1.0 normalizovana)",                'windspeed'),
        ]

        vrednosti = []
        for tekst, kljuc in ulazi:
            tip, min_val, max_val = OPSEZI[kljuc]
            while True:
                try:
                    vrednost = tip(input(f"{tekst}: "))
                    if min_val <= vrednost <= max_val:
                        vrednosti.append(vrednost)
                        break
                    else:
                        print(f"  ✗ Vrednost mora biti između {min_val} i {max_val}, pokušaj ponovo.")
                except ValueError:
                    print(f"  ✗ Unesite broj tipa {tip.__name__}, pokušaj ponovo.")

        rezultat = predvidi(*vrednosti)
        print(f"\n{'='*50}\n  Predviđen broj iznajmljivanja: {rezultat} bicikala\n{'='*50}\n")

    except KeyboardInterrupt:
        print("\n\nPrekinuto.")
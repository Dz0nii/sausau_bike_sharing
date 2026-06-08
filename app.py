import streamlit as st
import pandas as pd
import joblib

# --- Učitavanje ---
model         = joblib.load('models/najbolji_model_tuned.pkl')
naziv         = joblib.load('models/najbolji_model_naziv.pkl')
feature_names = joblib.load('models/feature_names.pkl')

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

# --- UI ---
st.title("🚲 Predikcija iznajmljivanja bicikala")
st.caption(f"Model: {naziv}")
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Kalendar")
    season     = st.selectbox("Sezona",        [1, 2, 3, 4],
                               format_func=lambda x: {1:"Proleće", 2:"Leto", 3:"Jesen", 4:"Zima"}[x])
    yr         = st.selectbox("Godina",        [0, 1],
                               format_func=lambda x: {0:"2011", 1:"2012"}[x])
    mnth       = st.slider("Mesec",            1, 12, 6)
    hr         = st.slider("Sat",              0, 23, 17)
    weekday    = st.selectbox("Dan u nedelji", [0,1,2,3,4,5,6],
                               format_func=lambda x: ["Ned","Pon","Uto","Sre","Čet","Pet","Sub"][x])
    holiday    = st.selectbox("Praznik",       [0, 1],
                               format_func=lambda x: {0:"Ne", 1:"Da"}[x])
    workingday = st.selectbox("Radni dan",     [0, 1],
                               format_func=lambda x: {0:"Ne", 1:"Da"}[x])

with col2:
    st.subheader("Vreme")
    weathersit = st.selectbox("Vremenski uslovi", [1, 2, 3, 4],
                               format_func=lambda x: {
                                   1:"☀️ Vedro",
                                   2:"🌫️ Magla/Oblačno",
                                   3:"🌧️ Kiša/Sneg",
                                   4:"⛈️ Jak sneg/Kiša"}[x])
    temp       = st.slider("Temperatura (normalizovana)",  0.0, 1.0, 0.5, 0.01)
    atemp      = st.slider("Feels like (normalizovana)",   0.0, 1.0, 0.5, 0.01)
    hum        = st.slider("Vlažnost (normalizovana)",     0.0, 1.0, 0.5, 0.01)
    windspeed  = st.slider("Brzina vetra (normalizovana)", 0.0, 1.0, 0.2, 0.01)

with col3:
    st.subheader("Predikcija")
    rezultat = predvidi(season, yr, mnth, hr, holiday, weekday,
                        workingday, weathersit, temp, atemp, hum, windspeed)

    st.metric(label="Predviđen broj iznajmljivanja", value=f"{rezultat} 🚲")
    st.divider()

    # Kontekst
    if rezultat < 50:
        st.info("🔵 Niska potražnja")
    elif rezultat < 200:
        st.success("🟢 Umerena potražnja")
    elif rezultat < 400:
        st.warning("🟡 Visoka potražnja")
    else:
        st.error("🔴 Veoma visoka potražnja")

    st.divider()
    st.caption("Uputstvo za temperaturu:")
    st.caption("0.0 ≈ -16°C  |  0.5 ≈ 16°C  |  1.0 ≈ 47°C")
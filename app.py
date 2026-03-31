import streamlit as st
import pickle
import numpy as np

with open('churn_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title("Prédicteur de churn client")

tenure = st.slider("Ancienneté (mois)", 0, 72, 12)
monthly = st.slider("Charges mensuelles ($)", 0, 120, 50)
contract = st.selectbox("Type de contrat", [0, 1, 2],
                        format_func=lambda x: ["Mensuel","1 an","2 ans"][x])

if st.button("Prédire"):
    # Exemple simplifié — adapte les features à ton dataset
    features = np.zeros((1, 19))
    features[0, 4] = tenure
    features[0, 18] = monthly
    features[0, 6] = contract
    features_scaled = scaler.transform(features)
    proba = model.predict_proba(features_scaled)[0][1]
    st.metric("Probabilité de churn", f"{proba:.1%}")
    if proba > 0.5:
        st.error("Client à risque élevé de churn")
    else:
        st.success("Client probablement fidèle")
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns

#chargement des données
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
import pandas as pd
import numpy as np

print(df.shape)
df.head()
#exploration des données
print(df.info())
print("\n--- Valeurs manquantes ---")
print(df.isnull().sum())
print("\n--- Distribution du churn ---")
print(df['Churn'].value_counts(normalize=True))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Distribution churn
df['Churn'].value_counts().plot(kind='bar', ax=axes[0], color=['#4C9BE8',"#D8764B"])
axes[0].set_title('Distribution du churn')
axes[0].set_xticklabels(['Non', 'Oui'], rotation=0)

# Churn par type de contrat
pd.crosstab(df['Contract'], df['Churn'], normalize='index').plot(
    kind='bar', ax=axes[1], color=['#4C9BE8','#E87B4C'])
axes[1].set_title('Churn par type de contrat')
axes[1].legend(['Non', 'Oui'])

# Churn selon l'ancienneté
df.boxplot(column='tenure', by='Churn', ax=axes[2])
axes[2].set_title('Ancienneté vs Churn')

plt.tight_layout()
plt.show()

# Supprimer la colonne ID inutile
df = df.drop('customerID', axis=1)

# TotalCharges est en string à cause de valeurs vides — corriger ça
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Remplir les 11 valeurs manquantes par la médiane
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Convertir la cible en binaire
df['Churn'] = (df['Churn'] == 'Yes').astype(int)

print("Nettoyage terminé. Shape:", df.shape)
print(df['Churn'].value_counts())


from sklearn.preprocessing import LabelEncoder

# Séparer colonnes numériques et catégorielles
cat_cols = df.select_dtypes(include='object').columns
print("Colonnes catégorielles:", list(cat_cols))

# Encoder chaque colonne catégorielle
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

print("\nAperçu après encodage:")
df.head()


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df.drop('Churn', axis=1)
y = df['Churn']

# Split 80% train / 20% test, stratifié pour garder les proportions de churn
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Normaliser les features numériques
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    results[name] = {'model': model, 'auc': auc, 'y_pred': y_pred}
    print(f"\n{'='*40}")
    print(f"  {name}  —  AUC: {auc:.3f}")
    print(classification_report(y_test, y_pred, target_names=['Non churn','Churn']))


from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (name, res) in zip(axes, results.items()):
    ConfusionMatrixDisplay.from_predictions(
        y_test, res['y_pred'],
        display_labels=['Non churn', 'Churn'],
        ax=ax, colorbar=False, cmap='Blues'
    )
    ax.set_title(f'{name}\nAUC: {res["auc"]:.3f}')

plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
for name, res in results.items():
    RocCurveDisplay.from_predictions(
        y_test,
        res['model'].predict_proba(X_test_scaled)[:, 1],
        name=name, ax=ax
    )
ax.set_title('Courbes ROC — comparaison des modèles')
plt.show()


rf_model = results['Random Forest']['model']

importances = pd.Series(
    rf_model.feature_importances_,
    index=X.columns
).sort_values(ascending=True).tail(10)

plt.figure(figsize=(8, 5))
importances.plot(kind='barh', color='#4C9BE8')
plt.title('Top 10 features les plus importantes')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()


import pickle

# Sauvegarder le meilleur modèle et le scaler
best_model = results['Gradient Boosting']['model']

with open('churn_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Modèle sauvegardé !")

# Tester sur un nouveau client fictif
nouveau_client = X_test.iloc[0:1].copy()
nouveau_client_scaled = scaler.transform(nouveau_client)
proba = best_model.predict_proba(nouveau_client_scaled)[0][1]
print(f"\nProbabilité de churn pour ce client : {proba:.1%}")

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
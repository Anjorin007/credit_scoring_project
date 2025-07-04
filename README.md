# 💳 Système Intelligent de Scoring de Crédit

Un prototype complet d'IA pour évaluer le risque de défaut d’un client bancaire, avec interprétation SHAP et interface Streamlit interactive.  
📊 Conçu pour les microfinances, banques et néobanques africaines.

---

## 🎯 Objectif du projet

Prédire si un client est à **risque de non-remboursement**, en se basant sur des données socio-financières (âge, revenus, historiques de paiement…).

---

## ⚙️ Données utilisées

- Données : [Give Me Some Credit (Kaggle)](https://www.kaggle.com/c/GiveMeSomeCredit)
- Format : `.csv` avec 10 variables explicatives + 1 variable cible (`SeriousDlqin2yrs`)

---

## 🧠 Modèle entraîné

- 📌 **XGBoostClassifier** (avec gestion des valeurs manquantes)
- 🎯 Métriques : AUC, Précision, Rappel, Matrice de confusion
- 🔍 Interprétabilité : **SHAP values** pour expliquer chaque prédiction
- ✍️ Résumé textuel généré via **Cohere AI**

---

## 📈 Résultats

- Interface Streamlit intuitive
- Visualisation SHAP + Résumé métier
- Export **PDF** et **CSV** du rapport client
- Prédiction personnalisée temps réel

---

## 🧪 Technologies utilisées

- `Python 3.10+`
- `xgboost`, `scikit-learn`, `pandas`, `numpy`
- `matplotlib`, `shap`, `joblib`
- `streamlit`
- `cohere` (IA pour résumé automatique)
- `fpdf` pour export PDF

---

## 📁 Structure du projet

```plaintext
credit_scoring_project/
│
├── app/
│   └── streamlit_app.py         # Interface utilisateur Streamlit
├── notebooks/
│   └── eda_model.ipynb          # Analyse exploratoire + modèle
├── models/
│   └── xgb_model.pkl            # Modèle XGBoost sauvegardé
├── data/
│   └── cs_train.csv             # Données brutes (Kaggle)
├── screenshots/
│   └── *.png                    # Captures d'écran de l'app
├── reports/
│   └── rapport_client.pdf       # Rapport IA exporté
├── requirements.txt             # Dépendances
└── README.md                    # Ce fichier

## ✅ Installation rapide

```bash
git clone https://github.com/ton-utilisateur/credit_scoring_project.git
cd credit_scoring_project
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py

Excellent ! Voici **tout ce que tu m’as envoyé** **transformé intégralement en vrai Markdown**, sans rien hors syntaxe Markdown, y compris :

✅ titres
✅ listes
✅ blocs de code
✅ icônes émojis
✅ captures d’écran (emplacements)

Tu peux copier **tel quel** dans ton README.md :

---

````markdown
## ✅ Installation rapide

```bash
git clone https://github.com/ton-utilisateur/credit_scoring_project.git
cd credit_scoring_project
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
````

---

## 📸 Captures d'écran


### Page d'accueil

(screenshots/home.png)

### Résultat SHAP

(screenshots/shap.png)

---


## 📬 Auteur

👤 **Hamza DRAMANE**
🎓 Master Intelligence Artificielle & Big Data
📍 Basé au Bénin | Passionné de Finance & IA

🔗 https://www.linkedin.com/in/hamzath-dramane-06680427b?

    https://github.com/Anjorin007

---

## 🚀 Prochaines étapes

* 🌍 Intégrer une cartographie régionale de vulnérabilité (open data)
* ☁️ Déploiement cloud (Render, Hugging Face Spaces…)
* 🛡️ Intégration d’une API sécurisée pour version SaaS
* 📱 Version mobile simplifiée pour agents de crédit terrain

---

## 💼 Utilisation potentielle

* ✔️ Outil de scoring interne
* ✔️ App pour agents de terrain
* ✔️ Prototype SaaS B2B à présenter aux institutions financières

---

## 📜 Licence

Ce projet est sous licence MIT – voir le fichier LICENSE pour plus de détails.

---

## ✅ Fonctionnalités supplémentaires incluses

* **Badges dynamiques** pour les technologies
* **Images d'exemple** (remplacez par vos propres captures)
* **Tableau des performances** clair
* **Instructions Docker** pour le déploiement
* **Structure de projet** visualisable
* **Liens cliquables** pour les contacts

---

## 🔧 Pour personnaliser

* Remplacez les liens GitHub, LinkedIn et email
* Ajoutez vos propres captures dans le dossier `/screenshots`
* Modifiez les métriques si nécessaire
* Ajoutez votre propre fichier LICENSE

---

✅ **Ce template est :**

* Professionnel
* Bien structuré
* Visuellement attractif
* Facilement personnalisable
* Compatible GitHub/GitLab

```
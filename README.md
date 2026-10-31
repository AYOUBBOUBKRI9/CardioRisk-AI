# 💜 CardioRisk AI — Prédiction du risque cardiovasculaire

**CardioRisk AI** est une application web interactive développée avec Python et Streamlit. Elle exploite des techniques de Machine Learning pour estimer le risque de maladie coronarienne à partir de différents facteurs médicaux et comportementaux.

Le projet combine prédiction, analyse des facteurs de risque et visualisation des données dans une interface moderne.

**Réalisé par : Mohamed Ayoub Boubkri**

---

## 🎯 Objectifs du projet

- Estimer le risque cardiovasculaire à partir des caractéristiques d'un individu.
- Analyser les principaux facteurs associés au risque coronarien.
- Proposer une interface interactive pour explorer les données.
- Illustrer l'utilisation du Machine Learning dans un contexte de santé.

## ✨ Fonctionnalités

- **Profil biomédical :** saisie des caractéristiques du patient.
- **Prédiction :** estimation du risque à partir du modèle entraîné.
- **Analyse des facteurs de risque :** exploration des variables cardiovasculaires.
- **Visualisations interactives :** représentation graphique des données.
- **Simulation temporelle :** exploration de scénarios d'évolution.
- **Analyse comparative :** comparaison de différents profils.
- **Rapport :** génération d'un rapport personnalisé.

## 🧬 Données utilisées

Le projet utilise le dataset `CHD.csv`, consacré à l'étude des maladies coronariennes.

Les variables exploitées comprennent notamment :

| Variable | Description |
|---|---|
| Age | Âge |
| SBP | Pression artérielle systolique |
| Tobacco | Consommation de tabac |
| LDL | Cholestérol LDL |
| Adiposity | Adiposité |
| Famhist | Antécédents familiaux |
| TypeA | Comportement de type A |
| Obesity | Obésité |
| Alcohol | Consommation d'alcool |
| CHD | Présence d'une maladie coronarienne |

## 🛠️ Technologies utilisées

- **Python** — Langage de programmation
- **Pandas & NumPy** — Manipulation et traitement des données
- **Scikit-learn** — Machine Learning et prétraitement
- **Imbalanced-learn** — Gestion du déséquilibre des classes
- **Joblib** — Chargement du modèle sauvegardé
- **Plotly** — Visualisations interactives
- **Streamlit** — Développement de l'application web

## 🧠 Machine Learning

Le projet s'appuie sur un modèle sauvegardé dans `Model.pkl`.

Le notebook `miniproject-vfinal.ipynb` présente les différentes étapes de préparation des données et d'expérimentation des modèles.

## 🚀 Installation et exécution

### 1. Cloner le dépôt

```bash
git clone https://github.com/AYOUBBOUBKRI9/CardioRisk-AI.git
cd CardioRisk-AI
```

### 2. Créer un environnement virtuel

```bash
python -m venv .venv
```

### 3. Activer l'environnement

Sous Windows :

```powershell
.venv\Scripts\activate
```

### 4. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 5. Lancer l'application

```bash
streamlit run app.py
```

L'application sera accessible à l'adresse :

`http://localhost:8501`

**Environnement recommandé :** Python 3.11 avec les versions de bibliothèques indiquées dans `requirements.txt`.

## 📁 Structure du projet

```text
CardioRisk-AI/
│
├── app.py
├── Model.pkl
├── CHD.csv
├── heart.png
├── miniproject-vfinal.ipynb
├── requirements.txt
└── README.md
```

## 👨‍💻 Auteur

**Mohamed Ayoub Boubkri**

Étudiant en Master Informatique — Gouvernance et Transformation Digitale.

Spécialité : Gouvernance Numérique et Ingénierie de Données.

- GitHub : [AYOUBBOUBKRI9](https://github.com/AYOUBBOUBKRI9)
- LinkedIn : [Mohamed Ayoub Boubkri](https://www.linkedin.com/in/mohamed-ayoub-boubkri-167aba392/)

## ⚠️ Avertissement médical

CardioRisk AI est un prototype académique développé à des fins pédagogiques et expérimentales. Il ne constitue pas un dispositif médical, ne fournit pas de diagnostic et ne remplace pas l'avis d'un professionnel de santé.

Les performances du modèle doivent être évaluées sur des données indépendantes avant toute utilisation dans un contexte réel.

---

*Développé avec Python, Machine Learning et Streamlit.* 💜

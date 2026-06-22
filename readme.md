# Dashboard TP Deep Learning

## Installation & lancement

```bash
pip install streamlit pandas numpy scikit-learn pillow
streamlit run dashboard.py
```

## Structure attendue

Placez le fichier `dashboard.py` à la **racine** du dossier `NOSSU/` :

```
NOSSU/
├── dashboard.py          ← ce fichier
├── partie1/
│   ├── data/bank-additional-full.csv
│   └── outputs/{figures,models,reports}/
├── partie2/
│   └── outputs/{figures,models,reports}/
└── partie3/
    └── outputs/{figures,models,reports}/
```

## Fonctionnalités

| Page | Contenu |
|------|---------|
| Accueil | KPIs globaux, structure projet |
| Partie 1 — EDA | Dataset stats, distributions, corrélations, scatter matrix |
| Partie 1 — Modèles & CV | Comparaison sklearn, arbre de décision optimisé |
| Partie 1 — Ensembles & ROC | Bagging, RF, Boosting, feature importances, ROC/AUC |
| Partie 2 — ANN | Architecture Keras, courbes apprentissage, résultats |
| Partie 3 — Deep Learning | Fashion-MNIST, comparaison architectures |
| Inférence | Formulaire manuel + batch CSV, tous modèles .pkl |

## Notes

- Les figures sont chargées directement depuis `outputs/figures/`
- Les modèles `.pkl` sont chargés depuis `outputs/models/`
- L'inférence aligne automatiquement les features via `feature_names.pkl` et `scaler.pkl`
- Le modèle Keras (`best_ann.keras`) nécessite TensorFlow : `pip install tensorflow`